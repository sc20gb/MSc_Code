import torch
import torch.nn as nn
import os
import gc


from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint


import psutil
def print_memory_usage():
    # Print CPU memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_memory_rss = mem_info.rss / (1024 ** 3)  # Resident Set Size (RSS) in GB
    cpu_memory_vms = mem_info.vms / (1024 ** 3)  # Virtual Memory Size (VMS) in GB
    print(f"CPU Memory Usage - VMS: {cpu_memory_vms:.2f} GB")



class _NetCheckpointWrapper(nn.Module):
    def __init__(self, net):
        super(_NetCheckpointWrapper, self).__init__()  # Initialize nn.Module
        self.net = net  # The network to wrap

    def forward(self, x):
        # Call the wrapped network using the inputs_embeds argument
        return self.net(inputs_embeds=x)
    

    def get_input_embeddings(self):
        return self.net.get_input_embeddings()

class Connector_LLM(nn.Module):
    def __init__(self, embed_dim, connector_layers,vicuna_path,device, MAX_LENGTH,accumulation_steps=-1):
        super(Connector_LLM, self).__init__()
        layers = []
        input_dim = embed_dim

        self.device = device

        self.accumulation_steps = accumulation_steps

        self.MAX_LENGTH = MAX_LENGTH

        vicuna,self.tokenizer = self.load_vicuna(vicuna_path,device)

        #vicuna.gradient_checkpointing_enable()

        self.w_vicuna = _NetCheckpointWrapper(vicuna)

        self.w_vicuna.eval()
        with torch.no_grad():
            embedding_size = self.w_vicuna.get_input_embeddings()(torch.tensor([0],dtype=torch.int64,device=device)).size(1)
        self.w_vicuna.train()

        # Create the specified number of layers
        for _ in range(connector_layers - 1):
            layers.append(nn.Linear(input_dim, embedding_size))
            layers.append(nn.GELU())
            input_dim = embedding_size

        # Add the final output layer
        layers.append(nn.Linear(embedding_size, embedding_size))

        # Build the Sequential model
        self.connector = nn.Sequential(*layers).to(device)

        self._initialize_weights()

        self.vicuna_path = vicuna_path


        self.attributes_to_delete = []


    def freeze_weights_for_PEFT(self):
            print("Named Parameters in Vicuna:")
            for parameter, _ in self.w_vicuna.named_parameters():
                print(parameter)
            print("End of named parameters")


        

    def _initialize_weights(self):
            for m in self.connector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # Initialize biases to zero


    def load_vicuna(self,model_path,device):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True, padding_side='left')

        model = LlamaForCausalLM.from_pretrained(os.path.join(model_path, "model")).to(device)

        return model, tokenizer
    
    def set_optim_scheduler(self,optim,scheduler):

        self.optim = optim

        self.scheduler = scheduler
    
    def filter_prompt(self,prompt):
        # Obtain the <UNK> token ID
        unk_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        
        # Get the valid token IDs from the tokenizer's vocabulary
        valid_token_ids = set(self.tokenizer.get_vocab().values())
        
        # Create a mask for valid tokens
        valid_token_mask = torch.tensor([[token_id in valid_token_ids for token_id in seq] for seq in prompt], dtype=torch.bool).to(prompt.device)
        
        # Replace invalid tokens with <UNK> token ID
        prompt_with_unk = torch.where(valid_token_mask, prompt, torch.tensor(unk_token_id, dtype=prompt.dtype, device=prompt.device))
        
        # No need to remove zero values (padding); keep them as they are.
        
        # Pad sequences to the same length
        if prompt_with_unk.size(1) > 0:  # Ensure there is data to pad
            filtered_prompt = torch.nn.utils.rnn.pad_sequence(
                [seq for seq in prompt_with_unk], 
                batch_first=True, 
                padding_value=0
            )
        else:
            # Handle the case where there might be no sequences at all
            filtered_prompt = torch.empty((prompt.size(0), 0), dtype=torch.long, device=prompt.device)
        
        return filtered_prompt
    
    def generate_attention(self,embeddings):
        causal_mask = torch.tril(torch.ones((embeddings.size(0), embeddings.size(0)))).bool().unsqueeze(0).to(self.device)
        return causal_mask

    def update_attention(self, size):
        return torch.tril(torch.ones((size, size),device=self.device)).bool().unsqueeze(0)


    def check_grad(self,t,strv):
        if not t.requires_grad:
            print("NO GRAD FOR ", strv)
        else:
            print("Grad FOR ", strv)
        

    def generate_using_forward_method(self, max_length, temperature, target, question, image_features, itr):
        # Project to LLM embedding space
        if torch.is_grad_enabled():
            image_features.requires_grad_()
            image_features = self.connector(image_features)
        else:
            image_features = self.connector(image_features)

        # Encode text and images into the embedding expected by the LLM
        if torch.is_grad_enabled():
            embeddings = self.encode_text_and_image(question, image_features)
        else:
            embeddings = self.encode_text_and_image(question, image_features)

        # Ensure embeddings have a batch dimension
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension if missing

        log_probs_sum = torch.tensor([0.0], device=self.device, requires_grad=True)
        count = 0
        gen_embeddings = embeddings
        gen_tokens = []

        # Autoregressive generation loop
        for i in range(max_length):
            # if vicuna does not need training save mem
            if not self.w_vicuna.training:
                with torch.no_grad():
                    outputs = self.w_vicuna(gen_embeddings)
            else:
                if torch.is_grad_enabled():
                    outputs = self.w_vicuna(gen_embeddings)
                else:
                    outputs = self.w_vicuna(gen_embeddings)

            temperature_ = torch.tensor(temperature, device=self.device).half()

            # Get the logits of the last token and apply temperature scaling
            new_tokens = outputs.logits[:, -1, :] / temperature_

            # Generate the loss for the model based on the answer
            if target is not None:
                index = torch.tensor([i for _ in range(target.size(0))], device=self.device)
                selected_values = target[torch.arange(target.size(0), device=self.device), index].unsqueeze(1)
                log_probs = torch.nn.functional.log_softmax(new_tokens, dim=1).half()
                log_probs_for_target = log_probs.gather(1, selected_values.to(torch.int64))
                log_probs_sum = log_probs_sum + log_probs_for_target.sum()
                count += 1

            # Apply softmax
            prob_logits = torch.nn.functional.softmax(new_tokens, dim=1)

            # Sample from the distribution to get the next token
            next_token_ids = torch.argmax(prob_logits, dim=1)
            gen_tokens.append(next_token_ids)
            next_embedding = self.w_vicuna.get_input_embeddings()(next_token_ids)
            next_embedding = next_embedding.unsqueeze(1)
           
            next_embedding.requires_grad_()

            gen_embeddings = torch.cat((gen_embeddings, next_embedding), dim=1).clone()

            # Check output token for EOS if batch size is just one
            if next_embedding.size()[0] < 2:
                if next_token_ids[0] == self.tokenizer.eos_token_id:
                    break

            # Return the generated tokens and the averaged negative log-likelihood (NLL loss)        
        
        # Return the generated tokens and the loss
        nll_loss = -log_probs_sum / float(count)

        if torch.is_grad_enabled():
            nll_loss.backward()
            if not self.accumulation_steps < 1:
                if ((itr + 1) % self.accumulation_steps == 0):
                    self.optim.step()
                    self.optim.zero_grad()
                    if self.w_vicuna.training:
                        self.w_vicuna.zero_grad()  # Clear all gradients
                    self.connector.zero_grad()


        final_output = torch.cat(gen_tokens)

        if final_output.requires_grad:
            final_output = final_output.detach()


        return torch.cat(gen_tokens), nll_loss.cpu().item()

    #This function takes the feature and question embeddings and combines them in the correct embedding format
    #It also embeds the text
    def encode_text_and_image(self, question, image_features):
        # Initialize lists for split ids and text embeddings
        split_ids = []

        # Tokenize all text segments across the batch
        tokenized_list = []
        for q in question:
            # Tokenize and prepare text
            tokens = self.tokenizer("Image: ").input_ids
            split_ids.append(len(tokens))
            tokens.extend(self.tokenizer(" Question: " + q + " Answer: ").input_ids[1:])
            tokenized_list.append(torch.tensor(tokens, dtype=torch.int64, device=self.device))

        # Embed all text tokens
        embedded_text = []
        for tokens in tokenized_list:
            embedded_text.append(self.w_vicuna.get_input_embeddings()(tokens))

        # Adding image embeddings
        for i in range(len(image_features)):
            # Split the embedded text at the appropriate index
            s1 = embedded_text[i][:split_ids[i]]
            s2 = embedded_text[i][split_ids[i]:]

            # Concatenate image features with embedded text
            combined_embeddings = torch.cat((s1, image_features[i], s2), dim=0)
            embedded_text[i] = combined_embeddings

        # Concatenate embeddings across batches
        #embeddings = torch.stack(embedded_text, dim=0)

        self.attributes_to_delete.extend([split_ids,tokenized_list])

        return embedded_text[0]


    def forward(self, image_features, question, answer, max_length, itr):

        #batch_size, n_patches, *feature_dims = image_features.shape

        # Reshape image features to merge the batch and 17 dimensions
        #image_features = image_features.view(batch_size * n_patches, *feature_dims)


        # Autoregressive prediction
        # Ensure no unnecessary intermediate results are kept
        gen, loss = self.generate_using_forward_method(max_length, 0.9, answer,question,image_features, itr)

        torch.cuda.empty_cache()  # Clear the CUDA cache

        return gen, loss


    def delete_non_weight_vars(self):
        # Iterate through all attributes of the class to delete
        for attr_name in self.attributes_to_delete:
            del attr_name

        # Optionally, clear unused memory
        torch.cuda.empty_cache()  # If using CUDA, clear unused GPU memory
