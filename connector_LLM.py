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


class Connector_LLM(nn.Module):
    def __init__(self, embed_dim, connector_layers,vicuna_path,device, MAX_LENGTH):
        super(Connector_LLM, self).__init__()
        layers = []
        input_dim = embed_dim

        self.device = device

        self.MAX_LENGTH = MAX_LENGTH

        self.vicuna,self.tokenizer = self.load_vicuna(vicuna_path,device)

        self.vicuna.eval()
        with torch.no_grad():
            embedding_size = self.vicuna.get_input_embeddings()(torch.tensor([0],dtype=torch.int64,device=device)).size(1)
        self.vicuna.train()

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


    def wrapper_vicuna_forward(self,gen_embeddings,attention_mask):
        # this is a wrapper for the forwards method in vicuna so the checkpoint method can be used
        outputs = self.vicuna(inputs_embeds=gen_embeddings,attention_mask=attention_mask)
        return outputs

    def generate_using_forward_method(self, embeddings,attention_mask, max_length, temperature, target):

        print(f"Memory allocated after gen start: {torch.cuda.memory_allocated() / 1e6} MB")
        log_probs_sum = 0.0

        count = 0

        gen_embeddings = embeddings

        gen_tokens = []


        # Autoregressive generation loop
        for i in range(max_length):
            # if vicuna does not need traning save mem
            print(f"Memory allocated after gen loop start: {torch.cuda.memory_allocated() / 1e6} MB")

            if not self.vicuna.training:
                with torch.no_grad():
                    outputs = self.vicuna(inputs_embeds=gen_embeddings,attention_mask=attention_mask)
            else:
                outputs = checkpoint(self.wrapper_vicuna_forward,gen_embeddings,attention_mask)


            print(f"Memory allocated after vicuna: {torch.cuda.memory_allocated() / 1e6} MB")

            #outputs.logits.requires_grad_()


            print(f"Memory allocated after outputs requires grad: {torch.cuda.memory_allocated() / 1e6} MB")

            # Get the logits of the last token and apply temperature scaling
            new_tokens = outputs.logits[:, -1, :] / temperature


            #del outputs


            print(f"Memory allocated after new_tokens: {torch.cuda.memory_allocated() / 1e6} MB")

            # Generate the loss for the model based on the answer
            if target  !=  None:
                index = torch.tensor([i for _ in range(target.size(0))], device=self.device)

                # # Use advanced indexing to select values from A
                # selected_values = target[torch.arange(target.size(0),device=self.device), index].unsqueeze(1)

                # loss_sum += F.cross_entropy(new_tokens.clone(),selected_values.flatten()).item()


                # Select the correct target token for the current position
                selected_values = target[torch.arange(target.size(0), device=self.device), index].unsqueeze(1)
                
                # Calculate the log-likelihood for the selected token
                log_probs = torch.nn.functional.log_softmax(new_tokens, dim=1).half()

                #This will be correct as the ;pg_probs is taken from new tokens which is just the next generated probs
                #It takes the log probabilities for the target

                log_probs_for_target = log_probs.gather(1, selected_values.to(torch.int64))

                # Accumulate the log likelihood
                log_probs_sum += log_probs_for_target.sum()

                count += 1
            

            print(f"Memory allocated after loss calc: {torch.cuda.memory_allocated() / 1e6} MB")

            #Apply softmax
            prob_logits = torch.nn.functional.softmax(new_tokens, dim=1)

            # Sample from the distribution to get the next token
            next_token_ids = torch.argmax(prob_logits, dim=1)

            gen_tokens.append(next_token_ids)
            
            next_embedding = self.vicuna.get_input_embeddings()(next_token_ids)

            print(f"Memory allocated after vicuna get input embeddings: {torch.cuda.memory_allocated() / 1e6} MB")

            gen_embeddings = torch.cat((gen_embeddings,next_embedding.unsqueeze(1)),dim=1)

            # check output token for eos if batch size is just one
            if (next_embedding.size()[0] < 2):
                if next_token_ids[0] == self.tokenizer.eos_token_id:
                    break

            print(f"Memory allocated after cating embeddings: {torch.cuda.memory_allocated() / 1e6} MB")

            self.attributes_to_delete.append(attention_mask)
        
            attention_mask = self.update_attention(gen_embeddings.size(1))

            print(f"Memory allocated after updating the attention mask: {torch.cuda.memory_allocated() / 1e6} MB")


            # Return the generated tokens and the averaged negative log-likelihood (NLL loss)
            nll_loss = -log_probs_sum / float(count)  # Maximize likelihood by minimizing negative log-likelihood

            print(f"Memory allocated after nll_loss and itr end: {torch.cuda.memory_allocated() / 1e6} MB")

            self.attributes_to_delete.extend([new_tokens])

        #return the generated tokens and the loss
        print(f"Memory allocated after gen function: {torch.cuda.memory_allocated() / 1e6} MB")


        nll_loss = -log_probs_sum / float(count)
        nll_loss.backward()

        self.attributes_to_delete.append(gen_embeddings)

        return torch.cat(gen_tokens)

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
            embedded_text.append(self.vicuna.get_input_embeddings()(tokens))

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


    def forward(self, image_features, question, answer, max_length):


        #To save memory at the cost of more computation
        #checkpoint()


        #batch_size, n_patches, *feature_dims = image_features.shape

        # Reshape image features to merge the batch and 17 dimensions
        #image_features = image_features.view(batch_size * n_patches, *feature_dims)
        print(f"Memory allocated after forward start: {torch.cuda.memory_allocated() / 1e6} MB")


        # Project to LLM embedding space
        image_features = checkpoint(self.connector,image_features)

        print(f"Memory allocated after connector: {torch.cuda.memory_allocated() / 1e6} MB")

        # Reshape back to original dimensions after projection
        #image_features = image_features.view(batch_size, n_patches, -1)

        # Encode text and images into the embedding expected by the LLM
        embeddings = checkpoint(self.encode_text_and_image,question, image_features)

        del image_features

        print(f"Memory allocated after encoding text: {torch.cuda.memory_allocated() / 1e6} MB")

        # Generate the attention mask
        attention_mask = checkpoint(self.generate_attention,embeddings)


        print(f"Memory allocated after generating attention: {torch.cuda.memory_allocated() / 1e6} MB")

        # Move embeddings to the device
        embeddings = embeddings.to(self.device)


        print(f"Memory allocated after embedings to device: {torch.cuda.memory_allocated() / 1e6} MB")

        # Ensure embeddings have a batch dimension
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension if missing

        print(f"Memory allocated after dim change: {torch.cuda.memory_allocated() / 1e6} MB")

        # Autoregressive prediction
        # Ensure no unnecessary intermediate results are kept
        gen = self.generate_using_forward_method(embeddings, attention_mask, max_length, 0.9, answer)


        print(f"Memory allocated after generate: {torch.cuda.memory_allocated() / 1e6} MB")

        # Clear any unused variables to free up memory when requested
        self.attributes_to_delete.extend([embeddings,attention_mask,gen])

        torch.cuda.empty_cache()  # Clear the CUDA cache


        print(f"Memory allocated after emptying cache and deleting variables: {torch.cuda.memory_allocated() / 1e6} MB")

        return gen


    def delete_non_weight_vars(self):
        # Iterate through all attributes of the class to delete
        for attr_name in self.attributes_to_delete:
            del attr_name

        # Optionally, clear unused memory
        torch.cuda.empty_cache()  # If using CUDA, clear unused GPU memory
