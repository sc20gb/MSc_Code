import torch
import torch.nn as nn
import os
import gc


from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F

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
    
    def generate_attention(self,embeddings,text_list):
        causal_mask = torch.tril(torch.ones((embeddings.size(0), embeddings.size(0)))).bool().unsqueeze(0).to(self.device)
        return causal_mask

    def update_attention(self,attention,size):
        return torch.tril(torch.ones((size, size),device=self.device)).bool().unsqueeze(0)

    def generate_using_forward_method(self, embeddings,attention_mask, max_length=50, temperature=1.0, target=None):
        print("In forward gen")
        log_probs_sum = 0.0

        count = 0

        gen_embeddings = embeddings

        gen_tokens = []

        # Autoregressive generation loop
        for i in range(max_length):

            # if vicuna does not need traning save mem

            if not self.vicuna.training:
                with torch.no_grad():
                    outputs = self.vicuna(inputs_embeds=gen_embeddings,attention_mask=attention_mask)
            else:
                outputs = self.vicuna(inputs_embeds=gen_embeddings,attention_mask=attention_mask)


            # Get the logits of the last token and apply temperature scaling
            new_tokens = outputs.logits[:, -1, :] / temperature

            # Generate the loss for the model based on the answer
            if target  !=  None:
                index = torch.tensor([i for _ in range(target.size(0))], device=self.device)

                # # Use advanced indexing to select values from A
                # selected_values = target[torch.arange(target.size(0),device=self.device), index].unsqueeze(1)

                # loss_sum += F.cross_entropy(new_tokens.clone(),selected_values.flatten()).item()


                # Select the correct target token for the current position
                selected_values = target[torch.arange(target.size(0), device=self.device), index].unsqueeze(1)

                # Calculate the log-likelihood for the selected token
                log_probs = torch.nn.functional.log_softmax(new_tokens, dim=1)

                #This will be correct as the ;pg_probs is taken from new tokens which is just the next generated probs
                #It takes the log probabilities for the target
                log_probs_for_target = log_probs.gather(1, selected_values)

                # Accumulate the log likelihood
                log_probs_sum += log_probs_for_target.sum().item()
                count += 1

            #Apply softmax
            prob_logits = torch.nn.functional.softmax(new_tokens, dim=1)

            # Sample from the distribution to get the next token
            next_token_ids = torch.argmax(prob_logits, dim=1)

            gen_tokens.append(next_token_ids)
            
            next_embedding = self.vicuna.get_input_embeddings()(next_token_ids)

            gen_embeddings = torch.cat((gen_embeddings,next_embedding.unsqueeze(1)),dim=1)

            # check output token for eos if batch size is just one
            if (next_embedding.size()[0] < 2):
                if next_token_ids[0] == self.tokenizer.eos_token_id:
                    break
        
            attention_mask = self.update_attention(attention_mask,gen_embeddings.size(1))

            # Return the generated tokens and the averaged negative log-likelihood (NLL loss)
            nll_loss = -log_probs_sum / float(count)  # Maximize likelihood by minimizing negative log-likelihood

        #return the generated tokens and the loss

        print("at end of forward gen")
        return torch.cat(gen_tokens, device=self.device), torch.tensor(nll_loss, requires_grad=True,device=self.device)

    #This function takes the feature and question embeddings and combines them in the correct embedding format
    #It also embeds the text
    def encode_text_and_image(self, question,image_features):

        #So we know where to put the img
        split_ids = []

        # so we know which are which embedding
        text_list = []

        #tokenise all text segments across the batch
        tokenised_list = []
        for i, q in enumerate(question):
            tokenised_list.append([])
            tokenised_list[i].extend(self.tokenizer("Image: ").input_ids)
            split_ids.append(len(tokenised_list[i]))
            tokenised_list[i].extend(self.tokenizer(" Question: " + q + " Answer: ").input_ids[1:])
            tokenised_list[i] = torch.tensor(tokenised_list[i],dtype=torch.int64).to(self.device)
            
        embedded_text = []
        #Embed all of the text tokens
        for b in tokenised_list:
                embedded_text.append(self.vicuna.get_input_embeddings()(b))

        # adding image embeddings
        for i in range(image_features.size(0)):
            # insert the image feature at the point shown in split_ids

            text_list.append([])
            embedded_text[i]
            s1 = embedded_text[i][:split_ids[i],:]
            s2 = embedded_text[i][split_ids[i]:,:]

            for token in s1: text_list[i].append(1)

            for token in image_features[0]: text_list.append(0)

            for token in s2: text_list[i].append(1)

            embedded_text[i] = torch.cat((s1,image_features.squeeze(0),s2), dim=0)


        embeddings = torch.cat(embedded_text, dim=-1)

        return embeddings, text_list

    def forward(self, image_features,question,answer,max_length):


        batch_size, n_patches, *feature_dims = image_features.shape # 1,1,*

        # Reshape image features to merge the batch and 17 dimensions
        image_features = image_features.view(batch_size * n_patches, *feature_dims) # 1, *

        # project to LLM embedding space
        image_features = self.connector(image_features)

        # Reshape back to original dimensions after projection
        image_features = image_features.view(batch_size, n_patches, -1)

        # Encode text and images into the embedding expected by the LLM
        embeddings, text_list = self.encode_text_and_image(question,image_features)

        # Generate the attention mask
        attention_mask = self.generate_attention(embeddings,text_list)

        #This passes the embeddings to the LLM
        embeddings =  embeddings.to(self.device)

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # Add batch dimension if missing

        #Autoregressive prediction
        gen,loss = self.generate_using_forward_method(embeddings,attention_mask,target=answer,max_length=max_length,temperature=0.9)

        return gen, loss