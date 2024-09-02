

# TODO: combine validationand train set

#TODO: add short question request as well

#TODO: we need to create the function that gets the final model results

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
    def __init__(self, embed_dim, connector_width, connector_output, connector_layers,vicuna_path,device, MAX_LENGTH):
        super(Connector_LLM, self).__init__()
        layers = []
        input_dim = embed_dim

        self.device = device

        self.MAX_LENGTH = MAX_LENGTH

        # Create the specified number of layers
        for _ in range(connector_layers - 1):
            layers.append(nn.Linear(input_dim, connector_width))
            layers.append(nn.GELU())
            input_dim = connector_width

        # Add the final output layer
        layers.append(nn.Linear(connector_width, connector_output))

        # Build the Sequential model
        self.connector = nn.Sequential(*layers).to(device)

        self._initialize_weights()

        self.vicuna,self.tokenizer = self.load_vicuna(vicuna_path,device)

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
        
    def generate_using_forward_method(self, filtered_prompt, max_length=50, temperature=1.0, target=None):
        batch_size = filtered_prompt.size(0)
        seq_len = filtered_prompt.size(1)

        # Preallocate tensors with the maximum possible size
        generated_ids = filtered_prompt#torch.zeros((batch_size, seq_len + max_length), dtype=torch.long, device=filtered_prompt.device)
        #generated_ids[:, :seq_len] = filtered_prompt

        loss_sum  = 0.0

        # Autoregressive generation loop
        for i in range(max_length):

            if not self.vicuna.training:
                with torch.no_grad():
                    outputs = self.vicuna(generated_ids)
            else:
                outputs = self.vicuna(generated_ids)

            new_tokens = outputs.logits[:, -1, :]

            # Generate the loss for the model based on the answer
            if target  !=  None:
                index = torch.tensor([i for _ in range(target.size(0))])  # Example indices to extract

                # Use advanced indexing to select values from A
                selected_values = target[torch.arange(target.size(0)), index].unsqueeze(1)

                loss_sum += F.cross_entropy(new_tokens.clone(),selected_values.flatten()).item()

            # Get the logits of the last token and apply temperature scaling
            logits = new_tokens  / temperature
            logits = torch.nn.functional.softmax(logits, dim=1)

            # Sample from the distribution to get the next token
            next_token_ids = torch.argmax(logits, dim=-1)

            # Append the generated token to the input IDs
            if generated_ids.size()[0] < 2:
                generated_ids = torch.cat([generated_ids, next_token_ids.unsqueeze(0)], dim=1)
            else:
                generated_ids = torch.cat([generated_ids, next_token_ids.unsqueeze(1)], dim=1)

            # check output token for eos if batch size is just one

            if (generated_ids.size()[0] < 2):
                if next_token_ids[0] == self.tokenizer.eos_token_id:
                    break

        return generated_ids, torch.tensor(loss_sum/(float(max_length)), requires_grad=True)

    def forward(self, image_features,question,answer,max_length):
        
        # prject to joint space
        image_embedding_tokenized = self.connector(image_features)

        #tokenization of the embedding sapce means < 0 is not allowed
        image_embedding_tokenized = torch.where(image_embedding_tokenized < 0, torch.tensor(0), image_embedding_tokenized)

        

        # decode from joint space
        image_prompt = self.tokenizer.batch_decode(image_embedding_tokenized.long())

        #image_prompt = torch.tensor(image_prompt, dtype=torch.long)
        #if image_features.size(0) < 2:
                    #prompt = torch.cat([self.tokenizer( "<s>" + image_prompt[i] + " " + a,return_tensors="pt").input_ids for i, a in enumerate(question)],0)[:, 1:].to(self.device)
        #else:
        prompt = torch.cat([self.tokenizer( "<s>" + image_prompt[i] + " " + a,return_tensors="pt",padding='max_length', max_length = self.MAX_LENGTH).input_ids for i, a in enumerate(question)],0)[:, 1:].to(self.device)


        gen,loss = self.generate_using_forward_method(prompt,target=answer,max_length=max_length,temperature=0.9)
      
        return gen, loss