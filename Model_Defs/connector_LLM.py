import torch
import torch.nn as nn
import os
import gc


from transformers import LlamaForCausalLM, AutoTokenizer
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from peft import get_peft_model, LoraConfig

import psutil
def print_memory_usage():
    # Print CPU memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_memory_rss = mem_info.rss / (1024 ** 3)  # Resident Set Size (RSS) in GB
    cpu_memory_vms = mem_info.vms / (1024 ** 3)  # Virtual Memory Size (VMS) in GB
    print(f"Memory cuda: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"CPU Memory Usage - VMS: {cpu_memory_vms:.2f} GB")


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype

        print("Type if input to LayerNorm = ", orig_type)
        ret = super().forward(x.type(torch.float32))
        print("ret type:", ret.dtype)

        ret = ret.type(orig_type)

        print("ret type:", ret.dtype)
        
        return ret

class Connector_LLM(nn.Module):
    def __init__(self, embed_dim, connector_layers,vicuna_path,device,accumulation_steps=-1, seed=42,norm=False):
        super(Connector_LLM, self).__init__()
        layers = []
        input_dim = embed_dim

        self.device = device

        self.accumulation_steps = accumulation_steps

        self.vicuna,self.tokenizer = self.load_vicuna(vicuna_path,device)

        self.vicuna.eval()
        with torch.no_grad():
            self.embedding_size = self.vicuna.get_input_embeddings()(torch.tensor([0],dtype=torch.int64,device=device)).size(1)
        self.vicuna.train()

        # Create the specified number of layers
        for _ in range(connector_layers - 1):
            layers.append(nn.Linear(input_dim, self.embedding_size))
            layers.append(nn.GELU())
            input_dim = self.embedding_size

        # Add the final output layer
        layers.append(nn.Linear(self.embedding_size, self.embedding_size))
        
        if norm:
            layers.append(LayerNorm(self.embedding_size))

        # Build the Sequential model
        self.connector = nn.Sequential(*layers).to(device)

        self._initialize_weights(seed)

        self.vicuna_path = vicuna_path

    #For training only attention projections
    def freeze_weights_for_PEFT(self):
            print("Named Parameters in Vicuna:")
            for name, param in self.vicuna.named_parameters():
                str = name
                param.requires_grad = False
                # Unfreeze the projection weights for q, k, v, o in self-attention layers
                if any(proj in name for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]):
                    param.requires_grad = True
                    str = str + "GRAD TRUE"
                print(str)
            print("End of named parameters")  
    
    #Applies lora to the vicuna model
    def apply_lora(self,rank=8,alpha=32,modules=["q_proj", "v_proj"],dropout=0.1):
        # Define the LoRA configuration
        lora_config = LoraConfig(
            r=rank,   # Rank of the low-rank adaptation
            lora_alpha=alpha,
            target_modules=modules,  # Modules to apply LoRA to
            lora_dropout=dropout
        )

        # Wrap your model with LoRA
        self.vicuna = get_peft_model(self.vicuna, lora_config)

    #Initilises the weights for the MLP connector (visual embedder)
    def _initialize_weights(self, seed):
            
            if seed is not None:
                torch.manual_seed(seed)  # Set seed for reproducibility
    
            for m in self.connector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # Initialize biases to zero

    # Loads pretrained weights into the vicuna model
    def load_vicuna(self,model_path,device):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True, padding_side='left')

        model = LlamaForCausalLM.from_pretrained(os.path.join(model_path, "model")).to(device)

        return model, tokenizer
    
    #A function for setting the optimiser and scheduler used in training during generate_using_forward_method
    def set_optim_scheduler(self,optim,scheduler):

        self.optim = optim

        self.scheduler = scheduler
    
    #A debuggging function to check the gradents of diffrent layers
    def check_grad(self,t,strv):
        if not t.requires_grad:
            print("NO GRAD FOR ", strv, t.size(), t.device)
        else:
            print("Grad FOR ", strv, t.size(), t.device)
    
    #Auto-regresivly generates the output of the vicuna-LLM. Also generates loss and performs backwards()
    def generate_using_forward_method(self, max_length, temperature, target, question, image_features, itr):

        #Deal with dfiffrent dims
        image_features = image_features.squeeze()
        batch_size, n_patches, *feature_dims = image_features.shape

        if image_features.dim() < 3:
            # Add a batch dimension at the front
            image_features = image_features.unsqueeze(0)  # Adds a dimension of size 1 at index 0

        image_features = image_features.view(batch_size * n_patches, *feature_dims)


        print("Type of Image features in = ", image_features.dtype)
        
        #embed the image features
        image_features = self.connector(image_features)

        print("Type of image_features after connector = ", image_features.dtype)

        image_features = image_features.view(batch_size,n_patches,-1)

        print("Type of image_features after view = ", image_features.dtype)

        # Encode text and images into the embedding expected by the LLM
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
            if not self.vicuna.training:
                with torch.no_grad():
                    outputs = self.vicuna(inputs_embeds=gen_embeddings)
            else:
                outputs = self.vicuna(inputs_embeds=gen_embeddings)
  
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
            next_embedding = self.vicuna.get_input_embeddings()(next_token_ids)
            next_embedding = next_embedding.unsqueeze(1)
           
            next_embedding.requires_grad_()

            gen_embeddings = torch.cat((gen_embeddings, next_embedding), dim=1).clone()

            # Check output token for EOS if batch size is just one
            if next_embedding.size()[0] < 2:
                if next_token_ids[0] == self.tokenizer.eos_token_id:
                    break
        
        # Return the generated tokens and the loss
        nll_loss = -log_probs_sum / float(count)

        if torch.is_grad_enabled():
            nll_loss.backward()
            print_memory_usage()
            if not self.accumulation_steps < 1:
                if ((itr + 1) % self.accumulation_steps == 0):
                    self.optim.step()
                    self.optim.zero_grad()
                    if self.vicuna.training:
                        self.vicuna.zero_grad()  # Clear all gradients
                    self.connector.zero_grad()


        final_output = torch.cat(gen_tokens)

        if final_output.requires_grad:
            final_output = final_output.detach()

        # Return the generated tokens and the averaged negative log-likelihood (NLL loss)        
        return final_output, nll_loss.cpu().item()

    #This function takes the feature and question embeddings and combines them in the correct embedding format
    def encode_text_and_image(self, question, image_features):

        # Tokenize the batch of questions with padding
        inputs = self.tokenizer(
            [" Question: " + q + " Answer: " for q in question],  # Batch of strings
            padding='longest',    # Pad to the length of the longest sequence in the batch
            truncation=True,      
            return_tensors='pt' 
        )

        header = self.tokenizer(
            ["Image: "],  # Batch of strings
            padding='longest',    # Pad to the length of the longest sequence in the batch
            truncation=True,      
            return_tensors='pt'   
        )
        
        # Remove the first token from each sequence in the batch (index 1 onwards)
        input_ids = inputs.input_ids[:, 1:].to(self.device)
        header_ids = header.input_ids[:, 1:].to(self.device)
      
        embedded_text = self.vicuna.get_input_embeddings()(input_ids)

        # Place the header across the number of batches
        embedded_header = self.vicuna.get_input_embeddings()(header_ids)
        batch_size = embedded_text.size(0) 
        embedded_header = embedded_header.repeat(batch_size, 1, 1).to(self.device) 

        # Concatenate the header and the text embeddings
        embeddings = torch.cat((embedded_header, image_features, embedded_text), dim=1)

        return embeddings

    def forward(self, image_features, question, answer, max_length, itr):
        # Autoregressive prediction
        # Ensure no unnecessary intermediate results are kept
        gen, loss = self.generate_using_forward_method(max_length, 0.9, answer,question,image_features, itr)

        torch.cuda.empty_cache()  # Clear the CUDA cache

        return gen, loss