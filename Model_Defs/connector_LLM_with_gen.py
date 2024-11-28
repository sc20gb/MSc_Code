import torch
import torch.nn as nn
import os
import gc

from utils.utils import LlamaForCausalLMCustom
from transformers import LlamaTokenizer

import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint
from peft import get_peft_model, LoraConfig
from contextlib import nullcontext

import psutil
def print_memory_usage():
    # Print CPU memory usage
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    cpu_memory_rss = mem_info.rss / (1024 ** 3)  # Resident Set Size (RSS) in GB
    cpu_memory_vms = mem_info.vms / (1024 ** 3)  # Virtual Memory Size (VMS) in GB
    print(f"Memory cuda: {torch.cuda.memory_allocated() / 1e6} MB")
    print(f"CPU Memory Usage - VMS: {cpu_memory_vms:.2f} GB")

class Connector_LLM_With_Gen(nn.Module):
    def __init__(self, image_emded_dim, device, connector_layers,llm_path,seed=42,max_length=100):
        super(Connector_LLM_With_Gen, self).__init__()

        self.image_emded_dim = image_emded_dim
        self.device = device 
        self.llm_path = llm_path
        self.max_length = max_length

        #Load the llm model
        self.llm ,self.tokenizer = self.load_llm()

        #Needed to return the loss value
        self.llm.generation_config.return_dict_in_generate = True

        # Get the embedding dim of the llm
        hidden_size = getattr(self.llm.config, "hidden_size", None)
        if hidden_size is not None:
            self.llm_hidden_size = self.llm.config.hidden_size
        else:
            # Incase the config uses a diffrent key fallback to a forward pass
            self.llm.eval()
            with torch.no_grad():
                self.llm_hidden_size = self.llm.get_input_embeddings()(torch.tensor([0],dtype=torch.int64,device=device)).size(1)
            self.llm.train()

        # Create the connector MLP
        layers = []
        input_dim = self.image_emded_dim
        for _ in range(connector_layers - 1):
            layers.append(nn.Linear(input_dim, self.llm_hidden_size))
            layers.append(nn.GELU())
            input_dim = self.llm_hidden_size
        layers.append(nn.Linear(self.llm_hidden_size, self.llm_hidden_size))
        self.connector = nn.Sequential(*layers).to(device)
        self._initialize_weights(seed)

    #Applies lora to the llm model
    def apply_lora(self,rank=8,alpha=32,modules=["q_proj", "v_proj"],dropout=0.1):
        # Define the LoRA configuration
        lora_config = LoraConfig(
            r=rank,   # Rank of the low-rank adaptation
            lora_alpha=alpha,
            target_modules=modules,  # Modules to apply LoRA to
            lora_dropout=dropout
        )

        # Wrap your model with LoRA
        self.llm = get_peft_model(self.llm, lora_config)

    #Initilises the weights for the MLP connector (visual embedder)
    def _initialize_weights(self, seed):
            
            if seed is not None:
                torch.manual_seed(seed)  # Set seed for reproducibility
    
            for m in self.connector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming initialization
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)  # Initialize biases to zero

    # Load the tokenizer and model
    def load_llm(self):
        tokenizer = LlamaTokenizer.from_pretrained(os.path.join(self.llm_path, "tokenizer"))
        llm = LlamaForCausalLMCustom.from_pretrained(os.path.join(self.llm_path,  "model"))

        return llm, tokenizer

    #A debuggging function to check the gradents of diffrent layers
    def check_grad(self,t,strv):
        if not t.requires_grad:
            print("NO GRAD FOR ", strv, t.size(), t.device)
        else:
            print("Grad FOR ", strv, t.size(), t.device)
    
    def encode_text_and_image(self, question, image_embeddings):
        """
        Encodes the text and image features together into embeddings for model input.
        
        Args:
            question (list of str): List of question strings for tokenization.
            image_features (Tensor): Image features to concatenate with text embeddings.
        
        Returns:
            embeddings (Tensor): Concatenated embeddings of the header, image features, and text.
            attention_mask (Tensor): The attention mask for the embeddings.

        """
        
        # # Combine header and question templates for batch tokenization
        prefix = "Question: "
        suffix = " Answer: "
        batch_text = [f"{prefix}{q}{suffix}" for q in question]

        # Tokenize batch text with padding and truncation
        inputs = self.tokenizer(batch_text,padding='longest',truncation=True,return_tensors='pt').to(self.device)  # Move tokenized inputs to the target device
       
        attention_mask = inputs["attention_mask"][:,1:]
        input_ids = inputs.input_ids[:, 1:]

        # Embed the text input IDs, excluding the first token to match the input format
        embedded_text = self.llm.get_input_embeddings()(input_ids)

        # Embed the static header ("Image:") once and expand across the batch
        header_tokens = self.tokenizer("Image: ", return_tensors='pt')

        header_ids = header_tokens.input_ids.to(self.device)

        header_mask = header_tokens["attention_mask"]

        embedded_header = self.llm.get_input_embeddings()(header_ids)
        embedded_header = embedded_header.expand(embedded_text.size(0), -1, -1)  # Expand header to match batch size

        header_mask = header_mask.expand(embedded_text.size(0),-1).to(self.device)

        # Concatenate the header padding, embedded header, image features, and embedded text
        embeddings = torch.cat((embedded_header, image_embeddings, embedded_text), dim=1)

        attention_mask  = torch.cat((header_mask,torch.ones(image_embeddings.shape[:2], dtype=torch.long, device=embeddings.device),attention_mask), dim=1)

        return embeddings, attention_mask

    def forward(self, image_embeddings, question, answer):

        # Project the image embeddings
        #projected_img_embeddings = self.connector(image_embeddings)
        
        ids = self.tokenizer(["Hello?"],padding='longest',truncation=True,return_tensors='pt')
        embeddings = self.llm.get_input_embeddings()(ids.input_ids.to(self.device))
        attention_mask = ids["attention_mask"].to(self.device)
        answer = self.tokenizer(["Hello? How are you today etc tect ..         ...  ... ... ... .. .. . . .. . . .."],padding='longest',truncation=True,return_tensors='pt').input_ids.to(self.device)


        # embed the text into the VAQ format, and concatnate them for llm generation
        #embeddings, attention_mask = self.encode_text_and_image(question, projected_img_embeddings)
        
       # Autoregressive prediction
        with torch.no_grad() if not self.llm.training else nullcontext():
            outputs = self.llm.generate(
                inputs_embeds=embeddings,
                labels=answer,
                attention_mask=attention_mask,
                max_length=self.max_length,
                generation_config=self.llm.generation_config,
            )


        
        print(self.tokenizer.decode(outputs.sequences[0]))




        return outputs.sequences, outputs.loss
    

    #loads the connector from a file 
    def load_connector(self,pre_trained_connector_path):
        state_dict = torch.load(pre_trained_connector_path)
        self.connector.load_state_dict(state_dict)