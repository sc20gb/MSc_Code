import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model
import os

class visual_encoder(nn.Module):
    def __init__(self,encoder):
        super(visual_encoder, self).__init__()
        self.visual_encoder = encoder

    def forward(self,x,return_hidden_states=True):
        vision_outputs = self.visual_encoder(x, output_hidden_states=return_hidden_states)

        return vision_outputs.pooler_output, vision_outputs.hidden_states

class CLIPWithLoRA(nn.Module):
    def __init__(self, 
                 clip_model_name="openai/clip-vit-base-patch32", device=torch.device("cpu")):
        super(CLIPWithLoRA, self).__init__()
        
        # Load the full CLIP model (both vision and text encoders)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)

        self.clip_model.to(device)
        
        # Separate access to vision and text encoders
        self.visual_encoder = self.clip_model.vision_model
        self.text_encoder = self.clip_model.text_model

        # Initialize CLIP processor
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.device = device

    def pre_process(self,images,texts):
        image_inputs = self.processor(images=images, return_tensors="pt",do_rescale=False)
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return image_inputs, text_inputs
    
    def pre_process_images(self,images):
         image_inputs = self.processor(images=images, return_tensors="pt")

         return image_inputs['pixel_values'].squeeze()
    
    def pre_process_texts(self, texts):
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        return text_inputs['input_ids']  # Returns only input IDs

    def apply_LORA(self,lora_r=8,lora_alpha=32,lora_dropout=0.1):
        # LoRA config for both encoders
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],  # Target attention layers
            lora_dropout=lora_dropout
        )
        # Apply LoRA to both vision and text encoders
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)
        self.text_encoder = get_peft_model(self.text_encoder, lora_config)

    def forward(self, pixel_values=None, input_ids=None):
         # Preprocess images and texts to get embeddings
        image_embeddings = self.clip_model.get_image_features(pixel_values)
        text_embeddings = self.clip_model.get_text_features(input_ids)

        
        return image_embeddings, text_embeddings

    def save_model(self, save_dir):

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, "fine_tuned_clip_with_lora.pth")
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load_model(self, load_dir):

        load_path = os.path.join(load_dir, "fine_tuned_clip_with_lora.pth")
        self.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")
        
    def get_visual_encoder(self):
        ve = visual_encoder(self.visual_encoder)
        return ve

    def get_text_encoder(self):

        return self.text_encoder