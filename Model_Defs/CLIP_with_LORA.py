import torch
import torch.nn as nn
from transformers import CLIPModel
from peft import LoraConfig, get_peft_model
import os

class CLIPWithLoRA(nn.Module):
    def __init__(self, 
                 clip_model_name="openai/clip-vit-base-patch32"):
        super(CLIPWithLoRA, self).__init__()
        
        # Load the full CLIP model (both vision and text encoders)
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        # Separate access to vision and text encoders
        self.visual_encoder = self.clip_model.vision_model
        self.text_encoder = self.clip_model.text_model

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
        vision_output, text_output = None, None
        
        # Process the visual encoder
        if pixel_values is not None:
            vision_outputs = self.visual_encoder(pixel_values, output_hidden_states=True)
            vision_output = vision_outputs.last_hidden_state  # Get final hidden state for vision
        
        # Process the text encoder
        if input_ids is not None:
            text_outputs = self.text_encoder(input_ids, output_hidden_states=True)
            text_output = text_outputs.last_hidden_state  # Get final hidden state for text
        
        return vision_output, text_output

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

        return self.visual_encoder

    def get_text_encoder(self):

        return self.text_encoder