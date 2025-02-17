import torch
import torch.nn as nn
from transformers import CLIPProcessor
from transformers import CLIPModel

class CLIP_Processor_Workaround(nn.Module):
    """
    A wrapper for the CLIPProcessor that simplifies image pre-processing.
    This class loads a CLIPProcessor from the pretrained model and provides a method to convert images
    into the pixel values required by a CLIP model.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", device=torch.device("cpu")):
        # Initialize the parent module.
        super(CLIP_Processor_Workaround, self).__init__()
        # Load the CLIP processor using the specified model and device.
        self.processor = CLIPProcessor.from_pretrained(clip_model_name, device=device)
        self.device = device

    def pre_process_images(self, images):
        """
        Pre-process a batch of images for the CLIP model.
        
        Args:
            images: A single image or a list of images (PIL images or other supported formats).
        
        Returns:
            A tensor containing the pre-processed pixel values.
            
        Raises:
            ValueError: If any image’s width or height is less than or equal to 1.
        """
        if not isinstance(images, list):
            images = [images]
            
        for idx, img in enumerate(images):
            try:
                width, height = img.width, img.height
            except AttributeError:
                raise ValueError(f"Image at index {idx} does not have width/height attributes.")
            if width <= 1 or height <= 1:
                raise ValueError(f"Image at index {idx} has invalid dimensions (width: {width}, height: {height}).")

        # Use the processor to convert images into a format required by the CLIP model.
        image_inputs = self.processor(images=images, return_tensors="pt")
        # Remove any unnecessary dimensions and return the pixel values.
        return image_inputs['pixel_values'].squeeze()


class CLIP_Encoder_Workaround(nn.Module):
    """
    A wrapper for the CLIPModel to expose the visual encoder.
    This class loads a pretrained CLIP model and extracts its visual encoder.
    It returns both the pooled output and the hidden states from the encoder.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", device=torch.device("cpu")):
        # Initialize the parent module.
        super(CLIP_Encoder_Workaround, self).__init__()
        # Load the pretrained CLIP model.
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        # Extract the visual encoder (the vision model) from the CLIP model.
        self.visual_encoder = self.clip.vision_model

    def forward(self, x, return_hidden_states=True):
        """
        Perform a forward pass through the visual encoder.
        
        Args:
            x: Input tensor of images.
            return_hidden_states (bool): Flag indicating whether to return all hidden states.
        
        Returns:
            A tuple containing:
                - pooled_output: The pooled visual representation of the input images.
                - hidden_states: A tuple with hidden states from the vision model (if requested).
        """
        # Forward the input through the visual encoder, requesting hidden states if specified.
        vision_outputs = self.visual_encoder(x, output_hidden_states=return_hidden_states)
        # Return both the pooled output and the complete set of hidden states.
        return vision_outputs.pooler_output, vision_outputs.hidden_states