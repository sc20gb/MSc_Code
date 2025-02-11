from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from Model_Defs.CLIP import CLIP
import torch
import torchvision.transforms as transforms
import os
from Data_Loading.data_loading import load_data, load_test_data

def load_image_encoder(visual_encoder_type,device,val_dataset,train_dataset, image_resolution,batch_size,rand_seed, **model_args):
    """Loads appropriate visual encoder and creates data loaders.
    
    Supports multiple visual encoder types:
    - CLIP-trained: Custom trained CLIP
    - CLIP-pretrained: Original CLIP
    - Fine-tuned CLIP with LoRA
    
    Args:
        visual_encoder_type (str): Type of encoder to load
        device: Device to load model on
        val_dataset: Optional validation dataset
        train_dataset: Optional training dataset
        image_resolution (int): Input image size
        batch_size (int): Batch size
        rand_seed (int): Random seed
        **model_args: Additional arguments for model creation
        
    Returns:
        tuple: (img_encoder, train_loader, validate_loader)
    """

    clip_model_path = model_args.get("clip_model_path", None)
     # Load the appropriate pipline and visual encoder for the visual encoder method
    if visual_encoder_type == "CLIP-trained":
        # LOAD ViT encoder from the CLIP model on the first GPU
        print("loading our visual encoder")
        img_encoder = load_ViT_img_encoder(
            device=device,
            **model_args
            )
        
        #  load data
        train_loader, validate_loader, test_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed)     
    elif visual_encoder_type == "CLIP-pretrained":
        print("loading pretrained CLIP visual encoder")
        clip = CLIPWithLoRA()
        img_encoder = clip.get_visual_encoder()
        img_encoder = img_encoder.to(device)

        #  load data
        train_loader, validate_loader, test_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed,processor=clip.pre_process_images)
    else:
        print("loading fine-tuned CLIP model")
        if clip_model_path == None:
            raise ValueError("clip_model_path must be specified for fine-tuned CLIP model")
        clip = CLIPWithLoRA()
        clip.apply_LORA(lora_r=8, lora_alpha=32, lora_dropout=0.1)
        clip.load_model(clip_model_path)
        img_encoder = clip.get_visual_encoder()
        img_encoder = img_encoder.to(device)
         #  load data
        train_loader, validate_loader, test_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed,processor=clip.pre_process_images)

    return img_encoder, train_loader, validate_loader, test_loader

def load_ViT_img_encoder(tokenizer,transformer_width,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    """Loads and returns a Vision Transformer image encoder from a CLIP model checkpoint.
    
    Args:
        tokenizer: Tokenizer for the language model
        transformer_width (int): Width of transformer layers
        transformer_layers (int): Number of transformer layers
        transformer_heads (int): Number of attention heads
        embed_dim (int): Embedding dimension
        vision_width (int): Width of vision layers
        image_resolution (int): Input image resolution
        vision_patch_size (int): Size of image patches
        vision_layers (int): Number of vision layers
        device: Device to load model on
        clip_model_path (str): Path to CLIP checkpoint
        
    Returns:
        torch.nn.Module: The visual encoder portion of CLIP
    """
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=256,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)

    state_dict = torch.load(clip_model_path)
    clip.load_state_dict(state_dict,strict=True)
  
    visual = clip.visual

    return visual.to(device)#clip.visual.to(device)

def load_data_loaders(val_dataset, train_dataset, visual_encoder_type, image_resolution, batch_size, rand_seed,processor=None):
    """Creates data loaders for training and validation datasets.
    
    Handles different visual encoder types and their preprocessing requirements.
    
    Args:
        val_dataset: Optional validation dataset
        train_dataset: Optional training dataset  
        visual_encoder_type (str): Type of visual encoder
        image_resolution (int): Target image size
        batch_size (int): Batch size for loading
        rand_seed (int): Random seed for reproducibility
        processor: Optional CLIP processor for preprocessing
        
    Returns:
        tuple: (train_loader, validate_loader) - DataLoader objects
    """
    # LOAD DATA
    if val_dataset is None or train_dataset is None:
        # Load data based on visual encoder type
        if visual_encoder_type == "CLIP-trained":
            data_transform = transforms.Compose([
            transforms.Resize((image_resolution, image_resolution)),
            transforms.ToTensor()
        ])
        else:
            if processor == None:
                print("No CLIPProcessor provided when using CLIPModel")
            data_transform = transforms.Compose([
            processor
        ])
        data_path = os.path.join(os.path.dirname(os.getcwd()), 'Slake1.0')
        train_loader, validate_loader = load_data(
            data_transform, batch_size, rand_seed, data_path
        )
        test_loader = load_test_data(data_transform, batch_size, rand_seed, data_path)
    else:
        train_loader = train_dataset
        validate_loader = val_dataset
        test_loader = None

    return train_loader, validate_loader, test_loader
