from Training.feature_aliginment_training import feature_aliginment_training, cross_val_train, multi_stage_feature_aliginment_training
from Data_Loading.data_loading import load_laion_coco_images, load_slake_data
from Data_Loading.pre_embedding_loading import PreEmbeddingDataset, PreEmbeddingCreator
from utils.device_handler import handle_devices

from Model_Defs.CLIP_workaround import CLIP_Processor_Workaround, CLIP_Encoder_Workaround

import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


# TODO: If the hidden_layer from end changes then create a new dataset for the embeddings
def load_embeddings_dataloader(embedding_dir, dataloader, encoder,hidden_layer_from_end, device_vit,batch_size,seed):
    # load the dataloader for the general datasets embeddings for training efficency

    if not os.path.exists(embedding_dir):
        print("Creating embeddings for the dataset", embedding_dir)
        # Create the embedding dataset
        embedding_creator = PreEmbeddingCreator(encoder, dataloader, embedding_dir,hidden_layer_from_end, device=device_vit)
        embedding_creator.create_embeddings()
    else:
        print("Embeddings already exist for the dataset", embedding_dir)
    general_dataloader = DataLoader(PreEmbeddingDataset(embedding_dir), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),num_workers=1, prefetch_factor=1, pin_memory=False)    

    return general_dataloader

if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ########### Device Handling: ###########
    device_vit, device_llm = handle_devices(cpu_only=False)

    # ########### Model Parameters: ###########
    path_TinyLLama_LOCAL = os.path.join(os.getcwd(), "Models", "TinyLLama-v1.0")
    path_TinyLLama_ARC = os.path.join("/nobackup", "sc20gwb", "Models", "TinyLLama-v1.0")

    params = {
        "vicuna_path": path_TinyLLama_LOCAL,
        "connector_layers": 2,
        "embed_dim": 768,
        "version": "3000_experiment",
        "lr": 0.001,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "per_warm": 0.333,
        "batch_size": 4,
        "vir_batch_size": 32,
        "rand_seed": 42,
        "use_half": False,
        "save": True,
        "cpu_only": False,
        "general_dataset": None,
        "training_stages": [1, 2, 3],
        "hidden_layer_from_end": 1,
        'lora_rank':4, 
        'lora_dropout':0.3, 
        'lora_alpha':32,
        "stage_params": {
            1: {"lr": 0.001, "eps": 1e-8, "weight_decay": 0.01, "per_warm": 0.333, "MAX_EPOC": 5},
            2: {"lr": 0.0005, "eps": 1e-9, "weight_decay": 0.005, "per_warm": 0.25, "MAX_EPOC": 5},
            3: {"lr": 0.0002, "eps": 1e-9, "weight_decay": 0.001, "per_warm": 0.2, "MAX_EPOC": 5}
        }
    }
    
    # ########### Loading Data: ###########

    # Specify the path to the data embedding directory and the original data directory
    data_embedding_dir = os.path.join("D:\\datasets")

    # create the efficent dataloaders for the general dataset
    general_data_embedding_dir = os.path.join(data_embedding_dir, "laion_coco_embeddings")
    general_data_orginal_dir = os.path.join(data_embedding_dir, "laion_coco_images")

    # create the efficent dataloaders for the domain specific dataset
    specific_data_embedding_dir = os.path.join(data_embedding_dir, "slake_embeddings")
    specific_data_orginal_dir = os.path.join(os.getcwd(), "Slake1.0")


    # Load the CLIP transform and encoder workarounds
    encoder = CLIP_Encoder_Workaround("openai/clip-vit-base-patch32",device_vit)
    transform = CLIP_Processor_Workaround("openai/clip-vit-base-patch32",device_vit).pre_process_images

    # load the validation and train data loaders for each dataset
    general_train_dataloader,general_val_dataloader = load_laion_coco_images(
        general_data_orginal_dir,
        split="train",
        transform=transforms.Compose([transform]),
        batch_size=params["batch_size"]
        )
    
    specific_train_dataloader,specific_val_dataloader = load_slake_data(
        specific_data_orginal_dir,
        transform=transforms.Compose([transform]),
        batch_size=params["batch_size"]
        )
    
    # Load the general dataset embeddings
    train_general_dataloader = load_embeddings_dataloader(
        os.path.join(general_data_embedding_dir, "train"),
        general_train_dataloader,
        encoder,
        params["hidden_layer_from_end"],
        device_vit,
        params["batch_size"],
        params["rand_seed"]
        )
    val_general_dataloader = load_embeddings_dataloader(
        os.path.join(general_data_embedding_dir, "validation"),
        general_val_dataloader,
        encoder,
        params["hidden_layer_from_end"],
        device_vit,
        params["batch_size"],
        params["rand_seed"]
    )

    # Load the domain specific dataset embeddings
    train_specific_dataloader = load_embeddings_dataloader(
        os.path.join(specific_data_embedding_dir, "train"), 
        specific_train_dataloader, 
        encoder,
        params["hidden_layer_from_end"],
        device_vit,
        params["batch_size"],
        params["rand_seed"])
    val_specific_dataloader = load_embeddings_dataloader(
        os.path.join(specific_data_embedding_dir, "validation"), 
        specific_val_dataloader, 
        encoder,
        params["hidden_layer_from_end"],
        device_vit, 
        params["batch_size"], 
        params["rand_seed"]
        )
    
    # Add the dataloaders to the params dictionary
    params["general_train_dataloader"] = train_general_dataloader
    params["general_val_dataloader"] = val_general_dataloader
    params["specific_train_dataloader"] = train_specific_dataloader
    params["specific_val_dataloader"] = val_specific_dataloader

    # ########### Training: ###########

    # Run the multi-stage training.
    latest_ckpt = multi_stage_feature_aliginment_training(**params)
    print("Multi-stage training finished. Latest checkpoint:", latest_ckpt)