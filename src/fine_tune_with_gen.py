from Training.feature_aliginment_training import feature_aliginment_training, cross_val_train, multi_stage_feature_aliginment_training
from Testing.testing import runtest
from Data_Loading.data_loading import load_laion_coco_images, load_slake_data
from Data_Loading.pre_embedding_loading import PreEmbeddingDataset, PreEmbeddingCreator
from utils.device_handler import handle_devices
from utils.model_loaders import load_image_encoder
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA


import os
import wandb
import torch
from torch.utils.data import TensorDataset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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

    # Choose devices from those available
    device_vit, device_llm = handle_devices(cpu_only=False)

    # Construct path to models directory
    path_TinyLLama_LOCAL = os.path.join(os.path.dirname(os.getcwd()), "Models", "TinyLLama-v1.0")

    path_TinyLLama_ARC = os.path.join("/nobackup", "sc20gwb", "Models", "TinyLLama-v1.0")

    # Specify batchsize for loading the model
    batch_size = 1

    # Specify the seed for reproducibility
    seed= 42

    # Specify the emedding layer to use in the encoder
    hidden_layer_from_end = 1

    # The path of the LLM to use. Must be a LlamaForCausalLM model
    lamaCausalLM_path = path_TinyLLama_LOCAL
    
    # Stage Training Script start ###########

    # TODO: Use the new dataloader to make the loading more efficent
        #  TODO: create a new traning function that loads the embedings instead of the images
    # TODO: use the needded CLIP transformer as the transformer for the dataloader
    # TODO: check code runs smoothly

    # Specify the path to the data embedding directory and the original data directory
    data_embedding_dir = os.path.join("D:\\datasets")

    # create the efficent dataloaders for the general dataset
    general_data_embedding_dir = os.path.join(data_embedding_dir, "laion_coco_embeddings")
    general_data_orginal_dir = os.path.join(data_embedding_dir, "laion_coco_images")

    # create the efficent dataloaders for the domain specific dataset
    specific_data_embedding_dir = os.path.join(data_embedding_dir, "slake_embeddings")
    specific_data_orginal_dir = os.path.join(os.path.dirname(os.getcwd()), "Slake1.0")


    # Load the CLIP transform and encoder
    clip = CLIPWithLoRA()
    transform = clip.pre_process_images
    encoder  = clip.get_visual_encoder()

    # load the validation and train data loaders for each dataset
    general_train_dataloader,general_val_dataloader = load_laion_coco_images(
        general_data_orginal_dir,
        split="train",
        transform=transforms.Compose([clip.pre_process_images]),
        batch_size=batch_size)
    
    specific_train_dataloader,specific_val_dataloader = load_slake_data(
        specific_data_orginal_dir,
        transform=transforms.Compose([clip.pre_process_images]),
        batch_size=batch_size
        )

    # Load the general dataset embeddings
    train_laion_dataloader = load_embeddings_dataloader(
        os.path.join(general_data_embedding_dir, "train"),
        general_train_dataloader,
        encoder,
        hidden_layer_from_end,
        device_vit,
        batch_size,
        seed
        )
    val_laion_dataloader = load_embeddings_dataloader(
        os.path.join(general_data_embedding_dir, "validation"),
        general_val_dataloader,
        encoder,
        hidden_layer_from_end,
        device_vit,
        batch_size,
        seed
    )

    # Load the domain specific dataset embeddings
    train_laion_dataloader = load_embeddings_dataloader(
        os.path.join(specific_data_embedding_dir, "train"), 
        specific_train_dataloader, 
        encoder,
        hidden_layer_from_end,
        device_vit,
        batch_size,
        seed)
    val_laion_dataloader = load_embeddings_dataloader(
        os.path.join(specific_data_embedding_dir, "validation"), 
        specific_val_dataloader, 
        encoder,
        hidden_layer_from_end,
        device_vit, 
        batch_size, 
        seed
        )


    exit()

    # TODO: check PreEmbeddingCreator
    # TODO: check PreEmbeddingDataset


    # TODO: Add the new dataloaders to the training function

    # # Define model parameters and stage-specific overrides.
    # params = {
    #     "vicuna_path": lamaCausalLM_path,
    #     "connector_layers": 2,
    #     "embed_dim": 768,
    #     "image_resolution": 224,
    #     "VERSION": "3000_experiment",  # Base version string.
    #     "lr": 0.001,
    #     "eps": 1e-8,
    #     "weight_decay": 0.01,
    #     "per_warm": 0.333,
    #     "batch_size": 4,
    #     "vir_batch_size": 32,
    #     "rand_seed": 42,
    #     "MAX_EPOC": 5,  # Global value (will be overridden per stage if set in stage_params).
    #     "pre_trained_connector_path": None,
    #     "lora_rank": 4,
    #     "lora_dropout": 0.3,
    #     "lora_alpha": 32,
    #     "hidden_layer_from_end": 1,
    #     "training_step": 2,  # Global value (will be set appropriately by multi-stage).
    #     "visual_encoder_type": "CLIP-pretrained",
    #     "use_half": False,
    #     "save": True,
    #     "cpu_only": False,
    #     # Pass only one general dataset (as a dataloader) to the training function.
    #     "general_dataset": laion_dataloader,
    #     "training_stages": [1, 2, 3],
    #     # Stage-specific parameter overrides.
    #     "stage_params": {
    #          1: {"lr": 0.001, "eps": 1e-8, "weight_decay": 0.01, "per_warm": 0.333, "MAX_EPOC": 5},
    #          2: {"lr": 0.0005, "eps": 1e-9, "weight_decay": 0.005, "per_warm": 0.25, "MAX_EPOC": 5},
    #          3: {"lr": 0.0002, "eps": 1e-9, "weight_decay": 0.001, "per_warm": 0.2, "MAX_EPOC": 5}
    #     }
    # }

    # # Run the multi-stage training.
    # latest_ckpt = multi_stage_feature_aliginment_training(**params)
    # print("Multi-stage training finished. Latest checkpoint:", latest_ckpt)