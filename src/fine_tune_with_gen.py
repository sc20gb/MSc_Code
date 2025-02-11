from Training.feature_aliginment_training import feature_aliginment_training, cross_val_train, multi_stage_feature_aliginment_training
from Testing.testing import runtest

import os
import wandb
import torch
from torch.utils.data import TensorDataset


if __name__ == '__main__':
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Modify as needed to run the training script, cross_val_train, or test script:
    
    #Test Script start ###########
    # Construct path to models directory
    path_TinyLLama_LOCAL = os.path.join(os.path.dirname(os.getcwd()), "Models", "TinyLLama-v1.0")

    path_TinyLLama_ARC = os.path.join("/nobackup", "sc20gwb", "Models", "TinyLLama-v1.0")

    # The path of the LLM to use. Must be a LlamaForCausalLM model
    lamaCausalLM_path = path_TinyLLama_LOCAL


    #for i in range(1, 21):
    # Test the model, only run testing code
    modelpath = os.path.join(os.path.dirname(os.getcwd()), "SavedModels", "MLLM_V_3000", "MLLM_model"  + str(20) + ".pth")
    test_params = {
        "vicuna_path": lamaCausalLM_path,
        "connector_layers": 2,
        "embed_dim": 768,
        "image_resolution": 224,
        "VERSION": 1,
        "lr": 0.001,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "per_warm": 0.333,
        "batch_size": 4,
        "pre_trained_connector_path": None,
        "vir_batch_size": 32,
        "rand_seed": 42,
        "MAX_EPOC": 1,
        "mllm_checkpoint_path": modelpath,
        "lora_rank": 4,
        "lora_dropout": 0.3,
        "lora_alpha": 32,
        "save": False,
        "cpu_only": False,
        "hidden_layer_from_end": 1,
        "training_step": 2,
        "visual_encoder_type": "CLIP-pretrained",
        "use_half": False
    }
    runtest(lamaCausalLM_path,modelpath, test_params)
    # end script, remove testing code to run training script
    exit()

    #Training Script start ###########

    #Connector parameters
    HIDDEN_LAYER_LIST = [1]
    CONNECTOR_LAYERS_LIST = [2]

    #Training parameters
    LR_LIST = [0.001]
    WEIGHT_DECAY_LIST = [0.01]
    PERC_WARM_LIST = [0.3333]
    VIR_BATCH_SIZE_LIST = [64]

    #LoRA parameters
    DROPOUT_LIST = [0.3,0.0,0.5]
    RANK_LIST = [8,4,12]
    LORA_ALPHA_LIST =  [32,16,8]

    optim_list = [{
            "vicuna_path":lamaCausalLM_path,
            "connector_layers":cl,
            "embed_dim":768,
            "image_resolution":224,
            "lr": lr,
            "eps":1e-8,
            "weight_decay":wd,
            "per_warm": pw,
            "batch_size":4,
            "vir_batch_size":vb,
            "rand_seed":42,
            "MAX_EPOC":20,
            "VERSION":3000,
            "save":True,
            "cpu_only":False,
            "hidden_layer_from_end": hl,
            "training_step":2,
            "lora_dropout":do,
            "lora_rank":r,
            "pre_trained_connector_path":None,
            "lora_alpha": a,
            "visual_encoder_type": "CLIP-pretrained",
            "use_half" : False
                }
                for lr in LR_LIST 
                for wd in WEIGHT_DECAY_LIST 
                for cl in CONNECTOR_LAYERS_LIST
                for pw in PERC_WARM_LIST
                for vb in VIR_BATCH_SIZE_LIST
                for hl in HIDDEN_LAYER_LIST
                for do in DROPOUT_LIST
                for r in  RANK_LIST
                for a in LORA_ALPHA_LIST
                ]
    # Remove the first 3
    optim_list = optim_list[3:]

    for i, para in enumerate(optim_list):
        para['VERSION'] += i
        wandb.init(project="path_TinyLLama",config=para)
        #print("Cross Validation for VERSION ", para["VERSION"])
        feature_aliginment_training(**para)
        #cross_val_train(para,n_splits=3,per_data=1.0)
        wandb.finish()



    #3 Stage Training Script start ###########

    # Create dummy datasets for demonstration.
    # Replace these with your actual dataset objects.
    dummy_images = torch.randn(100, 3, 224, 224)
    dummy_targets = torch.randint(0, 10, (100,))
    general_dataset = TensorDataset(dummy_images, dummy_targets)
    medical_dataset = TensorDataset(dummy_images, dummy_targets)

    # Define model parameters and stage-specific overrides.
    params = {
        "vicuna_path": lamaCausalLM_path,
        "connector_layers": 2,
        "embed_dim": 768,
        "image_resolution": 224,
        "VERSION": "3000_experiment",  # Base version string.
        "lr": 0.001,
        "eps": 1e-8,
        "weight_decay": 0.01,
        "per_warm": 0.333,
        "batch_size": 4,
        "vir_batch_size": 32,
        "rand_seed": 42,
        "MAX_EPOC": 5,  # Global value (will be overridden per stage if set in stage_params).
        "pre_trained_connector_path": None,
        "lora_rank": 4,
        "lora_dropout": 0.3,
        "lora_alpha": 32,
        "hidden_layer_from_end": 1,
        "training_step": 2,  # Global value (will be set appropriately by multi-stage).
        "visual_encoder_type": "CLIP-pretrained",
        "use_half": False,
        "save": True,
        "cpu_only": False,
        # Datasets for different stages:
        "general_dataset": general_dataset,
        "medical_dataset": medical_dataset,
        "training_stages": [1, 2, 3],
        # Stage-specific parameter overrides.
        "stage_params": {
             1: {"lr": 0.001, "eps": 1e-8, "weight_decay": 0.01, "per_warm": 0.333, "MAX_EPOC": 5},
             2: {"lr": 0.0005, "eps": 1e-9, "weight_decay": 0.005, "per_warm": 0.25, "MAX_EPOC": 8},
             3: {"lr": 0.0002, "eps": 1e-9, "weight_decay": 0.001, "per_warm": 0.2, "MAX_EPOC": 10}
        }
    }

    # Run the multi-stage training.
    latest_ckpt = multi_stage_feature_aliginment_training(**params)
    print("Multi-stage training finished. Latest checkpoint:", latest_ckpt)