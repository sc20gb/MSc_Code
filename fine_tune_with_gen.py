import torchvision.transforms as transforms
import os
from Data_Loading.data_loading import load_data, load_data_cross_val
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from Model_Defs.CLIP import CLIP
from transformers import get_cosine_schedule_with_warmup
from Model_Defs.connector_LLM_with_gen import Connector_LLM_With_Gen
import wandb
import math
from sklearn.model_selection import KFold
from collections import defaultdict
from torch.utils.data import random_split
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from torch.optim.lr_scheduler import LambdaLR
from utils.half import handle_half_for_layer_Norm
from utils.metrics import Metrics, calc_loss_and_metrics,  MetricsList


#I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Import the CustomCosineSchedulerWithWarmup class
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class CustomSchedulerWithWarmup(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, training_step, schedule_type="cosine"):
        self.training_step = training_step
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.schedule_type = schedule_type

        if self.training_step == 1:
            # Choose between linear or cosine schedule for phase 1
            if self.schedule_type == "linear":
                self.phase1_scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps, num_training_steps
                )
            elif self.schedule_type == "cosine":
                self.phase1_scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps, num_training_steps
                )
            else:
                raise ValueError("schedule_type must be 'linear' or 'cosine'")
        else:
            # Default to cosine schedule when training both projector and LLM (phase 2)
            self.phase2_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )

        def lr_lambda(current_step):
            if self.training_step == 1:
                return self.phase1_scheduler.lr_lambdas[0](current_step)
            else:
                return self.phase2_scheduler.lr_lambdas[0](current_step)

        super().__init__(optimizer, lr_lambda)

#Return  the object that is the visual encoder, return the heat scaling parameter as well
def load_ViT_img_encoder(tokenizer,transformer_width,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=256,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)

    state_dict = torch.load(clip_model_path)
    clip.load_state_dict(state_dict,strict=True)
  
    visual = clip.visual

    return visual.to(device)#clip.visual.to(device)

#A container for layernorms with .half
class CustomLayerNorm(torch.nn.LayerNorm):
    def forward(self, input):
        # Convert input to float32 for LayerNorm calculation, then cast back to the original dtype
        return super().forward(input.float()).to(input.dtype)

# Handle the assigining of model to the GPUs
def handle_devices(cpu_only=False):
    if torch.cuda.is_available() and not cpu_only:
        gpu_count = torch.cuda.device_count()
        if gpu_count >= 2:
            print(f"CUDA is available with {gpu_count} GPU(s)!")
            
            # Assign the first GPU for the visual encoder
            device_vit = torch.device("cuda:0")
            print(f"Visual encoder will run on GPU 0: {torch.cuda.get_device_name(0)}")

            # Assign the second GPU for the connector LLM
            device_llm = torch.device("cuda:1")
            print(f"Connector LLM will run on GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            print("Only one GPU available, models are split between GPU 0")
            device_vit = torch.device("cuda:0")
            device_llm = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device_vit = torch.device("cpu")
        device_llm = torch.device("cpu")

    return device_vit, device_llm

def load_data_loaders(val_dataset, train_dataset, visual_encoder_type, image_resolution, batch_size, rand_seed,processor=None):
    """
    Loads data loaders for training and validation, based on the provided dataset and visual encoder type.
    
    Args:
        val_dataset: The validation dataset or None if not provided.
        train_dataset: The training dataset or None if not provided.
        visual_encoder_type (str): The type of visual encoder (e.g., "CLIP-trained").
        image_resolution (int): The desired image resolution for resizing.
        batch_size (int): The batch size for data loading.
        rand_seed (int): The random seed for data shuffling.
        processor:The CLIPProcessor required when using the CLIPModel class
    
    Returns:
        train_loader, validate_loader: The training and validation data loaders.
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
        data_path = os.path.join(os.getcwd(), 'Slake1.0')
        train_loader, validate_loader = load_data(
            data_transform, batch_size, rand_seed, data_path
        )
    else:
        train_loader = train_dataset
        validate_loader = val_dataset

    return train_loader, validate_loader

# Handels the loading of the img encoder and the appropriate dataloaders, provides compatability with cross_fold_validation
def load_image_encoder(visual_encoder_type,device,val_dataset,train_dataset, image_resolution,batch_size,rand_seed, **model_args):

     # Load the appropriate pipline and visual encoder for the visual encoder method
    if visual_encoder_type == "CLIP-trained":
        # LOAD ViT encoder from the CLIP model on the first GPU
        print("loading our visual encoder")
        img_encoder = load_ViT_img_encoder(
            device=device,
            **model_args
            )
        
        #  load data
        train_loader, validate_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed)     
    elif visual_encoder_type == "CLIP-pretrained":
        print("loading pretrained CLIP visual encoder")
        clip = CLIPWithLoRA()
        img_encoder = clip.get_visual_encoder()
        img_encoder = img_encoder.to(device)

        #  load data
        train_loader, validate_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed,processor=clip.pre_process_images)
    else:
        print("loading fine-tuned CLIP model")
        clip = CLIPWithLoRA()
        clip.apply_LORA(lora_r=8, lora_alpha=32, lora_dropout=0.1)
        clip.load_model(clip_model_path)
        img_encoder = clip.get_visual_encoder()
        img_encoder = img_encoder.to(device)
         #  load data
        train_loader, validate_loader = load_data_loaders(val_dataset,train_dataset,visual_encoder_type,image_resolution,batch_size,rand_seed,processor=clip.pre_process_images)

    return img_encoder, train_loader, validate_loader

def feature_aliginment_training(
        vicuna_path,
        connector_layers,
        embed_dim,
        image_resolution,
        VERSION,
        lr,
        eps,
        weight_decay,
        per_warm,
        batch_size,
        vir_batch_size,
        rand_seed,
        MAX_EPOC,
        pre_trained_connector_path,
        lora_rank,
        lora_dropout,
        lora_alpha,
        save=False,
        cpu_only=False,
        hidden_layer_from_end=0,
        training_step=1, 
        val_dataset = None,
        train_dataset = None,
        visual_encoder_type="CLIP-trained",
        use_half=True,
        **model_args
        ):
    # CHECK GPU SUPPORT AND ASSIGN DEVICES
    device_image_encoder, device_llm = handle_devices(cpu_only)
    
    # deal with vir_batchsize
    accumulation_steps = vir_batch_size // batch_size

    # Load connector and vicuna model
    connector_llm = Connector_LLM_With_Gen(image_emded_dim=embed_dim,llm_path=vicuna_path,connector_layers=connector_layers, device=device_llm)

    # Load image_encoder and correct data_loaders
    img_encoder, train_loader, validate_loader = load_image_encoder(visual_encoder_type,device_image_encoder,val_dataset,train_dataset, image_resolution,batch_size,rand_seed,**model_args)

    # FREEZE CLIP TRAINING (should save memory and computation as well)
    img_encoder.eval()

    # half the size of weights to save memory
    if use_half:
        connector_llm.half()
        # Half does not work with some layers
        handle_half_for_layer_Norm(connector_llm)
        img_encoder.half()
        # Half does not work with some layers
        handle_half_for_layer_Norm(img_encoder)


    # Ensure correct device
    img_encoder.to(device_image_encoder)
    connector_llm.to(device_llm)

    # handle loading for step 2
    if training_step == 2:
        if pre_trained_connector_path != None:
            print("Loading the connector MLP")
            #Load the pre_trained connector stat_dict
            connector_llm.load_connector(pre_trained_connector_path)
        else:
            print("No connector given, training from scratch")

        #lora is only needed for step 2
        connector_llm.apply_lora(rank=lora_rank,dropout=lora_dropout,alpha=lora_alpha)
    else:
        #freeze llm training for step 1
        connector_llm.llm.eval()

    # Get the warm up steps for the scheduler
    total_training_steps = math.ceil(len(train_loader.dataset) / vir_batch_size) * MAX_EPOC
    num_warmup_steps = math.ceil(total_training_steps *  per_warm)

    print(num_warmup_steps , " warmup steps")
    print(total_training_steps, " total steps")
    print((len(train_loader.dataset)/batch_size), " train batches")

    # Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector_llm.parameters(), lr=lr,weight_decay=weight_decay, eps=eps)
    scheduler = CustomSchedulerWithWarmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps,training_step=training_step)

    # to store metrics
    metrics_training = Metrics()
    metrics_validate = Metrics()
    training_list = MetricsList()
    validate_list = MetricsList()
    for n in range(1, MAX_EPOC + 1):
        
        # Training the LLM is not needed in step 1
        connector_llm.train() if training_step == 2 else connector_llm.connector.train()

        count_t = 0
        optim.zero_grad()

        for image_tensor, mask_tensor, question, answer in train_loader:
            # Get image features from the img encoder
            with torch.no_grad():
                _, hidden_states = img_encoder(image_tensor.half().to(device_image_encoder),return_hidden_states=True)                    

            # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
            image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

            # Tokenize answer
            answer_ = connector_llm.tokenizer(answer, padding='longest', truncation=True, return_tensors='pt', add_special_tokens=True).input_ids.to(device_llm)[:, 1:]

            # Get MLLM prediction and NLL loss
            output, loss  = connector_llm(image_features.to(device_llm), question, answer_)

            # Eval
            metrics = Metrics(loss,**calc_loss_and_metrics(list(output),list(answer_),tokenizer=connector_llm.tokenizer))
            metrics_training += metrics     
            count_t = count_t + 1

            # Backward pass and optimizer step
            loss.backward()
            if ((count_t) % accumulation_steps == 0):
                optim.step()
                scheduler.step()
                optim.zero_grad()
                if connector_llm.llm.training:
                    connector_llm.llm.zero_grad()
                connector_llm.connector.zero_grad()


        # Ensure to perform a step if we have leftover gradients
        if (count_t + 1) % accumulation_steps != 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()
            connector_llm.zero_grad()

        # VALIDATE
        count = 0
        connector_llm.eval()
        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer in validate_loader:
                # Get image features from the img encoder
                _, hidden_states = img_encoder(image_tensor.half().to(device_image_encoder),return_hidden_states=True)                    

                # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                # Tokenize answer
                answer_ = connector_llm.tokenizer(answer, padding='longest', truncation=True, return_tensors='pt', add_special_tokens=True).input_ids.to(device_llm)[:, 1:]

                # Get MLLM prediction and NLL loss
                output, loss  = connector_llm(image_features.to(device_llm), question, answer_)

                # Eval
                metrics = Metrics(loss,**calc_loss_and_metrics(list(output),list(answer_),tokenizer=connector_llm.tokenizer))
                metrics_validate += metrics     
                count = count + 1


        # SAVE RESULTS
        if save:
            if training_step == 2:
                if not os.path.exists(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION))):
                    os.makedirs(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION)))
                torch.save(connector_llm.state_dict(), os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION), "MLLM_model" + str(n) + ".pth"))
            else:
                if not os.path.exists(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION))):
                    os.makedirs(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION)))
                torch.save(connector_llm.connector.state_dict(), os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION), "connector_LLM_model" + str(n) + ".pth"))

        # we need to record these as well
        if  val_dataset == None or train_dataset == None:
            if count != 0 and count_t != 0:
                    wandb.log((metrics_training/count_t).get_log("training_") | (metrics_validate/count).get_log("validate_"))
            else:
                wandb.log(wandb.log(Metrics(-1,-1,-1,-1,-1,-1,-1).get_log("training_") | Metrics(-1,-1,-1,-1,-1,-1).get_log("validate_")))
        else:
            training_list.append(metrics_training/count_t)
            validate_list.append(metrics_validate/count)
            

    return training_list, validate_list

import os
import torch
from torchvision import transforms

def cross_val_train(para, n_splits=3, per_data=1.0):

    # Load the train and val datasets concatnated
    dataset = load_data_cross_val( transforms.Compose([
            transforms.Resize((para["image_resolution"],para["image_resolution"] )),
            transforms.ToTensor()
        ]), os.path.join(os.getcwd(), 'Slake1.0'))
    
    if per_data != 1.0:
    
        # Define the split sizes
        train_size = int(per_data * len(dataset))
        discard_size = len(dataset) - train_size

        # Split the dataset into two parts (e.g., 80% and 20%)
        dataset, _ = random_split(dataset, [train_size, discard_size],torch.Generator().manual_seed(para["rand_seed"]))


    
    #Make sure to shuffle the data with the seed
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=para["rand_seed"])

    summed_list_training = MetricsList()
    summed_list_validate = MetricsList()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{n_splits}')
        
        # Create data loaders for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=para["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(para["rand_seed"]),num_workers=1,  prefetch_factor=1,pin_memory=False)
        val_loader = DataLoader(val_subset, batch_size=para["batch_size"],num_workers=1, prefetch_factor=1, pin_memory=False)
        training_list, validate_list = feature_aliginment_training(**para, train_dataset=train_loader, val_dataset=val_loader)

        if len(summed_list_training) == 0:
            summed_list_training = training_list
            summed_list_validate  = validate_list
        else:
            summed_list_training += training_list
            summed_list_validate  += validate_list

    avg_list_training = summed_list_training /  n_splits
    avg_list_validate = summed_list_validate / n_splits
        
    for (train, val) in zip(avg_list_training, avg_list_validate):
        wandb.log(wandb.log((train).get_log("training_") | (val).get_log("validate_")))

if __name__ == '__main__':

    path_TinyLLama_LOCAL = os.path.join(os.getcwd(),"Models","TinyLLama-v1.0")

    path_TinyLLama_ARC = os.path.join("/nobackup", "sc20gwb", "Models", "TinyLLama-v1.0")

    # The path of the LLM to use. Must be a LlamaForCausalLM model
    lamaCausalLM_path = path_TinyLLama_LOCAL

    LR_LIST = [1e-6,1e-3,1e-4]

    WEIGHT_DECAY_LIST = [1e-4]

    PERC_WARM_LIST = [0.0]

    VIR_BATCH_SIZE_LIST = [64,256]

    NORM_LIST = [False]

    DROPOUT_LIST = [0.3]

    RANK_LIST = [8]

    HIDDEN_LAYER_LIST = [1]

    CONNECTOR_LAYERS_LIST = [2]

    LORA_ALPHA_LIST =  [32]

    optim_list = [{
            "vicuna_path":lamaCausalLM_path,
            "connector_layers":cl,
            "embed_dim":768,
            "image_resolution":224,
            "lr": lr,
            "eps":1e-6,
            "weight_decay":wd,
            "per_warm": pw,
            "batch_size":1,
            "vir_batch_size":vb,
            "rand_seed":42,
            "MAX_EPOC":30,
            "VERSION":3000,
            "save":False,
            "cpu_only":False,
            "hidden_layer_from_end": hl,
            "training_step":1,
            "lora_dropout":do,
            "lora_rank":r,
            "norm":  norm,
            "pre_trained_connector_path":None,
            "lora_alpha": a,
            "visual_encoder_type": "CLIP-pretrained"
                }
                for lr in LR_LIST 
                for wd in WEIGHT_DECAY_LIST 
                for cl in CONNECTOR_LAYERS_LIST
                for pw in PERC_WARM_LIST
                for vb in VIR_BATCH_SIZE_LIST
                for hl in HIDDEN_LAYER_LIST
                for do in DROPOUT_LIST
                for r in  RANK_LIST
                for norm in NORM_LIST
                for a in LORA_ALPHA_LIST
                ]

    for i, para in enumerate(optim_list):
        para['VERSION'] += i
        wandb.init(project="path_TinyLLama",config=para)
        #print("Cross Validation for VERSION ", para["VERSION"])
        feature_aliginment_training(**para)
        #cross_val_train(para,n_splits=3,per_data=1.0)
        wandb.finish()