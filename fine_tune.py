import torchvision.transforms as transforms
import os
from Data_Loading.data_loading import load_data, load_data_cross_val
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from Model_Defs.CLIP import VisionTransformer, CLIP
from transformers import get_cosine_schedule_with_warmup
from Model_Defs.connector_LLM import Connector_LLM, print_memory_usage
import sys
import re
import torch.nn.functional as F
from torcheval.metrics.functional import bleu_score
import wandb
import math
import gc
import string
from sklearn.model_selection import KFold
from collections import defaultdict
from collections import Counter
from torch.utils.data import random_split
from Model_Defs.CLIP_with_LORA import CLIPWithLoRA
from torch.optim.lr_scheduler import LambdaLR

# Import the CustomCosineSchedulerWithWarmup class
class CustomCosineSchedulerWithWarmup(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, training_step):
        self.training_step = training_step
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.cosine_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

        def lr_lambda(current_step):
            if self.training_step == 1:
                return 1.0  # Constant learning rate when layers are frozen
            else:
                return self.cosine_scheduler.lr_lambdas[0](current_step)

        super().__init__(optimizer, lr_lambda)


#Return  the object that is the visual encoder, return the heat scaling parameter as well
def load_ViT_img_encoder(tokenizer,transformer_width,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=256,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)

    state_dict = torch.load(clip_model_path)
    clip.load_state_dict(state_dict,strict=True)
  
    visual = clip.visual

    return visual.to(device)#clip.visual.to(device)

def process_string(s):
        # Convert to lowercase
        s = s.lower()
        # Remove all whitespace characters except spaces
        s = re.sub(r'[^\S ]+', '', s)
        # Replace multiple spaces with a single space
        s = re.sub(r' +', ' ', s)

         # Remove punctuation
        s = ''.join([char for char in s if char not in string.punctuation])

        return s.strip()  # Optionally, remove leading/trailing spaces

def calc_loss_and_metrics(predicted, target, tokenizer):

    accuracy = 0
    bleu_scores = []
    precisions = []
    recalls = []
    f1_scores = []

    print(predicted)

    # Iterate over each sample in the batch
    for i in range(len(target)):  # Assuming target has shape (batch_size, ...)
        # Extract the individual target and predicted for the current batch item
        target_item = target[i]
        predicted_item = predicted[i]
        
        # Ensure the answer has its capitals and whitespace removed
        predicted_string = process_string(tokenizer.decode(predicted_item.long(), skip_special_tokens=True))
        target_string = process_string(tokenizer.decode(target_item.long(), skip_special_tokens=True))
        
        print("Predicted:")
        print(predicted_string)
        print("Answer:")
        print(target_string)

        predicted_list = predicted_string.split()
        target_list = target_string.split()

        print(predicted_list)
        print(target_list)

        if predicted_list == target_list:
            accuracy += 1.0

        if len(target_list) == 0 or len(predicted_list) == 0:
            bleu_score_ = 0.0
        else:
            bleu_score_ = bleu_score(predicted_string, [target_string], n_gram=1).item()

        # Calculate precision and recall
        prec = 0.0
        rec = 0.0
        if len(predicted_string) != 0 and len(target_string) != 0:
            predicted_count = Counter(predicted_list)
            target_count = Counter(target_list)

          # Calculate the number of common tokens (one-to-one matching)
            common_count = 0
            for token in predicted_count:
                if token in target_count:
                    # Take the minimum of the occurrences in both lists for exact matches
                    common_count += min(predicted_count[token], target_count[token])

            print(common_count)
            # Precision: proportion of predicted tokens that are correct
            prec = common_count / len(predicted_list)
            
            # Recall: proportion of target tokens that were predicted
            rec = common_count / len(target_list)



        if prec + rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        bleu_scores.append(bleu_score_)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    # Average the accuracy, BLEU scores, precision, recall, and F1 score over the batch
    accuracy = accuracy / len(target)  # Average accuracy
    average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    average_prec = sum(precisions) / len(precisions) if precisions else 0.0
    average_rec = sum(recalls) / len(recalls) if recalls else 0.0
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return accuracy, average_bleu_score, average_prec, average_rec, average_f1

# At the end of each epoch, append the calculated values
def log_metrics(metrics_dict, validation_loss_avg, trainng_loss_avg, val_accuracy_avg, train_accuracy_avg,
                val_precision_avg, train_precision_avg, val_recall_avg, train_recall_avg, val_f1_avg, train_f1_avg,
                train_bleu_score_avg, val_bleu_score_avg, count, count_t):
    
    if count==0 or count_t == 0:
        metrics_dict["loss_validate"].append(-1)
        metrics_dict["loss_training"].append(-1)
        metrics_dict["val_accuracy_avg"].append(-1)
        metrics_dict["train_accuracy_avg"].append(-1)
        metrics_dict["val_precision_avg"].append(-1)
        metrics_dict["train_precision_avg"].append(-1)
        metrics_dict["val_recall_avg"].append(-1)
        metrics_dict["train_recall_avg"].append(-1)
        metrics_dict["val_f1_avg"].append(-1)
        metrics_dict["train_f1_avg"].append(-1)
        metrics_dict["train_bleu_score_avg"].append(-1)
        metrics_dict["val_bleu_score_avg"].append(-1)
    else:
        metrics_dict["loss_validate"].append(validation_loss_avg / count)
        metrics_dict["loss_training"].append(trainng_loss_avg / count_t)
        metrics_dict["val_accuracy_avg"].append(val_accuracy_avg / count)
        metrics_dict["train_accuracy_avg"].append(train_accuracy_avg / count_t)
        metrics_dict["val_precision_avg"].append(val_precision_avg / count)
        metrics_dict["train_precision_avg"].append(train_precision_avg / count_t)
        metrics_dict["val_recall_avg"].append(val_recall_avg / count)
        metrics_dict["train_recall_avg"].append(train_recall_avg / count_t)
        metrics_dict["val_f1_avg"].append(val_f1_avg / count)
        metrics_dict["train_f1_avg"].append(train_f1_avg / count_t)
        metrics_dict["train_bleu_score_avg"].append(train_bleu_score_avg / count_t)
        metrics_dict["val_bleu_score_avg"].append(val_bleu_score_avg / count)

def handle_half_for_layerNorm(model):
    # Half does not work with some layers
    for layer in model.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.float()

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


#TODO: we need to make it so that only LORA weights get saved/loaded if they are used
#TODO: Training plan
#TODO: Add hyperparameter checks to make sure non conflict
#TODO: take a look at using biomed as well
def feature_aliginment_training_step_2_GPU_SPLIT(
        clip_transformer_width,
        clip_transformer_layers,
        clip_transformer_heads,
        clip_vision_width,
        clip_vision_patch_size,
        clip_vision_layers,
        clip_model_path,
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
        save=False,
        cpu_only=False,
        hidden_layer_from_end=0,
        training_step=1, 
        val_dataset = None,
        train_dataset = None,
        norm=False,
        visual_encoder_type="CLIP-trained"
        ):
    # CHECK GPU SUPPORT AND ASSIGN DEVICES
    device_vit, device_llm = handle_devices(cpu_only)
    
    accumulation_steps = vir_batch_size // batch_size

    # LOAD DATA
    if val_dataset == None or train_dataset == None:
        train_loader, validate_loader = load_data(
            transforms.Compose([
                transforms.Resize((image_resolution, image_resolution)),
                transforms.ToTensor()
            ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
        )
    else:
        train_loader = train_dataset
        validate_loader = val_dataset

    # Load connector and vicuna model
    connector_llm = Connector_LLM(embed_dim=embed_dim,vicuna_path=vicuna_path,connector_layers=connector_layers, device=device_llm, accumulation_steps=accumulation_steps,norm=norm)

    if training_step == 2:
        if pre_trained_connector_path != None:
            print("Loading the connector MLP")
            #Load the pre_trained connector stat_dict
            state_dict = torch.load(pre_trained_connector_path)
            connector_llm.connector.load_state_dict(state_dict)
        else:
            print("No connector given, training from scratch")

        #lora
        connector_llm.apply_lora(rank=lora_rank,dropout=lora_dropout)

    #Half the size of weights for the connector and LLM
    connector_llm.half()
    
    # Half does not work with some layers
    handle_half_for_layerNorm(connector_llm.connector)
    

    if visual_encoder_type == "CLIP-trained":
        # LOAD ViT encoder from the CLIP model on the first GPU
        print("loading our visual encoder")
        img_encoder = load_ViT_img_encoder(
            device=device_vit,
            tokenizer=connector_llm.tokenizer,
            transformer_width=clip_transformer_width,
            transformer_layers=clip_transformer_layers,
            transformer_heads=clip_transformer_heads,
            embed_dim=embed_dim,
            vision_width=clip_vision_width,
            image_resolution=image_resolution,
            vision_patch_size=clip_vision_patch_size,
            vision_layers=clip_vision_layers,
            clip_model_path=clip_model_path
            )
        
    elif visual_encoder_type == "CLIP-pretrained":
        print("loading pretrained CLIP visual encoder")
        clip = CLIPWithLoRA()
        img_encoder = clip.get_visual_encoder()
    else:
        print("loading fine-tuned CLIP model")
        clip = CLIPWithLoRA()
        clip.apply_LORA(lora_r=8, lora_alpha=32, lora_dropout=0.1)
        clip.load_model(clip_model_path)
        img_encoder = clip.get_visual_encoder()

    # FREEZE CLIP TRAINING (should save memory and computation as well)
    img_encoder.eval()

    # half the size of its weights to save memory
    img_encoder.half()

    # Half does not work with some layers
    handle_half_for_layerNorm(img_encoder)

    if training_step == 1:
        #freeze vicuna training
        connector_llm.vicuna.eval()

    total_training_steps = math.ceil(len(train_loader.dataset) / vir_batch_size) * MAX_EPOC
    num_warmup_steps = math.ceil(total_training_steps *  per_warm)

    print(num_warmup_steps , " warmup steps")
    print(total_training_steps, " total steps")


    # Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector_llm.parameters(), lr=lr,weight_decay=weight_decay, eps=eps)
    scheduler = CustomCosineSchedulerWithWarmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=total_training_steps,training_step=training_step)
    connector_llm.set_optim_scheduler(optim, scheduler)

    # to store metrics if the function is being used with cross_val
    metrics_dict = {
    "loss_validate": [],
    "loss_training": [],
    "val_accuracy_avg": [],
    "train_accuracy_avg": [],
    "val_precision_avg": [],
    "train_precision_avg": [],
    "val_recall_avg": [],
    "train_recall_avg": [],
    "val_f1_avg": [],
    "train_f1_avg": [],
    "train_bleu_score_avg": [],
    "val_bleu_score_avg": []
    }

    for n in range(1, MAX_EPOC + 1):
        
        # Training the LLM is not needed in step 1
        if training_step == 2:
            connector_llm.train()
        else:
            connector_llm.connector.train()

        trainng_loss_avg = 0.0
        train_accuracy_avg = 0.0
        train_precision_avg = 0.0
        train_recall_avg = 0.0
        train_f1_avg = 0.0
        train_bleu_score_avg = 0.0
        count_t = 0
        optim.zero_grad()

        for image_tensor, mask_tensor, question, answer in train_loader:
            try:
                # Get image features from the img encoder
                with torch.no_grad():
                    image_features, hidden_states = img_encoder(image_tensor.half().to(device_vit),return_hidden_states=True)

                # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                # Format data and "tokenize" answer for the LLM and eval metrics
                answer_ =  connector_llm.tokenizer([a + "</s>" for a in answer],padding='longest',truncation=True,return_tensors='pt').input_ids[:,1:].to(device_llm)

                # Get MLLM prediction and NLL loss
                output, loss= connector_llm(image_features.to(device_llm), question, answer_, answer_.size(1), count_t)

                # Eval
                accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics([batch_tensor for batch_tensor in output],[batch_tensor for batch_tensor in answer_],tokenizer=connector_llm.tokenizer)

                               
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('Skipping batch due to OOM')
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    print("Error:")
                    print(e)
                    print("Skipping batch")
                continue
                            
            trainng_loss_avg += loss
            train_accuracy_avg += accuracy
            train_precision_avg += precision
            train_recall_avg += recall
            train_f1_avg += f1
            train_bleu_score_avg += bleu_score
            count_t = count_t + 1

        # Ensure to perform a step if we have leftover gradients
        if (count_t + 1) % accumulation_steps != 0:
            optim.step()
            optim.zero_grad()
            scheduler.step()
            connector_llm.zero_grad()

        # VALIDATE
        validation_loss_avg = 0.0
        val_accuracy_avg = 0.0
        val_precision_avg = 0.0
        val_recall_avg = 0.0
        val_f1_avg = 0.0
        val_bleu_score_avg = 0.0
        count = 0
        connector_llm.eval()

        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer in validate_loader:
                try:
                    # Get image features from the img encoder
                    image_features, hidden_states = img_encoder(image_tensor.half().to(device_vit),return_hidden_states=True)

                    # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                    image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                    # Format data and "tokenize" answer for the LLM and eval metrics
                    answer_ =  connector_llm.tokenizer([a + "</s>" for a in answer],padding='longest',truncation=True,return_tensors='pt').input_ids[:,1:].to(device_llm)

                    # Get MLLM prediction and NLL loss
                    output, loss= connector_llm(image_features.to(device_llm), question, answer_, answer_.size(1), count_t)

                    # Eval
                    accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics([batch_tensor for batch_tensor in output],[batch_tensor for batch_tensor in answer_],tokenizer=connector_llm.tokenizer)

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('Skipping batch due to OOM')
                        print(e)
                        torch.cuda.empty_cache()
                    else:
                        print("Error:")
                        print(e)
                        print("Skipping batch")
                    continue

                validation_loss_avg += loss
                val_accuracy_avg += accuracy
                val_precision_avg += precision
                val_recall_avg += recall
                val_f1_avg += f1
                val_bleu_score_avg += bleu_score
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
                    wandb.log({
                        "loss_validate": validation_loss_avg / count,
                        "loss_training": trainng_loss_avg / count_t,
                        "val_accuracy_avg": val_accuracy_avg / count,
                        "train_accuracy_avg": train_accuracy_avg / count_t,
                        "val_precision_avg": val_precision_avg / count,
                        "train_precision_avg": train_precision_avg / count_t,
                        "val_recall_avg": val_recall_avg / count,
                        "train_recall_avg": train_recall_avg / count_t,
                        "val_f1_avg": val_f1_avg / count,
                        "train_f1_avg": train_f1_avg / count_t,
                        "train_bleu_score_avg": train_bleu_score_avg / count_t,
                        "val_bleu_score_avg": val_bleu_score_avg / count
                    })
            else:
                wandb.log({
                    "loss_validate": -1,
                    "loss_training": -1,
                    "val_accuracy_avg": -1,
                    "train_accuracy_avg": -1,
                    "val_precision_avg": -1,
                    "train_precision_avg": -1,
                    "val_recall_avg": -1,
                    "train_recall_avg": -1,
                    "val_f1_avg": -1,
                    "train_f1_avg": -1,
                    "train_bleu_score_avg": -1,
                    "val_bleu_score_avg": -1
                })
        else:
            log_metrics(metrics_dict,validation_loss_avg, trainng_loss_avg, val_accuracy_avg, train_accuracy_avg,
            val_precision_avg, train_precision_avg, val_recall_avg, train_recall_avg, val_f1_avg, train_f1_avg,
            train_bleu_score_avg, val_bleu_score_avg, count, count_t)

    return metrics_dict

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
    metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{n_splits}')
        
        # Create data loaders for training and validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=para["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(para["rand_seed"]))
        val_loader = DataLoader(val_subset, batch_size=para["batch_size"])
        metrics_list.append(feature_aliginment_training_step_2_GPU_SPLIT(**para, train_dataset=train_loader, val_dataset=val_loader))

    # Initialize a dictionary with lists to accumulate sums for each metric
    accumulated_metrics = defaultdict(list)
    
    # Loop through each metrics_dict in the list
    for metrics_dict in metrics_list:
        for key, value in metrics_dict.items():
            accumulated_metrics[key].append(value)
    
    # Calculate the average for each metric
    avg_metrics = {}
    for key, values in accumulated_metrics.items():
        avg_metrics[key] = [sum(x) / len(values) for x in zip(*values)] if values else None  # Compute average

    num_indices = len(next(iter(avg_metrics.values())))

    split_data = []

    # Loop over the range of indices
    for i in range(num_indices):
        # Create a new dictionary for each index
        dict_at_index = {key: value[i] for key, value in avg_metrics.items()}
        split_data.append(dict_at_index)

    for step in split_data:
        wandb.log(step)
        
#path1 = os.path.join(os.getcwd(), "Models_to_upload","v_2000", "clip_model_30.pth")
path1 = os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload" , "V_" + str(10320005),"clip_model_" + str(23) + ".pth")
#path = os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
path = os.path.join("/nobackup","sc20gwb","Models", "vicuna-7b-v1.5")
path3 = os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(3000), "connector_LLM_model" + str(2) + ".pth")



# LR_LIST = [1e-4]

# WEIGHT_DECAY_LIST = [1e-4]

# PERC_WARM_LIST = [0.0]

# VIR_BATCH_SIZE_LIST = [32]

# NORM_LIST = [False]

# DROPOUT_LIST = [0.1]

# RANK_LIST = [8]

# HIDDEN_LAYER_LIST = [1]

# CONNECTOR_LAYERS_LIST = [2]


LR_LIST = [1e-7,5e-7]

WEIGHT_DECAY_LIST = [1e-4]

PERC_WARM_LIST = [0.1]

VIR_BATCH_SIZE_LIST = [32]

NORM_LIST = [False]

DROPOUT_LIST = [0.1]

RANK_LIST = [10]

HIDDEN_LAYER_LIST = [1]

CONNECTOR_LAYERS_LIST = [2]

CONNECTOR_LIST = [path3]

# batch_size 4 for step 2, 8 for step 1

optim_list = [{
        "clip_transformer_width":512,
        "clip_transformer_layers":6,
        "clip_transformer_heads":8,
        "clip_vision_width":512,
        "clip_vision_patch_size":56,
        "clip_vision_layers":6,
        "clip_model_path":path1,
        "vicuna_path":path,
        "connector_layers":cl,
        "embed_dim":512,
        "image_resolution":224,
        "lr": lr,
        "eps":1e-6,
        "weight_decay":wd,
        "per_warm": pw,
        "batch_size":4,
        "vir_batch_size":vb,
        "rand_seed":42,
        "MAX_EPOC":8,
        "VERSION":3000,
        "pre_trained_connector_path":cp,
        "save":False,
        "cpu_only":False,
        "hidden_layer_from_end": hl,
        "training_step":2,
        "lora_dropout":do,
        "lora_rank":r,
        "norm":  norm
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
            for cp in CONNECTOR_LIST
            ]

for i, para in enumerate(optim_list):
    para['VERSION'] += i
    wandb.init(project="MSc_fine_tuning_step_2",config=para)
    #print("Cross Validation for VERSION ", para["VERSION"])
    feature_aliginment_training_step_2_GPU_SPLIT(**para)
    #cross_val_train(para,n_splits=3,per_data=0.1)
    wandb.finish()