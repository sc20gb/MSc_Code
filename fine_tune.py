import torchvision.transforms as transforms
import os
from Data_Loading.data_loading import load_data
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

    # Iterate over each sample in the batch
    for i in range(target.size(0)):  # Assuming target has shape (batch_size, ...)
        # Extract the individual target and predicted for the current batch item
        target_item = target[i]
        predicted_item = predicted[i]
        
        # Ensure the answer has its capitals and whitespace removed
        predicted_string = process_string(tokenizer.decode(predicted_item.long(), skip_special_tokens=True))
        target_string = process_string(tokenizer.decode(target_item.long(), skip_special_tokens=True))

        print(predicted_string)
        print(target_string)

        predicted_list = predicted_string.split()
        target_list = target_string.split()

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
            common_tokens = set(predicted_list) & set(target_list)
            prec = len(common_tokens) / len(predicted_list)
            rec = len(common_tokens) / len(target_list)

        if prec + rec == 0.0:
            f1 = 0.0
        else:
            f1 = 2 * (prec * rec) / (prec + rec)

        bleu_scores.append(bleu_score_)
        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    # Average the accuracy, BLEU scores, precision, recall, and F1 score over the batch
    accuracy /= target.size(0)  # Average accuracy
    average_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    average_prec = sum(precisions) / len(precisions) if precisions else 0.0
    average_rec = sum(recalls) / len(recalls) if recalls else 0.0
    average_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return accuracy, average_bleu_score, average_prec, average_rec, average_f1

  



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
        save=False,
        cpu_only=False,
        hidden_layer_from_end=0,
        training_step=1
        ):
    # CHECK GPU SUPPORT AND ASSIGN DEVICES
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


    if training_step != 1 and training_step !=2:
        print("Training Step must be 1 or 2 not ", training_step)
        return 0
    
    accumulation_steps = vir_batch_size // batch_size

    # LOAD DATA
    train_loader, validate_loader = load_data(
        transforms.Compose([
            transforms.Resize((image_resolution, image_resolution)),
            transforms.ToTensor()
        ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
    )

    # Load connector and vicuna model on the second GPU
    connector_llm = Connector_LLM(embed_dim=embed_dim,vicuna_path=vicuna_path,connector_layers=connector_layers, device=device_llm, accumulation_steps=accumulation_steps)

    if training_step == 2:
        #Load the pre_trained connector stat_dict
        state_dict = torch.load(pre_trained_connector_path)

        connector_llm.connector.load_state_dict(state_dict)

        #lora
        connector_llm.apply_lora()

    #Half the size of weights for the connector and LLM
    connector_llm.half()
    
    # LOAD ViT encoder from the CLIP model on the first GPU
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
    
    # FREEZE CLIP TRAINING (should save memory and computation as well)
    img_encoder.eval()

    # half the size of its weights to save memory
    img_encoder.half()
    
    # Half does not work with some layers
    for layer in img_encoder.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.float()

    if training_step == 1:
        #freeze vicuna training
        connector_llm.vicuna.eval()

    # Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector_llm.parameters(), lr=lr,weight_decay=weight_decay, eps=eps)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=math.ceil(MAX_EPOC * per_warm), num_training_steps=MAX_EPOC)
    connector_llm.set_optim_scheduler(optim,scheduler)

    # Record the loss at the epoch
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

        print(f"Memory allocated before loop: {torch.cuda.memory_allocated() / 1e6} MB")

        mem_alloc = torch.cuda.memory_allocated() / 1e6
        

        for image_tensor, mask_tensor, question, answer in train_loader:
            try:
                # Get image features from the img encoder
                with torch.no_grad():
                    image_features, hidden_states = img_encoder(image_tensor.half().to(device_vit),return_hidden_states=True)

                # We want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                # Format data and "tokenize" answer for the LLM and eval metrics
                answer_ =  connector_llm.tokenizer([a + "</s>" for a in answer],padding='longest',truncation=True,return_tensors='pt').input_ids[:,1:].to(device_llm)

                answer_temp = answer_.clone()

                # Get MLLM prediction and NLL loss
                output, loss= connector_llm(image_features.to(device_llm), question, answer_temp, answer_temp.size(1), count_t)

                # Eval
                accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(output,answer_,tokenizer=connector_llm.tokenizer)

                               
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

            print("Diff in mem = ", mem_alloc - (torch.cuda.memory_allocated() / 1e6))

        # Ensure to perform a step if we have leftover gradients
        if (count_t + 1) % accumulation_steps != 0:
            optim.step()
            optim.zero_grad()
            connector_llm.zero_grad()

        scheduler.step()

        # VALIDATE
        validation_loss_avg = torch.tensor([0.0])
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
                    answer_ =  connector_llm.tokenizer([a + "</s>" for a in answer],padding='longest',truncation=True,return_tensors='pt').input_ids[:,1:].half().to(device_llm)

                    # Get MLLM prediction and NLL loss
                    output, loss= connector_llm(image_features.to(device_llm), question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]), count_t)

                    # Eval
                    accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(output,answer_,tokenizer=connector_llm.tokenizer)

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


        if count != 0 and count_t != 0:
            wandb.log({
                "loss_validate": validation_loss_avg.to('cpu').detach().numpy()[0] / count,
                "loss_training": trainng_loss_avg.to('cpu').detach().numpy()[0] / count_t,
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

    return 0

#path1 = os.path.join(os.getcwd(), "Models_to_upload","v_2000", "clip_model_30.pth")
path1 = os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload" , "V_" + str(10320005),"clip_model_" + str(23) + ".pth")
#path = os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
path = os.path.join("/nobackup","sc20gwb","Models", "vicuna-7b-v1.5")
path3 = os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(1000), "connector_LLM_model" + str(1) + ".pth")

LR_LIST = [0.00001,0.0001,0.001]

WEIGHT_DECAY_LIST = [0.0001]

CONNECTOR_LAYERS_LIST = [2,3]

PERC_WARM_LIST = [0.2]

VIR_BATCH_SIZE_LIST = [32]

HIDDEN_LAYER_LIST = [0,1]

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
        "eps":0.0001,
        "weight_decay":wd,
        "per_warm": pw,
        "batch_size":16,
        "vir_batch_size":vb,
        "rand_seed":42,
        "MAX_EPOC":30,
        "VERSION":2000,
        "pre_trained_connector_path":path3,
        "save":False,
        "cpu_only":False,
        "hidden_layer_from_end": hl,
        "training_step":2
            }
            for lr in LR_LIST 
            for wd in WEIGHT_DECAY_LIST 
            for cl in CONNECTOR_LAYERS_LIST
            for pw in PERC_WARM_LIST
            for vb in VIR_BATCH_SIZE_LIST
            for hl in HIDDEN_LAYER_LIST
            ]

for i, para in enumerate(optim_list):
    para['VERSION'] += i
    wandb.init(project="MSc_fine_tuning_step_2",config=para)
    feature_aliginment_training_step_2_GPU_SPLIT(**para)
    wandb.finish()


