import torchvision.transforms as transforms
import os
from data_loading import load_data
import torch

import numpy as np

from CLIP import VisionTransformer, CLIP


from transformers import get_cosine_schedule_with_warmup


from connector_LLM import Connector_LLM, print_memory_usage

import sys

import re


import torch.nn.functional as F

from torcheval.metrics.functional import bleu_score

import wandb

import math

import gc

import string

#Return  the object that is the visual encoder, return the heat scaling parameter as well
def load_ViT_img_encoder(tokenizer,transformer_width,MAX_LENGTH,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=MAX_LENGTH,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)

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


def calc_loss_and_metrics(predicted, target, tokenizer, max_length):
    # We need it to be a list of tensors instead
    # Pad the predicted tensors after the </s> to the unk token [....,answer,2,44235,3153,...] -> [answer,2]
    target = target[0]

    # Calc accuracy
    accuracy = 0

    # we need to ensure that the answer has its captials and its whitespace removed
    predcted_string = process_string(tokenizer.decode(predicted, skip_special_tokens=True))

    print(predcted_string)

    target_string = process_string(tokenizer.decode(target, skip_special_tokens=True))

    print(target_string)

    predicted_list = predcted_string.split()
    target_list = target_string.split()

    print(predicted_list)
    print(target_list)

    if predicted_list == target_list:
        accuracy += 1.0

    if len(target_list) == 0 or len(predicted_list) == 0:
        bleu_score_ = 0.0
    else:
        print(predcted_string, target_string)
        bleu_score_ = bleu_score(
            predcted_string,
            [target_string],
            n_gram=1).item()

    # https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
    # precision here is the number of shared words / len(predict)
    # recall is the number of shared words / len(target)

    prec = 0.0
    rec = 0.0
    if len(predcted_string) != 0 and len(target_string) != 0:
        common_tokens = set(predicted_list) & set(target_list)
        prec = len(common_tokens) / len(predicted_list)
        rec = len(common_tokens) / len(target_list)

    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)

    return accuracy, bleu_score_, prec, rec, f1


def feature_aliginment_training_step_2_GPU_SPLIT(
        clip_parameters,
        optim_parameters,
        connector_llm_parameters,
        per_warm,image_size,
        batch_size,
        vir_batch_size,
        rand_seed,
        MAX_EPOC,
        MAX_LENGTH,
        MAX_LENGTH_LLM,
        VERSION,
        pre_trained_connector_path,
        save=False,
        cpu_only=False,
        hidden_layer_from_end=0
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
            print("Only one GPU available, models are split between CPU and GPU 0")
            device_vit = torch.device("cpu")
            device_llm = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device_vit = torch.device("cpu")
        device_llm = torch.device("cpu")

    accumulation_steps = vir_batch_size // batch_size


    torch.cuda.reset_peak_memory_stats()

    # Check memory before loading the model
    print(f"Memory allocated before any: {torch.cuda.memory_allocated() / 1e6} MB")

    # LOAD DATA
    train_loader, validate_loader = load_data(
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
    )

    # Check memory after loading the model
    print(f"Memory allocated after dataloaders: {torch.cuda.memory_allocated() / 1e6} MB")


    # Load connector and vicuna model on the second GPU
    connector_llm = Connector_LLM(**connector_llm_parameters, device=device_llm, MAX_LENGTH=MAX_LENGTH_LLM)

    #Load the pre_trained connector stat_dict
    state_dict = torch.load(pre_trained_connector_path)

    connector_llm.connector.load_state_dict(state_dict)

    #Half the size of weights for the connector and LLM
    connector_llm.half()
    
    # Check memory after loading the model
    print(f"Memory allocated after connector_LLM: {torch.cuda.memory_allocated() / 1e6} MB")


    # LOAD ViT encoder from the CLIP model on the first GPU
    img_encoder = load_ViT_img_encoder(**clip_parameters, device=device_vit, tokenizer=connector_llm.tokenizer, MAX_LENGTH=MAX_LENGTH)

    # FREEZE CLIP TRAINING (should save memory and computation as well)
    img_encoder.eval()

    # half the size of its weights to save memory
    img_encoder.half()

    
    # Half does not work with some layers
    for layer in img_encoder.modules():
        if isinstance(layer, torch.nn.LayerNorm):
            layer.float()


        
    # Check memory after loading the model
    print(f"Memory allocated after connector_LLM: {torch.cuda.memory_allocated() / 1e6} MB")


    # Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector_llm.parameters(), **optim_parameters)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=math.ceil(MAX_EPOC * per_warm), num_training_steps=MAX_EPOC)

    # Record the loss at the epoch
    for n in range(1, MAX_EPOC + 1):
        connector_llm.train()
        connector_llm.vicuna.train()
        connector_llm.connector.train()
        trainng_loss_avg = torch.tensor([0.0])
        train_accuracy_avg = 0.0
        train_precision_avg = 0.0
        train_recall_avg = 0.0
        train_f1_avg = 0.0
        train_bleu_score_avg = 0.0
        count_t = 0
        count_q = 0
        optim.zero_grad()


        print(f"Memory allocated before loop: {torch.cuda.memory_allocated() / 1e6} MB")

        mem_alloc = torch.cuda.memory_allocated() / 1e6
        

        for image_tensor, mask_tensor, question, answer in train_loader:
            print(" Train itr ", str(count_t), " of ", len(train_loader))

            if count_t > 300:
                break

            gc.collect()

            torch.cuda.empty_cache()
                        
            # Check memory after loading the model
            print(f"Memory allocated after clearing cache at start of itr: {torch.cuda.memory_allocated() / 1e6} MB")
            torch.cuda.reset_peak_memory_stats()
            
            try:

                # Get image features from the img encoder (on GPU 0)
                with torch.no_grad():
                    image_features, hidden_states = img_encoder(image_tensor.half().to(device_vit),return_hidden_states=True)


                #we want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]
                
                # Move image features to the second GPU for LLM processing
                image_features = image_features.to(device_llm)

                # Format data and "tokenize" inputs for the LLM
                answer_ = [connector_llm.tokenizer(a + "</s>", return_tensors="pt", max_length=MAX_LENGTH).input_ids for a in answer]

                for i, a in enumerate(answer_):
                    if len(a) < MAX_LENGTH:
                        answer_[i] = F.pad(a, (0, MAX_LENGTH - a.size(0)), 'constant', 0)

                answer_ = torch.cat(answer_, dim=0)[:, 1:].half().to(device_llm)

                               
 

                # here max(len(s) for s in answer) + 2 ,ensures that there is an extra loss for not finding the eos token, while also reducing memory
                output, loss= connector_llm(image_features, question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]))


 
                accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(
                    output,
                    [connector_llm.tokenizer(a + "</s>", return_tensors="pt").input_ids[:, 1:].flatten().to(device_llm) for a in answer],
                    tokenizer=connector_llm.tokenizer,
                    max_length=MAX_LENGTH_LLM
                )

                               
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('Skipping batch due to OOM')
                    print(e)
                    sys.exit()
                else:
                    print("Error:")
                    print(e)
                    print("Skipping batch")
                continue
            

            
            # Perform the optimizer step after accumulating the gradients for `accumulation_steps` batches
            if (count_t + 1) % accumulation_steps == 0:
                optim.step()
                optim.zero_grad()
                connector_llm.delete_non_weight_vars()
                            
            connector_llm.zero_grad()


            trainng_loss_avg += loss

            train_accuracy_avg += accuracy
            train_precision_avg += precision
            train_recall_avg += recall
            train_f1_avg += f1
            train_bleu_score_avg += bleu_score
            count_t = count_t + 1
            count_q += answer_.size(0)


  
            # Check memory after loading the model
            print(f"Memory allocated after optim and zero_grad and emptying cache: {torch.cuda.memory_allocated() / 1e6} MB")

            print("Diff in mem = ", mem_alloc - (torch.cuda.memory_allocated() / 1e6))

        # Ensure to perform a step if we have leftover gradients
        if (count_t + 1) % accumulation_steps != 0:
            optim.step()
            optim.zero_grad()

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
        connector_llm.vicuna.eval()
        connector_llm.connector.eval()

        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer in validate_loader:
                print("Validation itr ", str(count), " of ", len(validate_loader))

                if count > 100:
                    break

                try:
                    # Get image features from the img encoder (on GPU 0)
                    with torch.no_grad():
                        image_features, hidden_states = img_encoder(image_tensor.half().to(device_vit),return_hidden_states=True)


                    #we want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                    image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]
                    
                    # Move image features to the second GPU for LLM processing
                    image_features = image_features.to(device_llm)
                

                    # Format data and "tokenize" inputs for the LLM
                    answer_ = [connector_llm.tokenizer(a + "</s>", return_tensors="pt", max_length=MAX_LENGTH).input_ids for a in answer]

                    for i, a in enumerate(answer_):
                        if len(a) < MAX_LENGTH:
                            answer_[i] = F.pad(a, (0, MAX_LENGTH - a.size(0)), 'constant', 0)

                    answer_ = torch.cat(answer_, dim=0)[:, 1:].half().to(device_llm)


                    # here max(len(s) for s in answer) + 2 ,ensures that there is an extra loss for not finding the eos token, while also reducing memory
                    output, loss = connector_llm(image_features, question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]))


                    accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(
                        output,
                        [connector_llm.tokenizer(a + "</s>", return_tensors="pt").input_ids[:, 1:].flatten().to(device_llm) for a in answer],
                        tokenizer=connector_llm.tokenizer,
                        max_length=MAX_LENGTH_LLM
                    )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('Skipping batch due to OOM')
                        print(e)
                        sys.exit()
                    else:
                        print("Error:")
                        print(e)
                        print("Skipping batch")
                    continue
                # dont worry about the metrics here the values should be the same as +=  0.0, also the count only increases after the continue so the loss avg is fine too

                

                validation_loss_avg += loss
                val_accuracy_avg += accuracy
                val_precision_avg += precision
                val_recall_avg += recall
                val_f1_avg += f1
                val_bleu_score_avg += bleu_score
                count = count + 1

        # SAVE RESULTS
        if save:
            if not os.path.exists(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION))):
                os.makedirs(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION)))
            torch.save(connector_llm.state_dict(), os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(VERSION), "MLLM_model" + str(n) + ".pth"))

        if count != 0 and count_t != 0:
            wandb.log({
                "loss_validate": validation_loss_avg.to('cpu').detach().numpy()[0] / count,
                "loss_training": trainng_loss_avg.to('cpu').detach().numpy()[0] / count_t,
                "val_accuracy_avg": val_accuracy_avg / count,
                "train_accuracy_avg": train_accuracy_avg / count_q,
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

#/nobackup/sc20gwb/Models/Models_to_upload
#path1 = os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload", "clip_model_30.pth")
#path1 = os.path.join(os.getcwd(), "Models_to_upload","v_2000", "clip_model_30.pth")
path1 = os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload" , "V_" + str(10320005),"clip_model_" + str(23) + ".pth")
clip_parameters  =  {
"transformer_width":512,
"transformer_layers":6,
"transformer_heads":8,
"embed_dim":512,
"vision_width":512,
"image_resolution":224,
"vision_patch_size":56,
"vision_layers":6,
"clip_model_path": path1

}



LR_LIST = [0.0005]
#WEIGHT_DECAY_LIST = [0.0001,0.001,0.00001]
WEIGHT_DECAY_LIST = [0.0001]

optim_list = [{

"lr":  lr,

"eps":0.0001,

"weight_decay":wd

} for lr in LR_LIST 
for wd in WEIGHT_DECAY_LIST ]


#Vicuna Path os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
#path = os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
path = os.path.join("/nobackup","sc20gwb","Models", "vicuna-7b-v1.5")
connector_llm_parameters = {
"vicuna_path":path,
"embed_dim": 512, # this is the width of the CLIP ViT
"connector_layers":2
}

# Additional parameters from the function call
additional_parameters = {
    "per_warm": 0.2,
    "image_size": 224,
    "batch_size": 1,
    "vir_batch_size": 32,
    "rand_seed": 42,
    "MAX_EPOC": 30,
    "MAX_LENGTH": 256,
    "VERSION": 2000,
    "MAX_LENGTH_LLM": 48,
    "save": True,
    "cpu_only": False,
    "hidden_layer_from_end": 0
}


for i, para in enumerate(optim_list):
    p = {**connector_llm_parameters,**para,**clip_parameters,**additional_parameters}
    wandb.init(
        project="MSc_fine_tuning_step_2",
        config=p
    )
    feature_aliginment_training_step_2_GPU_SPLIT(
        clip_parameters=clip_parameters,
        optim_parameters=para,
        connector_llm_parameters=connector_llm_parameters,
        per_warm=p['per_warm'],
        image_size=p['image_size'],
        batch_size=p['batch_size'],
        vir_batch_size=p['vir_batch_size'],
        rand_seed=p['rand_seed'],
        MAX_EPOC=p['MAX_EPOC'],
        MAX_LENGTH=p['MAX_LENGTH'],
        VERSION=(i + 1)*1000,
        pre_trained_connector_path=os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(1000), "connector_LLM_model" + str(1) + ".pth"),
        MAX_LENGTH_LLM=p['MAX_LENGTH_LLM'],
        save=p['save'],
        cpu_only=p['cpu_only']
    )

    wandb.finish()


