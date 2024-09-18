import torchvision.transforms as transforms
import os
from data_loading import load_test_data
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

def run_test_itr( clip_parameters,
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
        hidden_layer_from_end=0):
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

        # LOAD DATA
        test_loader = load_test_data(
            transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor()
            ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
        )

        # Load connector and vicuna model on the second GPU
        connector_llm = Connector_LLM(**connector_llm_parameters, device=torch.device("cpu"), MAX_LENGTH=MAX_LENGTH_LLM)

        
        #Load the pre_trained connector stat_dict
        state_dict = torch.load(pre_trained_connector_path)

        connector_llm.load_state_dict(state_dict)

        connector_llm.eval()

        #Half the size of weights for the connector and LLM
        connector_llm.half()

        connector_llm.to(device_llm)

        connector_llm.device = device_llm
        
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


    # TEST
        validation_loss_avg = torch.tensor([0.0])

        val_accuracy_avg_OPEN = 0.0
        val_precision_avg_OPEN = 0.0
        val_recall_avg_OPEN = 0.0
        val_f1_avg_OPEN = 0.0
        val_bleu_score_avg_OPEN = 0.0

        val_accuracy_avg_CLOSED = 0.0
        val_precision_avg_CLOSED = 0.0
        val_recall_avg_CLOSED = 0.0
        val_f1_avg_CLOSED = 0.0
        val_bleu_score_avg_CLOSED = 0.0

        count_OPEN = 0
        count_CLOSED = 0
        connector_llm.eval()
        connector_llm.w_vicuna.eval()
        connector_llm.connector.eval()

        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer,catogory in test_loader:
                print("Validation itr ", str(count), " of ", len(test_loader))

                try:
                    # Get image features from the img encoder (on GPU 0)
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
                    output, loss= connector_llm(image_features, question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]), 0)

    
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
                        torch.cuda.empty_cache()
                    else:
                        print("Error:")
                        print(e)
                        print("Skipping batch")
                    continue
                # dont worry about the metrics here the values should be the same as +=  0.0, also the count only increases after the continue so the loss avg is fine too
                


                if catogory == "OPEN":
                    val_accuracy_avg_OPEN += loss
                    val_precision_avg_OPEN += accuracy
                    val_recall_avg_OPEN += recall
                    val_f1_avg_OPEN += f1
                    val_bleu_score_avg_OPEN += bleu_score
                    count_OPEN += 1
                else:
                    val_accuracy_avg_CLOSED += loss
                    val_precision_avg_CLOSED += accuracy
                    val_recall_avg_CLOSED += recall
                    val_f1_avg_CLOSED += f1
                    val_bleu_score_avg_CLOSED += bleu_score
                    count_CLOSED += 1

            wandb.log({
                "val_accuracy_avg_OPEN": val_accuracy_avg_OPEN/count_OPEN,
                    "val_precision_avg_OPEN": val_precision_avg_OPEN/count_OPEN,
                    "val_recall_avg_OPEN": val_recall_avg_OPEN/count_OPEN,
                    "val_f1_avg_OPEN": val_f1_avg_OPEN/count_OPEN,
                    "val_bleu_score_avg_OPEN":val_bleu_score_avg_OPEN/count_OPEN,

                    "val_accuracy_avg_CLOSED": val_accuracy_avg_CLOSED/count_CLOSED,
                    "val_precision_avg_CLOSED": val_precision_avg_CLOSED/count_CLOSED,
                    "val_recall_avg_CLOSED": val_recall_avg_CLOSED/count_CLOSED,
                    "val_f1_avg_CLOSED": val_f1_avg_CLOSED/count_CLOSED,
                    "val_bleu_score_avg_CLOSED":val_bleu_score_avg_CLOSED/count_CLOSED,

                    "val_accuracy_avg": (val_accuracy_avg_CLOSED + val_accuracy_avg_OPEN)/(count_CLOSED + count_OPEN),
                    "val_precision_avg": (val_precision_avg_CLOSED + val_precision_avg_OPEN)/(count_CLOSED+ count_OPEN),
                    "val_recall_avg": (val_recall_avg_CLOSED + val_recall_avg_OPEN)/(count_CLOSED+ count_OPEN),
                    "val_f1_avg": (val_f1_avg_CLOSED + val_f1_avg_OPEN)/(count_CLOSED+ count_OPEN),
                    "val_bleu_score_avg":(val_bleu_score_avg_CLOSED + val_bleu_score_avg_OPEN)/(count_CLOSED+ count_OPEN)

            })


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



LR_LIST = [0.00001]
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
    run_test_itr(
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
        pre_trained_connector_path=os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "MLLM_V_" + str(1000), "MLLM_model" + str(6) + ".pth"),
        MAX_LENGTH_LLM=p['MAX_LENGTH_LLM'],
        save=p['save'],
        cpu_only=p['cpu_only']
    )

    wandb.finish()








