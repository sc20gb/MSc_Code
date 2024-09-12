import torchvision.transforms as transforms
import os
from data_loading import load_data
import torch

import numpy as np

from CLIP import VisionTransformer


from transformers import LlamaForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup, CLIPVisionModel


from connector_LLM import Connector_LLM


import torch.nn.functional as F

from torcheval.metrics.functional import bleu_score

import wandb

import math

#Return  the object that is the visual encoder, return the heat scaling parameter as well
def load_ViT_img_encoder(tokenizer,transformer_width,MAX_LENGTH,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    # clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=MAX_LENGTH,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)
    # clip.load_state_dict(torch.load(clip_model_path))

    #LOAD just the parameters for the vit to save mem
    # Step 1: Load the full state dict
    state_dict = torch.load(clip_model_path)
    # Step 2: Filter out only the 'visual' (VisionTransformer) parameters
    vision_state_dict = {k: v for k, v in state_dict.items() if k.startswith('visual')}

    for k in state_dict.items():
        print(k)

    print(vision_state_dict)


    vision_heads = vision_width // 64
    visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

    # Step 3: Load the filtered state dict into the 'visual' part of your CLIP model
    # Assuming 'model' is your CLIP model instance
    visual.load_state_dict(vision_state_dict, strict=True)
    return visual.to(device)#clip.visual.to(device)

def calc_loss_and_metrics(predicted,target,tokenizer,max_length):

    # We need it to be a list of tensors instead 
    # Pad the predicted tensors after the </s> to the unk token [....,answer,2,44235,3153,...] -> [answer,2]

    target = target[0]

    # Calc accuracy
    accuracy = 0
    # This here is the same as EM see the link below
    if predicted.size(0) == target.size(0):
        accuracy += (predicted == target).all()


    print(tokenizer.decode(predicted,skip_special_tokens=True))

    print(tokenizer.decode(target,skip_special_tokens=True))

    # this score is calculated from the plain english sentences
    pred = [tokenizer.decode(predicted,skip_special_tokens=True)]

    ans = [[tokenizer.decode(target,skip_special_tokens=True)]]

    if not pred[0] or pred[0].isspace():
        bleu_score_ = 0.0
    else:
        bleu_score_ = bleu_score(
            pred,
            ans,
            n_gram=1)
    
    # https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
    # precision here is the number of shared words / len(predict)
    #recall is the number of shared words / len(target)

    prec = 0.0
    rec = 0.0

    common_tokens = set(predicted) & set(target)
    prec = len(common_tokens) / predicted.size(0)
    rec = len(common_tokens) /  target.size(0)

    if prec + rec == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (prec * rec) / (prec + rec)

    return accuracy, bleu_score_,prec,rec,f1

# This trains the MLP between the visual encoder and LLM. It can be seen as traing a compatible visual tokenizer for the for the frozen LLM
def feature_aliginment_training_step_1_GPU_SPLIT(
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
            print("Only one GPU available, both models will run on GPU 0.")
            device_vit = torch.device("cuda:0")
            device_llm = torch.device("cuda:0")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device_vit = torch.device("cpu")
        device_llm = torch.device("cpu")

    accumulation_steps = vir_batch_size // batch_size

    # LOAD DATA
    train_loader, validate_loader = load_data(
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
    )

    # Load connector and vicuna model on the second GPU
    connector_llm = Connector_LLM(**connector_llm_parameters, device=device_llm, MAX_LENGTH=MAX_LENGTH_LLM)

    # FREEZE vicuna TRAINING (should save memory and computation as well)
    connector_llm.vicuna.eval()

    # LOAD ViT encoder from the CLIP model on the first GPU
    img_encoder = load_ViT_img_encoder(**clip_parameters, device=device_vit, tokenizer=connector_llm.tokenizer, MAX_LENGTH=MAX_LENGTH)

    # FREEZE CLIP TRAINING (should save memory and computation as well)
    img_encoder.eval()

    # Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector_llm.connector.parameters(), **optim_parameters)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=math.ceil(MAX_EPOC * per_warm), num_training_steps=MAX_EPOC)

    # Record the loss at the epoch
    loss_epoch = []
    for n in range(1, MAX_EPOC + 1):
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
        for image_tensor, mask_tensor, question, answer in train_loader:
            
            try:
                # Get image features from the img encoder (on GPU 0)
                image_features, hidden_states = img_encoder(image_tensor.to(device_vit),return_hidden_states=True)
                #we want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                # Move image features to the second GPU for LLM processing
                image_features = image_features.to(device_llm)

                # Format data and "tokenize" inputs for the LLM
                answer_ = [connector_llm.tokenizer(a + "</s>", return_tensors="pt", max_length=MAX_LENGTH).input_ids for a in answer]

                for i, a in enumerate(answer_):
                    if len(a) < MAX_LENGTH:
                        answer_[i] = F.pad(a, (0, MAX_LENGTH - a.size(0)), 'constant', 0)

                answer_ = torch.cat(answer_, dim=0)[:, 1:].to(device_llm)

                # here max(len(s) for s in answer) + 2 ,ensures that there is an extra loss for not finding the eos token, while also reducing memory
                output, loss = connector_llm(image_features, question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]) + 2)
                
                accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(
                    output,
                    [connector_llm.tokenizer(a + "</s>", return_tensors="pt").input_ids[:, 1:].flatten() for a in answer],
                    tokenizer=connector_llm.tokenizer,
                    max_length=MAX_LENGTH_LLM
                )


                loss.backward()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print('Skipping batch due to OOM')
                else:
                    print(e)
                continue
                    


             # Perform the optimizer step after accumulating the gradients for `accumulation_steps` batches
            
            if (count_t + 1) % accumulation_steps == 0:
                print("Optim.step()1")
                optim.step()
                optim.zero_grad()

            connector_llm.connector.zero_grad()

            trainng_loss_avg += loss.to('cpu')

            train_accuracy_avg += accuracy
            train_precision_avg += precision
            train_recall_avg += recall
            train_f1_avg += f1
            train_bleu_score_avg += bleu_score

            count_t = count_t + 1
            count_q += answer_.size(0)

            # Ensure to perform a step if we have leftover gradients
        if (count_t + 1) % accumulation_steps != 0:
            print("Optim.step()2")
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
        connector_llm.connector.eval()

        with torch.no_grad():
            for image_tensor, mask_tensor, question, answer in validate_loader:

                try:
                     # Get image features from the img encoder (on GPU 0)
                        image_features, hidden_states = img_encoder(image_tensor.to(device_vit),return_hidden_states=True)

                        #we want the hidden state at the specified layer (len(hidden_states) - 1) is the last layer, so 0 is 0 from the end, 1 one from the end
                        image_features = hidden_states[(len(hidden_states) - 1) - hidden_layer_from_end]

                        # Move image features to the second GPU for LLM processing
                        image_features = image_features.to(device_llm)

                        # Format data and "tokenize" inputs for the LLM
                        answer_ = [connector_llm.tokenizer(a + "</s>", return_tensors="pt", max_length=MAX_LENGTH).input_ids for a in answer]

                        for i, a in enumerate(answer_):
                            if len(a) < MAX_LENGTH:
                                answer_[i] = F.pad(a, (0, MAX_LENGTH - a.size(0)), 'constant', 0)

                        answer_ = torch.cat(answer_, dim=0)[:, 1:].to(device_llm)

                        # here max(len(s) for s in answer) + 2 ,ensures that there is an extra loss for not finding the eos token, while also reducing memory
                        output, loss = connector_llm(image_features, question, answer_, max([len(connector_llm.tokenizer(s).input_ids) for s in answer]) + 2)
                        
                        accuracy, bleu_score, precision, recall, f1 = calc_loss_and_metrics(
                            output,
                            [connector_llm.tokenizer(a + "</s>", return_tensors="pt").input_ids[:, 1:].flatten() for a in answer],
                            tokenizer=connector_llm.tokenizer,
                            max_length=MAX_LENGTH_LLM
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print('Skipping batch due to OOM')
                    else:
                        print(e)
                    continue
                # dont worry about the metrics here the values should be the same as +=  0.0, also the count only increases after the continue so the loss avg is fine too

                validation_loss_avg += loss.to('cpu')

                val_accuracy_avg += accuracy
                val_precision_avg += precision
                val_recall_avg += recall
                val_f1_avg += f1
                val_bleu_score_avg += bleu_score
                count = count + 1

        # SAVE RESULTS
        if save:
            if not os.path.exists(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION))):
                os.makedirs(os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION)))
            torch.save(connector_llm.connector.state_dict(), os.path.join("/nobackup", "sc20gwb", "Models", "SavedModels", "C_V_" + str(VERSION), "connector_LLM_model" + str(n) + ".pth"))

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

    return loss_epoch

#/nobackup/sc20gwb/Models/Models_to_upload
path1 = os.path.join("/nobackup","sc20gwb","Models", "Models_to_upload", "clip_model_30.pth")
#path1 = os.path.join(os.getcwd(), "Models_to_upload","v_2000", "clip_model_30.pth")
clip_parameters  =  {
"transformer_width":512,
"transformer_layers":12,
"transformer_heads":8,
"embed_dim":512,
"vision_width":768,
"image_resolution":224,
"vision_patch_size":56,
"vision_layers":12,
"clip_model_path": path1

}



LR_LIST = [0.1,0.5]#[0.01,0.05]#[0.001,0.0001, 0.00001]
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
"embed_dim": 768, # this is the width of the CLIP ViT
"connector_layers":2
}

# Additional parameters from the function call
additional_parameters = {
    "per_warm": 0.2,
    "image_size": 224,
    "batch_size": 1,
    "vir_batch_size": 16,
    "rand_seed": 42,
    "MAX_EPOC": 3,
    "MAX_LENGTH": 256,
    "VERSION": 2000,
    "MAX_LENGTH_LLM": 48,
    "save": True,
    "cpu_only": False,
    "hidden_layer_from_end": 1
}


for i, para in enumerate(optim_list):
    p = {**connector_llm_parameters,**para,**clip_parameters,**additional_parameters}
    wandb.init(
        project="MSc_fine_tuning_step_1",
        config=p,
        resume=False  # Ensure it starts a new run
    )
    feature_aliginment_training_step_1_GPU_SPLIT(
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
        MAX_LENGTH_LLM=p['MAX_LENGTH_LLM'],
        save=p['save'],
        cpu_only=p['cpu_only']
    )

    wandb.finish()


