import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
import torch

import numpy as np

from CLIP import CLIP, convert_weights

from transformers import LlamaForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import torch.nn.functional as F

import csv

#Load Loss

# Train
    # ViT
    # projection matrix
    # Vicuna LLM

#validate and record data

#  test data on chosen epoc according to the validation data








#Return the vicuna model and its tokenizer
def load_vicuna(model_path,device):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True)

    model = LlamaForCausalLM.from_pretrained(os.path.join(model_path, "model")).to(device)

    return model, tokenizer

#Return  the object that is the visual encoder, return the heat scaling parameter as well
def load_ViT_img_encoder(tokenizer,transformer_width,MAX_LENGTH,transformer_layers,transformer_heads,embed_dim,vision_width,image_resolution,vision_patch_size,vision_layers,device,clip_model_path):
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=MAX_LENGTH,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)
    clip.load_state_dict(torch.load(clip_model_path))

    return clip.visual, clip.logit_scale


# This trains the MLP between the visual encoder and LLM. It can be seen as traing a compatible visual tokenizer for the for the frosen LLM
def feature_aliginment_training(clip_parameters,optim_parameters,per_warm,image_size,batch_size,rand_seed,connector_width,MAX_EPOC,MAX_LENGTH,vicuna_path=os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5"),save=False):
    # CHECK GPU SUPPORT AND ASSIGN DEVICE
    if torch.cuda.is_available():
        # Get the number of GPUs available
        gpu_count = torch.cuda.device_count()
        print(f"CUDA is available with {gpu_count} GPU(s)!")

        # Print the name of each GPU available
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        device = torch.device("cuda")
    else:
        print("CUDA is not available. Training will proceed on CPU.")
        device = torch.device("cpu")

    #LOAD DATA
    test_loader, train_loader, validate_loader = load_data(transforms.Compose([
                                                           transforms.Resize((image_size, image_size)),
                                                           transforms.ToTensor()
                                                                              ]), batch_size, rand_seed, os.path.join(os.getcwd(), 'Slake1.0')
    )

    #LOAD vicuna
    vicuna, tokenizer = load_vicuna(vicuna_path,device)

    # FREEZE vicuna TRANING (should save  memory and computation as well)
    vicuna.eval()

    # LOAD ViT encoder from the CLIP model
    img_encoder,logit_scale = load_ViT_img_encoder(**clip_parameters,device=device)

    # FREEZE CLIP TRANING (should save  memory and computation as well)
    img_encoder.eval()

    #MAKE MLP vision-language connector
    connector = torch.nn.Sequential([torch.nn.Linear(clip_parameters['embed_dim'],connector_width),torch.nn.Linear(connector_width,clip_parameters['embed_dim'])])

    #Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(connector.parameters(), **optim_parameters)
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=MAX_EPOC // (100 * per_warm),num_training_steps=MAX_EPOC)

    # Record the loss at the epoch
    loss_epoch = []
    for n in range(1,MAX_EPOC + 1):
        connector.train()
        trainng_loss_avg  = torch.tensor([0.0])
        count_t = 0
        for image_tensor, mask_tensor, question, answer in train_loader:

            optim.zero_grad()
            
            #Get image features from the img encoder
            image_features = img_encoder(image_tensor.to(device))
            
            #Format data and "tokenize" inputs for the LLM, combine them in the form <s> image_encoder_tokenized question </s>
            question = torch.cat([tokenizer(a + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a in question],0)[:, 1:, :].to(device)
            answer = torch.cat([tokenizer(a + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a in answer],0).to(device)
            image_embedding_tokenized = connector(image_features)
            start_token = torch.tensor([tokenizer.all_special_tokens[tokenizer.all_special_ids.index("<s>")]])
            prompt = torch.cat((start_token,image_embedding_tokenized,question),dim=1)

            #Pass the prompt to vicuna
            output = vicuna.generate(prompt)

            #TODO: Ensure that the answer is set to  the length of the tokenizer. where the values are the same make them 0 where they are diffrent make them 1?
            #TODO:calculate the loss

            loss.backward()

            optim.step()

            connector.zero_grad()

            trainng_loss_avg += loss.to('cpu')

            count_t +=1

        scheduler.step()

        # TODO: VALIDATE,MAKE SURE THAT WE INCLUDE THE  ACCURACY, PREC,RECALL, SPECITIVITY ect AS WELL
        #  VALIDTATE
        loss_avg  = torch.tensor([0.0])
        count=0
        clip.eval()
        with torch.no_grad():
            for image_tensor, mask_tensor, text in validate_loader:
              
            # TODO:  SAVE THE CONNECTOR MODEL AND THE METRICS CALCULATED
            #SAVE RESULTS
            if save:
                if not os.path.exists(os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION))):
                    os.makedirs(os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION)))
                torch.save(clip,os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION),"clip_model_" + str(n) + ".pth"))
            
            loss_epoch.append([
                n,
                loss_avg.to('cpu').detach().numpy()[0]/count,
                trainng_loss_avg.to('cpu').detach().numpy()[0]/count_t,
                BATCHSIZE,
                RANDSEED,
                MAX_LENGTH,
                IMAGESIZE,
                MAX_EPOC,
                VERSION,
                transformer_width,
                transformer_layers,
                transformer_heads,
                embed_dim,
                vision_width,
                image_resolution,
                vision_patch_size,
                vision_layers,
                lr,
                weight_decay,
                eps,
                T_0,
                T_mult
                ])
            wandb.log({"loss_validate":loss_avg.to('cpu').detach().numpy()[0]/count, "loss_training":trainng_loss_avg.to('cpu').detach().numpy()[0]/count_t})


    return loss_epoch


