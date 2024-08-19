import torch.utils
import torchvision.transforms as transforms
import os
from data_loading import load_combined_text_data
import torch

from CLIP import CLIP

from transformers import AutoTokenizer

import torch.nn.functional as F

import csv

import wandb

#V_1 uses torch.optim.AdamW(clip.parameters(), lr=1e-4, weight_decay=1e-4, eps=1.0e-08)

# TODO: fix tokeniser, i.e make sure that we use one that has medical vocab

def train(BATCHSIZE = 16, RANDSEED  = 42, MAX_LENGTH = 256, IMAGESIZE = 224, MAX_EPOC = 100, VERSION = 2, transformer_width=512, transformer_layers=12,transformer_heads=8,embed_dim=512,vision_width=768, image_resolution=224, vision_patch_size=8, vision_layers=12,lr=1e-4, weight_decay=1e-4, eps=1.0e-08,T_0=10, T_mult=2,save=False):
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
    test_loader, train_loader, validate_loader = load_combined_text_data(transforms.Compose([
        transforms.Resize((IMAGESIZE, IMAGESIZE)),
        transforms.ToTensor()
    ]), BATCHSIZE, RANDSEED, os.path.join(os.getcwd(), 'Slake1.0')
    )

    model_path =  os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True)

    # LOAD CLIP model
    clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=transformer_width,context_length=MAX_LENGTH,transformer_layers=transformer_layers,transformer_heads=transformer_heads, embed_dim=embed_dim, vision_width=vision_width, image_resolution=image_resolution, vision_patch_size=vision_patch_size, vision_layers=vision_layers,device=device)

    # Training
    clip.to(device)
    clip.train()
    clip.zero_grad()

    #Optimizer and learning rate scheduling
    optim = torch.optim.AdamW(clip.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=T_0, T_mult=T_mult)

    # Record the loss at the epoch
    loss_epoch = []
    for n in range(1,MAX_EPOC + 1):
        clip.train()
        trainng_loss_avg  = torch.tensor([0.0])
        count_t = 0
        for image_tensor, mask_tensor, text in train_loader:

            optim.zero_grad()
            
            text_tensor = torch.cat([tokenizer(a + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a in text],0).to(device)
            
            image_tensor = image_tensor.to(device)

            image_features,text_features = clip(image_tensor,text_tensor)

            #CALC LOSS FROM FEATURES
            # normalised features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = clip.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

            labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss  = F.cross_entropy(logits_per_text, labels)

            loss = (image_loss + text_loss) / 2
            loss.backward()

            optim.step()

            # logit scaling set as max 100 as mentioned in CLIP paper # log(100) = 4.6052
            clip.logit_scale.data = torch.clamp(clip.logit_scale.data, 0, 4.6052)

            clip.zero_grad()

            trainng_loss_avg += loss.to('cpu')

            count_t +=1

        scheduler.step()
        #  VALIDTATE
        loss_avg  = torch.tensor([0.0])
        count=0
        clip.eval()
        with torch.no_grad():
            for image_tensor, mask_tensor, text in validate_loader:
                text_tensor = torch.cat([tokenizer(a + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a in text],0).to(device)
                image_tensor = image_tensor.to(device)

                image_features,text_features = clip(image_tensor,text_tensor)

                #CALC LOSS FROM FEATURES

                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = clip.logit_scale.exp()

                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ image_features.t()

                labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)

                image_loss = F.cross_entropy(logits_per_image, labels)
                text_loss  = F.cross_entropy(logits_per_text, labels)

                loss = (image_loss + text_loss) / 2
                loss_avg += loss.to('cpu')
                count  += 1
            

            #print(loss_avg.to('cpu').detach().numpy()[0]/count,trainng_loss_avg.to('cpu').detach().numpy()[0]/count_t)

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



def saveResults(VERSION,loss_epoch_):

    if not os.path.exists(os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION))):
                    os.makedirs(os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION)))
    
    filename = os.path.join(os.getcwd(),"SavedModels", "V_" + str(VERSION),"epoch_loss.csv")

    # Writing to the CSV file
    with open(filename, mode='w', newline='\n') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Epoch', 'Validation Loss','Training Loss'])
        
        # Write the data
        writer.writerows(loss_epoch_)

    print(f"CSV file '{filename}' created successfully.")



# Configure parameters

# Lists for each parameter
BATCHSIZE_LIST = [8,16]
MAX_EPOC_LIST = [30]
VERSION_LIST = [8]
LR_LIST = [1e-4,5e-5,8e-5]
WEIGHT_DECAY_LIST = [1e-4,1e-3]
EPS_LIST = [1.0e-08]
T_0_LIST = [10]
T_MULT_LIST = [2]

# Generate the list of dictionaries with all combinations
config_list = [
    {
        'BATCHSIZE': batchsize,
        'MAX_EPOC': max_epoc,
        'VERSION': version,
        'lr': lr,
        'weight_decay': weight_decay,
        'eps': eps,
        'T_0': t_0,
        'T_mult': t_mult
    }
    for batchsize in BATCHSIZE_LIST
    for max_epoc in MAX_EPOC_LIST
    for version in VERSION_LIST
    for lr in LR_LIST
    for weight_decay in WEIGHT_DECAY_LIST
    for eps in EPS_LIST
    for t_0 in T_0_LIST
    for t_mult in T_MULT_LIST
]

for i, p in enumerate(config_list):

    wandb.init(
        # set the wandb project where this run will be logged
        project="MSc",
        # track hyperparameters and run metadata
        config=p
    )

    loss_epoch  = train(**p)
    # save local results
    # Specify the CSV file name

    wandb.finish()

    saveResults(10000 + i,loss_epoch)

