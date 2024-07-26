import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
import torch
import transformers

import torch.nn as nn

import numpy as np

from CLIP import CLIP, VisionTransformer, convert_weights

from transformers import LlamaForCausalLM, AutoTokenizer

import torch.nn.functional as F


BATCHSIZE  = 4

RANDSEED  = 42

MAX_LENGTH =  76

IMAGESIZE = 224


# CHECK GPU SUPPORT AND ASSIGN DEVICE
if torch.cuda.is_available():
    # Get the number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s)!")

    # Print the name of each GPU available
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    #TODO: We will need to change this so that we can use multipple GPUs
    device = torch.device("cuda")
else:
    print("CUDA is not available. Training will proceed on CPU.")
    device = torch.device("cpu")

#LOAD DATA
test_loader, train_loader, validate_loader = load_data(transforms.Compose([
    transforms.Resize((IMAGESIZE, IMAGESIZE)),
    transforms.ToTensor()
]), BATCHSIZE, RANDSEED, os.path.join(os.getcwd(), 'Slake1.0')
)

# Tokenizer, this is temp prehaps. We need to work out how to create our own from medical data, or use one created frommedical data.  Should be BPE to conform to  clip. When we do we need to make sure that EOS is the largest token
model_path =  os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True)

print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)


# LOAD CLIP model

clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=512,context_length=76,transformer_layers=12,transformer_heads=8, embed_dim=512, vision_width=768, image_resolution=224, vision_patch_size=8, vision_layers=12,device=device)

#reduce the size of the weights to fp16 where possible
convert_weights(clip)

# Training
clip.to(device)

# loss
# Create the loss function object
criterion = nn.CrossEntropyLoss()

# Optimizer TODO: We use the Adam optimizer (Kingma & Ba, 2014) with decoupled weight decay regularization (Loshchilov & Hutter, 2017) applied to all weights that are not gains or biases, and decay the learning rate using a cosine schedule (Loshchilov & Hutter, 2016).
optim  = torch.optim.Adam(clip.parameters())

sim_mat = torch.tensor([0])

for n in range(30):

    for image_tensor, mask_tensor, question, answer in train_loader:
        
        text = torch.cat([tokenizer(a +  " " + b + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a, b in zip(question,answer)],0).to(device)

        image_tensor = image_tensor.to(device)

        array = np.arange(image_tensor.size(0))

        np.random.shuffle(array)

        sim_mat = clip(image_tensor,text)

        labels = torch.eye(image_tensor.size(0), dtype=torch.half, device=device)
        
        sim_mat_i = torch.softmax(sim_mat,dim=0)
        sim_mat_t = torch.softmax(sim_mat, dim=1)

        #Softmax i.e predicting image from text,or predicting text from image. These are now prob distributions 

        loss_i = F.binary_cross_entropy(sim_mat_i, labels)
        loss_t = F.binary_cross_entropy(sim_mat_t, labels)

        #  Simertric cross entropy

        loss = (loss_i + loss_t)/2

        optim.zero_grad()

        loss.backward()

        # TODO: how do we combine the text and answers

        # TODO: fix tokeniser, i.e make sure that we use one that has medical vocab

    #  VALIDTATE
    loss_avg  = 0.0
    loss_i_avg = 0.0
    loss_t_avg = 0.0
    clip.eval()
    count=0

    print("Validating at epoc ", n,":")
    for image_tensor, mask_tensor, question, answer in validate_loader:

        text = torch.cat([tokenizer(a +  " " + b + "</s>",return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a, b in zip(question,answer)],0).to(device)

        image_tensor = image_tensor.to(device)

        array = np.arange(image_tensor.size(0))

        np.random.shuffle(array)

        print("Model")
        sim_mat = clip(image_tensor,text)

        print("aftermodel")

        labels = torch.eye(image_tensor.size(0), dtype=torch.half, device=device)
        
        sim_mat_i = torch.softmax(sim_mat,dim=0)
        sim_mat_t = torch.softmax(sim_mat, dim=1)

        #Softmax i.e predicting image from text,or predicting text from image. These are now prob distributions 
        print("BCE")
        loss_i = F.binary_cross_entropy(sim_mat_i, labels)
        loss_t = F.binary_cross_entropy(sim_mat_t, labels)

        loss_i_avg += loss_i
        loss_t_avg +=  loss_t

        #  Simertric cross entropy
        loss = (loss_i + loss_t)/2
        loss_avg += loss
        count  += 1
        

    print(loss_avg.to('cpu').detach().numpy()/count,loss_i_avg.to('cpu').detach().numpy()/count,loss_t_avg.to('cpu').detach().numpy()/count)
    clip.train()






def load_vicuna(model_dir):
    # LOADING VICUNA
    from transformers import LlamaForCausalLM, AutoTokenizer

    model_path =  os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")

    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True)
    print("Tokenizer loaded successfully")

    string = "Hello?"
    print(tokenizer(string))

    model = LlamaForCausalLM.from_pretrained(os.path.join(model_path, "model")).to(device)
    print("Victuna Model loaded successfully")

    # Example of running the model on the correct device
    inputs = tokenizer(string, return_tensors="pt").to(device)

    # The forward methods does not deal with the pre and post processing steps
    generated_ids = model.generate(inputs.input_ids)

    print(generated_ids)
    decoded = tokenizer.batch_decode(generated_ids,clean_up_tokenization_spaces=False)

    print(decoded[0])