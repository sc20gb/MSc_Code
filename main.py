import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
#from TextTransformerEncoder import TextTransformerEncoder
import torch
import transformers

import torch.nn as nn

import numpy as np

from CLIP import CLIP, VisionTransformer, convert_weights

from transformers import LlamaForCausalLM, AutoTokenizer




# import warnings
# warnings.filterwarnings("ignore")
# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# #TODO: deal with this issue


BATCHSIZE  = 2

RANDSEED  = 42

MAX_LENGTH =  76

IMAGESIZE = 224

VOCABSIZE = 10000


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

# Tokenizer, this is temp prehaps. We need to work out how to create our own from medical data, or use one created frommedical data.  Should be BPE to conform to  clip
model_path =  os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"),do_sample=True)


# LOAD CLIP model

clip = CLIP(vocab_size=tokenizer.vocab_size, transformer_width=512,context_length=76,transformer_layers=12,transformer_heads=8, embed_dim=512, vision_width=768, image_resolution=224, vision_patch_size=8, vision_layers=12,device=device)

#reduce the size of the weights to fp16 where possible
convert_weights(clip)

# Training
clip.to(device)

# loss
# Create the loss function object
criterion = nn.CrossEntropyLoss()

for image_tensor, mask_tensor, question, answer in train_loader:

    
    
    text = torch.cat([tokenizer(a +  " " + b,return_tensors="pt",padding='max_length', max_length = MAX_LENGTH).input_ids for a, b in zip(question,answer)],0).to(device)

    image_tensor = image_tensor.to(device)

    print(image_tensor.size())

    print(text.size())
    try:
            logits_per_image, logits_per_text = clip(image_tensor,text)
    except torch.cuda.CudaError as e:
        print("CUDA error occurred:", e)
    labels = np.arange(BATCHSIZE)

    loss_i = criterion(logits_per_image, labels) 
    loss_t = criterion(logits_per_text, labels) 
    loss = (loss_i + loss_t)/2

    print(loss.to("cpu"))

    break

    # TODO: how do we combine the text and answers

    # TODO:Sort out loss

    # TODO: fix tokeniser, i.e make sure that we use one that is BPE and has medical vocab


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
