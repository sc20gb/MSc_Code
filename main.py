import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
#from TextTransformerEncoder import TextTransformerEncoder
import torch
import transformers

import torch.nn as nn

# import warnings
# warnings.filterwarnings("ignore")
# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# #TODO: deal with this issue


BATCHSIZE  = 32

RANDSEED  = 42

MAX_LENGTH =  76

IMAGESIZE = 240


# CHECK GPU SUPPORT AND ASSIGN DEVICE
if torch.cuda.is_available():
    # Get the number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s)!")

    # Print the name of each GPU available
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    #TODO: We will need to change this so that we can use multipple GPUs
    device = torch.device("cpu")
else:
    print("CUDA is not available. Training will proceed on CPU.")
    device = torch.device("cpu")

#LOAD DATA
test_loader, train_loader, validate_loader = load_data(transforms.Compose([
    transforms.Resize((IMAGESIZE, IMAGESIZE)),
    transforms.ToTensor()
]), BATCHSIZE, RANDSEED, os.path.join(os.getcwd(), 'Slake1.0')
)

# CONTRASTIVE LANGUAGE-IMAGE PRE-TRAINING

# Load the ViT
from clip_ViT import ClipEncoder
imageEncoder = ClipEncoder(d_model=512,n_heads=8,r_mlp=4, img_size=IMAGESIZE, patch_size=30, n_channels=3,max_seq_length=256)

# Load GPT2 Model (text-encoder)
from transformers import GPT2Tokenizer,  GPT2Model, GPT2Config, GPT2LMHeadModel

# Change the config to that described by the CLIP paper
config = GPT2Config.from_json_file(os.path.join(os.getcwd(), "gpt2Config","config.json"))

# The Clip paper does not create a new tokenizer but uses one with a 49,152 vocab size
textEncoderTokeniser = GPT2Tokenizer.from_pretrained('gpt2')
textEncoderTokeniser.pad_token = textEncoderTokeniser.eos_token

#Load the head as well
LMHead = GPT2LMHeadModel(GPT2Config.from_json_file(os.path.join(os.getcwd(), "gpt2Config","config.json")))

# Initalise an untrained model 
textEncoder = GPT2Model(config)

# Training

for image_tensor, mask_tensor, question, answer in train_loader:

    #Encode Image
    imgEncodings = imageEncoder(image_tensor)
    print("Image Encoded to ", imgEncodings.size())

    #Tokenize question and answer
    text = [a + " " + b for a, b in zip(question,answer)]

    tokens = textEncoderTokeniser(text, padding=True, truncation=True, return_tensors='pt', max_length=MAX_LENGTH)

    print(type(tokens))

    #Pass to the textEncoder,  Includes the mask
    #EncoderOut = textEncoder(**tokens)
    DecoderOut = LMHead(**tokens)

    print(DecoderOut.logits.size())
    break

    #  TODO:  Work out how to Train the two
        # Could be that we train the llm on the question and answer pairs?
    # TODO: Make sure that the  GPT2 model is the  right model, config and all
    # TODO: Make sure taht the tokenizer is correct










# Take the two embeddings and preject them linearly to a new joint embedding space



# for m in textEncoder.modules():
#     if not isinstance(m, nn.Sequential):
#         print(m)


textEncoder = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')

# load empty tokenizer for gpt2 model
tokenizer = torch.hub.load('huggingface/transformers', 'tokenizer', 'gpt2')


string = "Hello?"





#CREATE/LOAD CORPUS
corpus_filename = os.path.join(os.getcwd(), 'corpus', 'corpus.txt')

# Check if corpus file exists
if os.path.exists(corpus_filename):
    # If corpus file exists, load it
    with open(corpus_filename, 'r') as file:
        corpus = file.read()
else:
    # If corpus file doesn't exist, create it
    corpus = ""
    for images, masks, questions, answers in train_loader:
        for q in questions:
            corpus += "<SOS> " + q + " <EOS> "
        for a in answers:
            corpus += "<SOS> " + a + " <EOS> "

    # Save the corpus to file
    with open(corpus_filename, 'w') as file:
        file.write(corpus)

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
