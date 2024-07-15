import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
#from TextTransformerEncoder import TextTransformerEncoder
from vicuna_llm import CustomLlamaModel
import torch

# import warnings
# warnings.filterwarnings("ignore")
# import torchtext
# torchtext.disable_torchtext_deprecation_warning()
# #TODO: deal with this issue


BATCHSIZE  = 32

RANDSEED  = 42

IMAGESIZE = 240


# Check GPU support
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


# #CREATE TEXT ENCODER
# TextEncoder = TextTransformerEncoder(corpus)


# Loading Vicuna

#
from transformers import LlamaModel, LlamaConfig

from transformers import LlamaForCausalLM, LlamaTokenizer

model_path =  os.path.join(os.getcwd(), "Models", "vicuna-7b-v1.5")

print("Before")

try:
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(model_path, "tokenizer"))
    print("Tokenizer loaded successfully")

    string = "Hello"
    print(tokenizer(string))
    
    model = LlamaForCausalLM.from_pretrained(os.path.join(model_path, "model")).to(device)
    print("Model loaded successfully")

    # Example of running the model on the correct device
    inputs = tokenizer(string, return_tensors="pt").to(device)
    outputs = model(**inputs)
    print("Model executed successfully on the correct device, output was:",outputs.size())
    
except Exception as e:
    print(f"An error occurred: {e}")


#

