import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
from TextTransformerEncoder import TextTransformerEncoder
from vicuna_llm import CustomLlamaModel
import torch

import warnings
warnings.filterwarnings("ignore")
import torchtext
torchtext.disable_torchtext_deprecation_warning()
#TODO: deal with this issue


BATCHSIZE  = 32

RANDSEED  = 42

IMAGESIZE = 240


# CHECK GPU SUPPORT
if torch.cuda.is_available():
    # Get the number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s)!")
    # Print the name of each GPU available
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Training will proceed on CPU.")


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


#CREATE TEXT ENCODER
TextEncoder = TextTransformerEncoder(corpus)


for images, masks, questions, answers in train_loader:
    print(TextEncoder(questions,["<SOS>"  for i in answers]).size())
    break









# Loading Vicuna

# #
# from llama_cpp import Llama

# path = os.path.join(os.getcwd(),"LLMModels","stable-vicuna-13B.ggmlv3.q8_0.bin")

# try:
#     # Load the model
#     model = Llama(model_path=path)
#     print("Model loaded successfully")
# except Exception as e:
#     print(f"Failed to load model: {e}")

# print(path)

# # Load the model
# model = CustomLlamaModel(modelpath=path)


# try:
#     # Define the input prompt
#     input_prompt = "Once upon a time there was a bob, "

#     # Generate the output
#     output = model(input_prompt, 50)

    

#     # Print the generated text
#     print(output["choices"][0]["text"])
# finally:
#     # Explicitly clean up the model, if not an exception occurs on windows.
#     del model


# #

