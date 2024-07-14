
from llama_cpp import Llama
import torch.nn as nn

class CustomLlamaModel(nn.Module):
    #https://huggingface.co/lmsys/vicuna-13b-v1.5/tree/main
    def __init__(self, modelpath):
        super(CustomLlamaModel, self).__init__()
        self.llama = Llama(model_path=modelpath)
    
    def forward(self, input_prompt,  max):
        output = self.llama(input_prompt, max_tokens=max)
        return output