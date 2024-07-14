import math
import torch.nn as nn
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform



# The Transformer model presented by:
# Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L. and Polosukhin, I. 2023. Attention Is All You Need.
# TODO: Changes were made by  Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., and Sutskever, I. Language models are unsupervised multitask learners. 2019.
# TODO: Add a designated padding token
# TODO: Add a mask
# TODO: the activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text

import math
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super().__init__()
        self.model_dim = model_dim
        self.max_len = max_len
        self.positional_encoding = self.create_positional_encoding(max_len, model_dim)

    def forward(self, x):
        batch_len  = x.size(0)
        seq_len = x.size(1)

        temp = self.positional_encoding[:seq_len, :]

        temp = temp.repeat(batch_len,1,1)

        return x + self.positional_encoding[:seq_len, :]

    def create_positional_encoding(self, max_len, model_dim):
        positional_encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        #TODO:Check this
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return positional_encoding

class TextTransformerEncoder(nn.Module):
    def __init__(self, corpus, model_dim=512, num_heads=8, num_layers=6, dropout=0.1, merges=4, maxlength=76):
        super().__init__()
        self.model_dim = model_dim
        self.positional_encoding = PositionalEncoding(model_dim)
        self.init_encoder(corpus, merges)
        self.embedding = nn.Embedding(len(self.vocab), model_dim)
        self.transformer = nn.Transformer(d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout)
        self.max_length = maxlength
        self.linear = nn.Linear(model_dim * maxlength, 512)

    
    def embed(self,inputs):
        token_ids_list =  []
        for text in inputs:
            #Tokenize the inputs
            tokens = self.tokenizer(text)
            token_ids = [self.vocab[token] for token in tokens]
            token_ids_list.append(token_ids)

        # Pad sequences to ensure equal length in a batch (if necessary)
        max_len = self.max_length
        token_ids_padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids_list]
        token_tensor = torch.tensor(token_ids_padded)
        
        input_embeddings = self.embedding(token_tensor) * math.sqrt(self.model_dim)
        input_embeddings = self.positional_encoding(input_embeddings)

        return input_embeddings

    def forward(self, inputs, outputs):

        #embed inputs
        input_embeddings = self.embed(inputs)

        #embed outputs
        output_embeddings = self.embed(outputs)

        # Now we need to create the transformer
        transformer_out = self.transformer(input_embeddings,output_embeddings)


        return transformer_out

    def init_encoder(self, corpus, merges):
        self.tokenizer = get_tokenizer('basic_english')
        tokens = self.tokenizer(corpus)

        #TODO: Set up pair byte encoding here

        # Create a vocabulary from the tokens
        self.vocab = build_vocab_from_iterator([tokens], specials=["<unk>",  "<EOS>", "<SOS>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
