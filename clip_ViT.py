import torch.nn as nn
import torch
import numpy as np

# This class was found https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model)
    
    return x
  
# This class was found https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()

    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

    # Creating positional encoding
    pe = torch.zeros(max_seq_length, d_model)

    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    # Expand to have class token for every image in batch,we are not using classification.
    # CLIP as in  https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
    #indicates that they use a slightly  diffrent intialisation scheme, we will  not have need of classification as that is not the use of this transformer
    #TODO

    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

    # Adding class tokens to the beginning of each embedding
    x = torch.cat((tokens_batch,x), dim=1)

    # Add positional encoding to embeddings
   # This has been altered so that the max_seq_length is not addded to x but the number of patches
    x = x + self.pe[:, :x.size()[1], :].repeat(x.size(0), 1, 1)

    return x

# This class was found https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    # Obtaining Queries, Keys, and Values
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1)

    # Scaling
    attention = attention / (self.head_size ** 0.5)

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention

# This class was found https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    # Combine attention heads
    out = torch.cat([head(x) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out

# This class was found https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6
class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # Sub-Layer 1 Normalization
    self.ln1 = nn.LayerNorm(d_model)

    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads)

    # Sub-Layer 2 Normalization
    self.ln2 = nn.LayerNorm(d_model)

    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )
    #TODO: FIND r_mlp in AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE paper

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out

# This class is derived from
# 1.) the paper by open AI https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
# 2.) The paper AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
# 3.)  https://medium.com/correll-lab/building-a-vision-transformer-model-from-scratch-a3054f707cc6

# The max seq len is the predefined size of the PositionalEncoding. This should be the size of the number of patches.
# However as this number can change i will set it to 16
class ClipEncoder(nn.Module):
    def __init__(self,d_model=512,n_heads=8,r_mlp=4, img_size=240, patch_size=30, n_channels=3,max_seq_length=256):
        super(ClipEncoder, self).__init__()

        self.encoder = TransformerEncoder(d_model,n_heads,r_mlp)

        self.patchEmbedding = PatchEmbedding(d_model, img_size, patch_size, n_channels)

        # In https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf
        # An  extra normilisation layer is between the combined patch and position encodings and the transformer

        self.norm = nn.LayerNorm(d_model)

        self.posEncoding = PositionalEncoding(d_model, max_seq_length)
      

    def forward(self, x):

        x =  self.patchEmbedding(x)

        x = self.posEncoding(x)

        self.norm(x)

        x = self.encoder(x)

        return x