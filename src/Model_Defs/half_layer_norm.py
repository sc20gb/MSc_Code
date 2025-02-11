import torch


class CustomLayerNorm(torch.nn.LayerNorm):
    """Custom LayerNorm that handles half-precision inputs.
    
    Converts input to float32 for computation then back to original dtype.
    Used to avoid numerical instability with half-precision training.
    """
    def forward(self, input):
        # Convert input to float32 for LayerNorm calculation, then cast back to the original dtype
        return super().forward(input.float()).to(input.dtype)