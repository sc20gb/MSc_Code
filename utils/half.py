
import torch.nn as nn
import torch

class CustomLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype

        x = x.to(torch.float32)
        ret = super().forward(x)

        ret = ret.to(orig_type)
        
        return ret

#if the model needs to use float16 then layernorms need to be handled
def handle_half_for_layer_Norm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
                parent = model
                name_parts = name.split('.')
                for part in name_parts[:-1]:  # Traverse to the parent module
                    parent = getattr(parent, part)
                
                # Replace with CustomLayerNorm
                setattr(parent, name_parts[-1], CustomLayerNorm(module.normalized_shape))
