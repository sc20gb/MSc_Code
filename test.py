
import torch

import torch.nn.functional as F


import math

def compute_normalized_loss(logits_per_image, labels):

        
    # Compute the cross-entropy loss with mean reduction
    loss = F.cross_entropy(logits_per_image, labels, reduction='mean')
    num_classes = logits_per_image.size(1)  # Number of classes
    return loss / torch.log(torch.tensor(num_classes, dtype=torch.float))


print(compute_normalized_loss(torch.tensor([[100.0, 0.0, 0.0],
                                            [5.0, 10.0, 5.0],
                                            [0.0, 0.0, 100.0]])
                                            
                                            ,torch.arange(3)))



print(compute_normalized_loss(torch.tensor([[100.0, 0.0], [5.0, 10.0]]), labels=torch.arange(2)))