
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from torch.optim.lr_scheduler import LambdaLR

class CustomSchedulerWithWarmup(LambdaLR):
    """Custom learning rate scheduler with warmup period.
    
    Implements different scheduling strategies for training phases 1 and 2:
    - Phase 1: Choice between linear or cosine warmup for connector training
    - Phase 2: Cosine warmup for full model training with LoRA
    
    Args:
        optimizer: The optimizer to schedule
        num_warmup_steps (int): Number of warmup steps
        num_training_steps (int): Total number of training steps
        training_step (int): Current training phase (1 or 2)
        schedule_type (str): "linear" or "cosine" for phase 1
        
    Returns:
        LambdaLR scheduler with appropriate learning rate adjustment
    """
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, training_step, schedule_type="cosine"):
        self.training_step = training_step
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.schedule_type = schedule_type

        if self.training_step == 1:
            # Choose between linear or cosine schedule for phase 1
            if self.schedule_type == "linear":
                self.phase1_scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps, num_training_steps
                )
            elif self.schedule_type == "cosine":
                self.phase1_scheduler = get_cosine_schedule_with_warmup(
                    optimizer, num_warmup_steps, num_training_steps
                )
            else:
                raise ValueError("schedule_type must be 'linear' or 'cosine'")
        else:
            # Default to cosine schedule when training both projector and LLM (phase 2)
            self.phase2_scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps, num_training_steps
            )

        def lr_lambda(current_step):
            if self.training_step == 1:
                return self.phase1_scheduler.lr_lambdas[0](current_step)
            else:
                return self.phase2_scheduler.lr_lambdas[0](current_step)

        super().__init__(optimizer, lr_lambda)