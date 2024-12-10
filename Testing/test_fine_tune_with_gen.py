import unittest
import torch
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from fine_tune_with_gen import (
    handle_devices,
    feature_aliginment_training,
    CustomSchedulerWithWarmup
)
import os
import matplotlib.pyplot as plt

class TestFineTuneWithGen(unittest.TestCase):

    def test_handle_devices(self):
        device_vit, device_llm = handle_devices(cpu_only=True)
        self.assertEqual(device_vit.type, 'cpu')
        self.assertEqual(device_llm.type, 'cpu')
        
    def test_scheduler_behavior(self):
        # Mock parameters
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))], lr=0.1)
        num_warmup_steps = 10
        num_training_steps = 100
        training_step = 2  # Assume phase 2 for cosine decay after warmup

        # Initialize the custom scheduler
        scheduler = CustomSchedulerWithWarmup(optimizer, num_warmup_steps, num_training_steps, training_step)

        # Collect learning rates
        lrs = []
        for step in range(num_training_steps):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        # Check linear increase during warmup
        for step in range(num_warmup_steps):
            expected_lr = (step + 1) / num_warmup_steps * 0.1  # Linear increase
            if not self.assertAlmostEqual(lrs[step], expected_lr, places=5):
                self.plot_lr(lrs, num_warmup_steps, num_training_steps)
                self.fail(f"Learning rate did not match expected linear increase at step {step}")

        # Check cosine decay after warmup
        cosine_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        for step in range(num_warmup_steps, num_training_steps):
            expected_lr = cosine_scheduler.get_lr()[0]
            if not self.assertAlmostEqual(lrs[step], expected_lr, places=5):
                self.plot_lr(lrs, num_warmup_steps, num_training_steps)
                self.fail(f"Learning rate did not match expected cosine decay at step {step}")

    def plot_lr(self, lrs, num_warmup_steps, num_training_steps):
        if not os.path.exists('Errors'):
            os.makedirs('Errors')
        plt.figure()
        plt.plot(range(num_training_steps), lrs, label='Learning Rate')
        plt.axvline(x=num_warmup_steps, color='r', linestyle='--', label='Warmup End')
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.savefig('Errors/lr_schedule.png')
        plt.close()

if __name__ == '__main__':
    unittest.main()