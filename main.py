import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample
import torch


BATCHSIZE  = 32

RANDSEED  = 42

IMAGESIZE = 240

# Example usage
test_loader, train_loader, validate_loader = load_data(transforms.Compose([
    transforms.Resize((IMAGESIZE, IMAGESIZE)),
    transforms.ToTensor()
]), BATCHSIZE, RANDSEED, os.path.join(os.getcwd(), 'Slake1.0')
)

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # Get the number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"CUDA is available with {gpu_count} GPU(s)!")
    # Print the name of each GPU available
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Training will proceed on CPU.")


for images, masks, questions, answers in train_loader:
    # Visualize the first image in the batch
    display_sample(images[0],  masks[0], questions[0], answers[0],os.path.join(os.getcwd(), 'outputs', 'sample_plot.png') )
    break

