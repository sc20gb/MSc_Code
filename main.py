import torchvision.transforms as transforms
import os
from data_loading import load_data,  display_sample

# Example usage
test_loader, train_loader, validate_loader = load_data(transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor()
]), 32, 42, os.path.join(os.getcwd(), 'Slake1.0')
)

for images, masks, questions, answers in train_loader:
    # Visualize the first image in the batch
    display_sample(images[0],  masks[0], questions[0], answers[0],os.path.join(os.getcwd(), 'outputs', 'sample_plot.png') )
    break

