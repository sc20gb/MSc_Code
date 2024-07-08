import json
import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CustomMaskTransform(transforms.Compose):
    def __init__(self, transforms_list):
        super().__init__(transforms_list.transforms)
    
    def __call__(self, mask):
        transformed_mask = mask
        for transform in self.transforms:
            if isinstance(transform, (transforms.ColorJitter, transforms.Normalize)):
                continue  # Skip color jitter and normalization transformations
            transformed_mask = transform(transformed_mask)
        return transformed_mask
    
# Custom Dataset class
class JsonDataset(Dataset):

    def __init__(self, json_file, transform=transforms.ToTensor()):
        self.data = self.load_json(json_file)
        self.data_frame = pd.DataFrame(self.data)
        self.data_frame = self.data_frame[self.data_frame['q_lang'] == 'en']
        self.transform = transform
        dir = os.path.join(os.getcwd(),'Slake1.0', 'imgs')
        self.img_dir = os.path.normpath(dir)

    def load_image_as_tensor(self, img_path, transform):
        # Load the image
        image = Image.open(img_path).convert('RGB')
                
        # Apply the transformation
        image_tensor = transform(image)
        
        return image_tensor
    
    def string_to_padded_tensor(self, string, max_length):
        unicode_code_points = [ord(char) for char in string]
        # Pad the sequence with zeros if it's shorter than max_length
        padded_code_points = unicode_code_points + [0] * (max_length - len(unicode_code_points))
        # Truncate the sequence if it's longer than max_length
        padded_code_points = padded_code_points[:max_length]
        return torch.tensor(padded_code_points, dtype=torch.float32)
    
    def load_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Get the image path for the current index
        img_path = os.path.normpath(os.path.join(self.img_dir, *self.data_frame.iloc[idx]['img_name'].split('/')))

        # Load the image and convert it to a tensor
        image_tensor = self.load_image_as_tensor(img_path, self.transform)
        
        # Get the row from the DataFrame as a dictionary
        df_row_dict = self.data_frame.iloc[idx].to_dict()

        # word tokenization
        question_tensor = df_row_dict['question']

        answer_tensor = df_row_dict['answer']

        # Get the img
        img_directory = os.path.dirname(img_path)

        # Get the mask path for the current index
        mask_path = os.path.normpath(os.path.join(img_directory, 'mask.png'))
        mt = CustomMaskTransform(self.transform)
        
        # Load the mask and convert it to a tensor
        mask_tensor = self.load_image_as_tensor(mask_path, mt)
                
        return image_tensor, mask_tensor, question_tensor, answer_tensor


def display_sample(image_tensor, mask_tensor, question, answer, save_path=None):
    """
    Display the image, its mask, the question, and the answer using Matplotlib.
    Optionally save the plot as an image instead of displaying it.

    Args:
    - image_tensor (torch.Tensor): The image tensor.
    - mask_tensor (torch.Tensor): The mask tensor.
    - question (str): The question associated with the image.
    - answer (str): The answer associated with the question.
    - save_path (str, optional): If provided, the path to save the plot as an image. 
                                 If None, the plot is displayed.
    """

    # Convert tensors to PIL images for displaying
    image = transforms.ToPILImage()(image_tensor)
    mask = transforms.ToPILImage()(mask_tensor)

    # Create a matplotlib figure with two subplots: one for the image and one for the mask
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the image
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[0].axis('off')

    # Display the mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Mask")
    axes[1].axis('off')

    # Display the question and answer as the main title
    plt.suptitle(f"Question: {question}\nAnswer: {answer}")

    if save_path:
        # Save the plot as an image
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Plot saved to {save_path}")
    else:
        # Show the plot
        plt.show()


def load_data(transform,batchSize,seed, dataDir):

    test_json_path = os.path.normpath(os.path.join(dataDir, 'test.json'))
    train_json_path = os.path.normpath(os.path.join(dataDir, 'train.json'))
    validate_json_path = os.path.normpath(os.path.join(dataDir, 'validate.json'))


    # Create Dataset objects
    test_dataset = JsonDataset(test_json_path, transform)
    train_dataset = JsonDataset(train_json_path, transform)
    validate_dataset = JsonDataset(validate_json_path, transform)

    # Create DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validate_loader = DataLoader(validate_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))

    return test_loader, train_loader, validate_loader