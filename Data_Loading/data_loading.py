import json
import os
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import re
from torch.utils.data import ConcatDataset

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

    def process_string(self,s):
        # Convert to lowercase
        s = s.lower()
        # Remove all whitespace characters except spaces
        s = re.sub(r'[^\S ]+', '', s)
        # Replace multiple spaces with a single space
        s = re.sub(r' +', ' ', s)
        return s.strip()  # Optionally, remove leading/trailing spaces


    def __init__(self, json_file, transform=transforms.ToTensor()):
        self.data = self.load_json(json_file)
        self.data_frame = pd.DataFrame(self.data)        
        self.data_frame = self.data_frame[self.data_frame['q_lang'] == 'en']

          
        # Process and clean question and answer fields
        self.data_frame['question'] = self.data_frame['question'].apply(self.process_string)
        self.data_frame['answer'] = self.data_frame['answer'].apply(self.process_string)
        
        # Remove entries where question or answer is empty after processing
        self.data_frame = self.data_frame[(self.data_frame['question'] != '') & (self.data_frame['answer'] != '')].reset_index(drop=True)

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

        # Get the img
        img_directory = os.path.dirname(img_path)

        # Get the mask path for the current index
        mask_path = os.path.normpath(os.path.join(img_directory, 'mask.png'))
        mt = CustomMaskTransform(self.transform)
        
        # Load the mask and convert it to a tensor
        mask_tensor = self.load_image_as_tensor(mask_path, mt)
                
        return image_tensor, mask_tensor, df_row_dict['question'], df_row_dict['answer']

# Custom Dataset class
class JsonDatasetTest(Dataset):

    def process_string(self,s):
        # Convert to lowercase
        s = s.lower()
        # Remove all whitespace characters except spaces
        s = re.sub(r'[^\S ]+', '', s)
        # Replace multiple spaces with a single space
        s = re.sub(r' +', ' ', s)
        return s.strip()  # Optionally, remove leading/trailing spaces


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
        question_tensor = self.process_string(df_row_dict['question'])

        answer_tensor = self.process_string(df_row_dict['answer'])

        catogory = df_row_dict['answer_type']

        # Get the img
        img_directory = os.path.dirname(img_path)

        # Get the mask path for the current index
        mask_path = os.path.normpath(os.path.join(img_directory, 'mask.png'))
        mt = CustomMaskTransform(self.transform)
        
        # Load the mask and convert it to a tensor
        mask_tensor = self.load_image_as_tensor(mask_path, mt)
                
        return image_tensor, mask_tensor, question_tensor, answer_tensor, catogory

def display_sample(image_tensor, mask_tensor, question, answer, batchsize=1, save_path=None):
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
    if batchsize <= 1:
        # Convert tensors to PIL images for displaying
        image = transforms.ToPILImage()(image_tensor)
        mask = transforms.ToPILImage()(mask_tensor)
    else:
        # Convert tensors to PIL images for displaying
        image = transforms.ToPILImage()(image_tensor[0])
        mask = transforms.ToPILImage()(mask_tensor[0])

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
    if batchsize <=1:
        plt.suptitle(f"Question: {question}\nAnswer: {answer}")
    else:
        plt.suptitle(f"Question: {question[0]}\nAnswer: {answer[0]}")


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
    train_json_path = os.path.normpath(os.path.join(dataDir, 'train.json'))
    validate_json_path = os.path.normpath(os.path.join(dataDir, 'validate.json'))

    # Create Dataset objects
    train_dataset = JsonDataset(train_json_path, transform)
    validate_dataset = JsonDataset(validate_json_path, transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed),num_workers=1, prefetch_factor=1, pin_memory=False)
    validate_loader = DataLoader(validate_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed),num_workers=1, prefetch_factor=1, pin_memory=False)

    return train_loader, validate_loader

def load_data_cross_val(transform, dataDir):
    train_json_path = os.path.normpath(os.path.join(dataDir, 'train.json'))
    validate_json_path = os.path.normpath(os.path.join(dataDir, 'validate.json'))

    # Create Dataset objects
    train_dataset = JsonDataset(train_json_path, transform)
    validate_dataset = JsonDataset(validate_json_path, transform)

    return ConcatDataset([train_dataset, validate_dataset])

def load_test_data(transform,batchSize,seed, dataDir):

    test_json_path = os.path.normpath(os.path.join(dataDir, 'test.json'))


    # Create Dataset objects
    test_dataset = JsonDatasetTest(test_json_path, transform)


    # Create DataLoader objects
    test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))

    return test_loader

class CLIPTrainJsonDataset(JsonDataset):
    def __init__(self, json_file, transform=transforms.ToTensor()):
        super().__init__(json_file,transform)
        self.data = self.load_json(json_file)
        self.data_frame = pd.DataFrame(self.data)
        self.data_frame = self.data_frame[self.data_frame['q_lang'] == 'en']
        grouped_data = self.data_frame.groupby('img_id')
        columns=["img_id","img_name","text"]
        self.data_frame = pd.DataFrame(columns=columns)
    
        for img_id, group in grouped_data:
            text = ""
            imageName = ""
            for index, row in group.iterrows():
                text += row['question'] + " " + row['answer']
                text += ", " 
                imageName = row['img_name']

            text = text[:-2]
             
            new_row = {
                "img_id": img_id,
                "img_name": imageName,
                "text": text
            }
           # Convert the dictionary to a DataFrame
            new_row_df = pd.DataFrame([new_row])

            # Append the new row DataFrame to the existing DataFrame
            self.data_frame = pd.concat([self.data_frame, new_row_df], ignore_index=True)

        self.transform = transform
        dir = os.path.join(os.getcwd(),'Slake1.0', 'imgs')
        self.img_dir = os.path.normpath(dir)
    
    def __getitem__(self, idx):
        # Get the image path for the current index
        img_path = os.path.normpath(os.path.join(self.img_dir, *self.data_frame.iloc[idx]['img_name'].split('/')))

        # Load the image and convert it to a tensor
        image_tensor = self.load_image_as_tensor(img_path, self.transform)
        
        # Get the row from the DataFrame as a dictionary
        df_row_dict = self.data_frame.iloc[idx].to_dict()

        # word tokenization
        text = self.process_string(df_row_dict['text'])

        # Get the img
        img_directory = os.path.dirname(img_path)

        # Get the mask path for the current index
        mask_path = os.path.normpath(os.path.join(img_directory, 'mask.png'))
        mt = CustomMaskTransform(self.transform)
        
        # Load the mask and convert it to a tensor
        mask_tensor = self.load_image_as_tensor(mask_path, mt)
                
        return image_tensor, mask_tensor, text

from medmnist import ChestMNIST

class CustomChestMNISTDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.dataset = ChestMNIST(split=split, download=True)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor()  # Default to just converting to tensor if no transform is provided
        ])
       
        
        # Map numeric labels to class names
        self.class_names = (
            "Atelectasis", "Cardiomegaly", "Consolidation", 
            "Edema", "Effusion", "Emphysema", 
            "Fibrosis", "Hernia", "Mass", 
            "Nodule", "Pleural_Thickening", "Pneumonia", 
            "Pneumothorax", "Normal"  # List all possible classes
        )

         # Define the number of classes
        self.num_classes = len(self.class_names)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        
        # Convert image to tensor
        #image = torch.tensor(image, dtype=torch.float32)

        # Apply transformation
        image = self.transform(image)

        # Convert label to one-hot encoding
        #target = self._to_one_hot(label)
        label = int(torch.argmax(torch.tensor(label)))
        
        return image, label

    def get_classes(self):
        # Return the class names directly
        return self.class_names

    def _to_one_hot(self, label):
        # Create a one-hot encoded tensor for the label
        one_hot = torch.zeros(self.num_classes)
        one_hot[label] = 1.0
        return one_hot
    
def load_chest_mnist_data(transform, batch_size, seed, split='test'):
    """
    Load the ChestMNIST dataset as a DataLoader.
    
    Parameters:
        transform: The transformations to apply to the dataset.
        batch_size: The batch size for the DataLoader.
        seed: The random seed for shuffling.
        split: The dataset split ('train' or 'test').
        
    Returns:
        DataLoader: DataLoader object for the ChestMNIST dataset.
    """
    # Create an instance of the dataset
    dataset = CustomChestMNISTDataset(split=split, transform=transform)

    # Create DataLoader object
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        generator=torch.Generator().manual_seed(seed)
    )

    return dataloader

def load_clip_eval_test_data(transform,batchSize,seed, dataDir):
    test_json_path = os.path.normpath(os.path.join(dataDir, 'test.json'))
    test_dataset = CLIPTrainJsonDataset(test_json_path, transform)

    # Create DataLoader objects
    train_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))

    return train_loader

def load_combined_text_data(transform,batchSize,seed, dataDir):

    train_json_path = os.path.normpath(os.path.join(dataDir, 'train.json'))
    validate_json_path = os.path.normpath(os.path.join(dataDir, 'validate.json'))

    # Create Dataset objects
    train_dataset = CLIPTrainJsonDataset(train_json_path, transform)
    validate_dataset = CLIPTrainJsonDataset(validate_json_path, transform)

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))
    validate_loader = DataLoader(validate_dataset, batch_size=batchSize, shuffle=True, generator=torch.Generator().manual_seed(seed))

    return train_loader, validate_loader