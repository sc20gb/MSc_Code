import torch
import json
import os
from tqdm import tqdm
import numpy as np

class PreEmbeddingCreator:
    """Creates and saves embeddings for images using a given encoder."""
    
    def __init__(self, encoder, data_loader, save_dir, device='cuda'):
        """
        Args:
            encoder: PyTorch model for creating embeddings
            data_loader: DataLoader containing images, and optionally additional data
            save_dir: Directory to save per-image JSON files
            device: Device to run encoder on
        """
        self.encoder = encoder.to(device)
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def create_embeddings(self):
        """Creates embeddings for all images in the dataloader and saves each as a separate JSON file."""
        self.encoder.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                # Handle both tuple/list inputs and direct tensor inputs
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                    # Store all additional data (only shape info for images)
                    batch_data = {
                        'images': batch[0].shape,
                        **{f'data_{i}': item for i, item in enumerate(batch[1:], 1)}
                    }
                else:
                    images = batch
                    batch_data = {'images': images.shape}
                
                # Get embeddings for batch
                images = images.to(self.device)
                embeddings = self.encoder(images)
                
                # Convert embeddings to CPU numpy arrays
                embeddings = embeddings.cpu().numpy()
                
                # Save each embedding and its batch data in a separate file
                for idx in range(len(images)):
                    item = {
                        'embedding': embeddings[idx].tolist(),
                        'batch_data': {
                            k: v[idx] if isinstance(v, (torch.Tensor, list, tuple)) 
                              else v for k, v in batch_data.items()
                        }
                    }
                    key = f"img_{batch_idx}_{idx}"
                    file_path = os.path.join(self.save_dir, f"{key}.json")
                    with open(file_path, 'w') as f:
                        json.dump(item, f)
                        
class PreEmbeddingLoader:
    """Loads pre-computed embeddings for images from individual JSON files."""
    
    def __init__(self, embeddings_dir):
        """
        Args:
            embeddings_dir: Directory containing JSON files with embeddings
        """
        self.embeddings_dir = embeddings_dir
        
    def get_embedding(self, img_id):
        """Loads the embedding and associated data for a specific image ID.
        
        Args:
            img_id: Image identifier matching the filename (e.g., 'img_0_1')
            
        Returns:
            tuple: (embedding array, batch_data dictionary)
        """
        file_path = os.path.join(self.embeddings_dir, f"{img_id}.json")
        if not os.path.exists(file_path):
            raise KeyError(f"No file found for image {img_id}")
            
        with open(file_path, 'r') as f:
            item = json.load(f)
        return np.array(item['embedding']), item['batch_data']