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
            data_loader: DataLoader containing images
            save_dir: Directory to save embeddings
            device: Device to run encoder on
        """
        self.encoder = encoder.to(device)
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def create_embeddings(self):
        """Creates embeddings for all images in the dataloader."""
        self.encoder.eval()
        embeddings_dict = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                # Handle both tuple/list inputs and direct tensor inputs
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                    # Store all additional data
                    batch_data = {
                        'images': batch[0].shape,  # Store shape info only
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
                
                # Store embeddings with batch data
                for idx in range(len(images)):
                    item = {
                        'embedding': embeddings[idx].tolist(),
                        'batch_data': {
                            k: v[idx] if isinstance(v, (torch.Tensor, list, tuple)) 
                              else v for k, v in batch_data.items()
                        }
                    }
                    key = f"img_{batch_idx}_{idx}"
                    embeddings_dict[key] = item
        
        # Save embeddings to JSON
        save_path = os.path.join(self.save_dir, 'embeddings.json')
        with open(save_path, 'w') as f:
            json.dump(embeddings_dict, f)
            
class PreEmbeddingLoader:
    """Loads pre-computed embeddings for images."""
    
    def __init__(self, embeddings_path):
        """
        Args:
            embeddings_path: Path to JSON file containing embeddings
        """
        self.embeddings_path = embeddings_path
        self.embeddings = self._load_embeddings()
        
    def _load_embeddings(self):
        """Loads embeddings from JSON file."""
        with open(self.embeddings_path, 'r') as f:
            return json.load(f)
    
    def get_embedding(self, img_id):
        """Gets embedding and associated data for a specific image ID.
        
        Args:
            img_id: Image identifier
            
        Returns:
            tuple: (embedding array, batch_data dictionary)
        """
        if img_id not in self.embeddings:
            raise KeyError(f"No embedding found for image {img_id}")
            
        item = self.embeddings[img_id]
        return np.array(item['embedding']), item['batch_data']