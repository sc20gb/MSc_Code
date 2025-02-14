import torch
import json
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset

# TODO: To support use cases in the future where multple tensors are passed in the dataloader
# TODO: Make the json file updatable and saved with each batch to save on ram for large datasets
# TODO: Add support for checking if the whole dataset has been created or not. and if not then continue from where it left off
class PreEmbeddingCreator:
    """Creates and saves embeddings for images using a given encoder."""
    
    def __init__(self, encoder, data_loader, save_dir, hidden_layer_from_end, device='cuda'):
        """
        Args:
            encoder: PyTorch model for creating embeddings.
            data_loader: DataLoader containing images (and optionally additional data).
            save_dir: Directory to save per-image tensor files and the manifest JSON file.
            device: Device to run the encoder on.
        """
        self.hidden_layer_from_end = hidden_layer_from_end
        self.encoder = encoder.to(device)
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.device = device
        
        # Create save directory if it doesn't exist.
        os.makedirs(save_dir, exist_ok=True)
        
    def create_embeddings(self):
        """Creates embeddings for all images in the dataloader and saves each as a separate tensor file.
           Also creates a manifest JSON file mapping each image ID to its tensor file address and
           accompanying batch data. Extra data that is a tensor is skipped.
        """
        self.encoder.eval()
        manifest = {}  # Mapping for each image.
        
        # Helper function to safely convert extra data to JSON-serializable types.
        def safe_convert(val):
            # If the value is a tensor, skip it.
            if isinstance(val, torch.Tensor):
                return None  # Skip saving tensors.
            elif isinstance(val, (list, tuple)):
                converted = []
                for elem in val:
                    if isinstance(elem, torch.Tensor):
                        continue
                    else:
                        converted.append(elem)
                return converted
            else:
                try:
                    json.dumps(val)
                    return val
                except TypeError:
                    return str(val)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.data_loader)):
                # Handle both tuple/list inputs and direct tensor inputs.
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                    batch_data = {
                        'images_shape': list(batch[0].shape),
                        **{f'data_{i}': item for i, item in enumerate(batch[1:], 1)}
                    }
                else:
                    images = batch
                    batch_data = {'images_shape': list(images.shape)}
                
                # Get embeddings for the batch.
                images = images.to(self.device)
                _, hidden_states = self.encoder(images)
                embeddings = hidden_states[(len(hidden_states) - 1) - self.hidden_layer_from_end]
                embeddings = embeddings.cpu()
                
                # Save each image's embedding separately.
                for idx in range(len(images)):
                    safe_batch_data = {}
                    for k, v in batch_data.items():
                        if isinstance(v, (torch.Tensor, list, tuple)):
                            try:
                                safe_val = safe_convert(v[idx])
                                if safe_val is not None:
                                    safe_batch_data[k] = safe_val
                            except Exception:
                                continue
                        else:
                            safe_batch_data[k] = v
                    
                    item = {
                        'embedding': embeddings[idx],
                        'batch_data': safe_batch_data
                    }
                    key = f"img_{batch_idx}_{idx}"
                    pt_file = os.path.join(self.save_dir, f"{key}.pt")
                    torch.save(item, pt_file)
                    
                    manifest[key] = {
                        "pt_file": pt_file,
                        "batch_data": safe_batch_data
                    }
        
        manifest_file = os.path.join(self.save_dir, "embedding_manifest.json")
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest for {len(manifest)} images to '{manifest_file}'.")


class PreEmbeddingDataset(Dataset):
    """Loads pre-computed embeddings for images using a manifest JSON file."""
    
    def __init__(self, embeddings_dir):
        """
        Args:
            embeddings_dir: Directory containing the .pt files and the manifest JSON file.
        """
        self.embeddings_dir = embeddings_dir
        self.manifest_file = os.path.join(embeddings_dir, "embedding_manifest.json")
        
        if not os.path.exists(self.manifest_file):
            raise ValueError(f"No manifest file found in directory {embeddings_dir}")
            
        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            self.manifest = json.load(f)
            
        # List of keys corresponding to each stored image embedding.
        self.keys = list(self.manifest.keys())
        
    def __len__(self):
        return len(self.keys)
        
    def get_embedding(self, img_id):
        """Loads the embedding and associated data for a specific image ID.
        
        Args:
            img_id: Image identifier matching one of the keys in the manifest (e.g., 'img_0_1').
            
        Returns:
            tuple: (embedding tensor, batch_data dictionary)
        """
        if img_id not in self.manifest:
            raise KeyError(f"No entry found for image {img_id} in the manifest.")
            
        pt_file = self.manifest[img_id]['pt_file']
        if not os.path.exists(pt_file):
            raise FileNotFoundError(f"The tensor file for image {img_id} was not found at {pt_file}.")
        
        item = torch.load(pt_file)
        return item['embedding']
    def get_extra_data(self, img_id):
        """Returns extra data associated with an image ID excluding the embedding filename.
        
        Args:
            img_id: Image identifier matching one of the keys in the manifest (e.g., 'img_0_1').
            
        Returns:
            dict: Extra data associated with the image ID, excluding the 'pt_file' entry.
        """
        if img_id not in self.manifest:
            raise KeyError(f"No entry found for image {img_id} in the manifest.")
        
        data = self.manifest[img_id].copy()
        data.pop("pt_file", None)
        return data
    
    def __getitem__(self, idx):
        img_id = self.keys[idx]
        return self.get_embedding(img_id), self.get_extra_data(img_id)


