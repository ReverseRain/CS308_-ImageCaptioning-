import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor

class CLIPVisionTower(nn.Module):
    """
    CLIP Vision Tower for encoding images.
    """
    def __init__(self, vision_tower_name, cache_dir=None):
        super().__init__()
        
        self.vision_tower_name = vision_tower_name
        self.vision_tower = CLIPVisionModel.from_pretrained(
            vision_tower_name, cache_dir=cache_dir
        )
        
        self.image_processor = CLIPImageProcessor.from_pretrained(
            vision_tower_name, cache_dir=cache_dir
        )
        
        # Get hidden size from the configuration
        self.hidden_size = self.vision_tower.config.hidden_size
        
    def forward(self, images):
        """
        Forward pass through the vision tower.
        
        Args:
            images: Preprocessed images
            
        Returns:
            Image features from the vision tower
        """
        if isinstance(images, list):
            # Handle batch of images
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.unsqueeze(0)).last_hidden_state
                image_features.append(image_forward_out)
            image_features = torch.cat(image_features, dim=0)
        else:
            # Handle single image
            image_features = self.vision_tower(images).last_hidden_state
            
        return image_features 