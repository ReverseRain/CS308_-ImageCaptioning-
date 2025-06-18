"""
Vision encoder module using ViT
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig,AutoModel


class VisionEncoder(nn.Module):
    """Vision encoder using ViT model"""
    
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size
        
    def forward(self, images):
        """
        Args:
            images: batch of images, shape [batch_size, channels, height, width]
        Returns:
            image_features: shape [batch_size, num_patches, hidden_size]
        """
        outputs = self.model(images)
        image_features = outputs.last_hidden_state  # shape: [batch_size, num_patches, hidden_size]
        
        # Exclude the [CLS] token
        patch_features = image_features[:, 1:, :]
        
        return patch_features 