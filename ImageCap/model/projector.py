"""
Multi-modal projector module
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    Simple MLP projector to map vision features to language model feature space
    """
    
    def __init__(self, vision_hidden_size, language_hidden_size, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(language_hidden_size, language_hidden_size),
            nn.LayerNorm(language_hidden_size)
        )
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: features from vision encoder [batch_size, num_patches, vision_hidden_size]
        Returns:
            projected_features: features projected to language model space [batch_size, num_patches, language_hidden_size]
        """
        return self.projector(vision_features) 