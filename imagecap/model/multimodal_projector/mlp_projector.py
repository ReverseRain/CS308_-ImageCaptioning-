import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """
    MLP projector for connecting vision features to language features
    """
    def __init__(self, vision_hidden_size, language_hidden_size, hidden_size=None):
        """
        Initialize the MLP projector
        
        Args:
            vision_hidden_size: Vision feature dimension
            language_hidden_size: Language feature dimension
            hidden_size: Hidden layer dimension, defaults to vision_hidden_size
        """
        super().__init__()
        
        if hidden_size is None:
            hidden_size = vision_hidden_size
        
        self.vision_hidden_size = vision_hidden_size
        self.language_hidden_size = language_hidden_size
        self.hidden_size = hidden_size
        
        # Build two-layer MLP
        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, language_hidden_size)
        )
    
    def forward(self, vision_features):
        """
        Forward pass, project vision features to language space
        
        Args:
            vision_features: Vision features [batch_size, num_patches, vision_hidden_size]
            
        Returns:
            projected_features: Projected features [batch_size, num_patches, language_hidden_size]
        """
        projected_features = self.projector(vision_features)
        return projected_features 