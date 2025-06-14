import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class MLPProjector(nn.Module):
    """
    MLP Projector for connecting vision encoder and language model
    """
    def __init__(self, input_size, output_size, hidden_size=None, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size or output_size
        
        layers = []
        current_size = input_size
        
        # Create MLP layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_size, self.hidden_size))
            layers.append(nn.GELU())
            current_size = self.hidden_size
            
        # Final layer
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'mlp':
        # Default to 2-layer MLP with GELU activation
        return MLPProjector(
            input_size=config.mm_hidden_size, 
            output_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=2
        )

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        return MLPProjector(
            input_size=config.mm_hidden_size, 
            output_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=mlp_depth
        )

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}') 