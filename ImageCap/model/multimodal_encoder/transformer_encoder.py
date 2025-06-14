import torch
import torch.nn as nn
from transformers import AutoModel, AutoFeatureExtractor


class TransformerVisionTower(nn.Module):
    def __init__(self, vision_tower, args=None, delay_load=False):
        super().__init__()
        
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.delay_load = delay_load
        
        if not delay_load:
            self.load_model()
    
    def load_model(self):
        self.image_processor = AutoFeatureExtractor.from_pretrained(self.vision_tower_name)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_name)
        
        # Set device to match current device if available
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        self.vision_tower.to(device)
        
        # Freeze the vision encoder by default
        for param in self.vision_tower.parameters():
            param.requires_grad = False
            
        self.is_loaded = True
    
    @property
    def hidden_size(self):
        if not self.is_loaded:
            self.load_model()
        return self.vision_tower.config.hidden_size
    
    @property
    def num_patches_per_side(self):
        if hasattr(self.vision_tower.config, 'image_size') and hasattr(self.vision_tower.config, 'patch_size'):
            return self.vision_tower.config.image_size // self.vision_tower.config.patch_size
        else:
            # Default for ViT-like models
            return 14  # 224 / 16 = 14
    
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()
        
        if type(images) is list:
            image_features = []
            for image in images:
                # Process a single image
                image_forward = self.vision_tower(image, output_hidden_states=True)
                
                # Get the hidden states from the specified layer
                hidden_states = image_forward.hidden_states[self.select_layer]
                
                if self.select_feature == 'patch':
                    # For ViT: Extract patch embeddings (skip the [CLS] token)
                    image_feature = hidden_states[:, 1:]
                elif self.select_feature == 'cls':
                    # For ViT: Only use the [CLS] token features
                    image_feature = hidden_states[:, 0:1]
                else:
                    raise ValueError(f"Unsupported feature selection: {self.select_feature}")
                    
                image_features.append(image_feature)
            
            # Concat all image features
            image_features = torch.cat(image_features, dim=0)
        else:
            # Process batch of images
            image_forward = self.vision_tower(images, output_hidden_states=True)
            
            # Get the hidden states from the specified layer
            hidden_states = image_forward.hidden_states[self.select_layer]
            
            if self.select_feature == 'patch':
                # For ViT: Extract patch embeddings (skip the [CLS] token)
                image_features = hidden_states[:, 1:]
            elif self.select_feature == 'cls':
                # For ViT: Only use the [CLS] token features
                image_features = hidden_states[:, 0:1]
            else:
                raise ValueError(f"Unsupported feature selection: {self.select_feature}")
        
        return image_features
    
    @torch.no_grad()
    def preprocess(self, images):
        if not self.is_loaded:
            self.load_model()
            
        if isinstance(images, list):
            # Process a list of images
            pixel_values = [self.image_processor(img, return_tensors='pt')['pixel_values'].squeeze(0) for img in images]
            return pixel_values
        else:
            # Process a single image or a batch of images
            pixel_values = self.image_processor(images, return_tensors='pt')['pixel_values']
            return pixel_values 