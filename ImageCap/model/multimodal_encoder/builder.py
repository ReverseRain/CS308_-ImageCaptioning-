import os
from .transformer_encoder import TransformerVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # Support for different vision models
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("google") or "vit" in vision_tower.lower():
        return TransformerVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # Fallback to default ViT model if no specific model is specified
    return TransformerVisionTower("google/vit-base-patch16-224", args=vision_tower_cfg, **kwargs) 