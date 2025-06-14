import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .imagecap_arch import ImageCapModel
from .multimodal_encoder.clip_encoder import CLIPVisionTower
from .multimodal_projector.mlp_projector import MLPProjector

def build_imagecap_model(
    vision_tower_name="openai/clip-vit-large-patch14",
    language_model_name="Qwen/Qwen-0.6B",
    pretrained_model_path=None,
    freeze_vision_tower=True,
    freeze_language_model=False
):
    """
    Build and initialize the ImageCapModel.
    """
    # Initialize the vision encoder
    vision_tower = CLIPVisionTower(vision_tower_name)
    
    # Initialize the language model
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
    
    # Initialize the projector
    projector = MLPProjector(
        vision_hidden_size=vision_tower.hidden_size,
        language_hidden_size=language_model.config.hidden_size
    )
    
    # Create the full model
    model = ImageCapModel(
        vision_tower=vision_tower,
        language_model=language_model,
        projector=projector,
        tokenizer=tokenizer
    )
    
    # Freeze components as specified
    if freeze_vision_tower:
        for param in model.vision_tower.parameters():
            param.requires_grad = False
    
    if freeze_language_model:
        for param in model.language_model.parameters():
            param.requires_grad = False
    
    # Load pre-trained weights if available
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
    
    return model 