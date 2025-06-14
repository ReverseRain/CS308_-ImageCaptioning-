import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .multimodal_encoder import build_vision_tower
from .multimodal_projector import build_vision_projector


class ImageCaptioningModel(nn.Module):
    """
    Image Captioning model that connects:
    1. A transformer-based vision encoder 
    2. An MLP connector
    3. Qwen-3-0.6B language model
    """
    def __init__(self, config):
        super().__init__()
        
        # Initialize vision encoder
        self.vision_tower = build_vision_tower(config)
        
        # Initialize the language model (Qwen-3-0.6B)
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model_path,
            torch_dtype=torch.float16 if getattr(config, 'use_fp16', False) else torch.float32
        )
        
        # Set up the config parameters
        self.config = config
        if not hasattr(self.config, 'mm_hidden_size'):
            self.config.mm_hidden_size = self.vision_tower.hidden_size
        if not hasattr(self.config, 'hidden_size'):
            self.config.hidden_size = self.language_model.config.hidden_size
            
        # Initialize the MLP projector (connector)
        self.mm_projector = build_vision_projector(self.config)
        
        # Special tokens for image captioning
        self.image_start_token = getattr(config, 'image_start_token', "<image>")
        self.image_end_token = getattr(config, 'image_end_token', "</image>")
        self.add_caption_token = getattr(config, 'add_caption_token', "<caption>")
        
        # Add special tokens to tokenizer if they don't exist
        special_tokens = []
        for token in [self.image_start_token, self.image_end_token, self.add_caption_token]:
            if token not in self.tokenizer.vocab:
                special_tokens.append(token)
                
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
            # Resize token embeddings
            self.language_model.resize_token_embeddings(len(self.tokenizer))
    
    def encode_images(self, images):
        """Process images through vision encoder and connector"""
        # Get image features from vision tower
        image_features = self.vision_tower(images)
        # Project image features to language model dimension
        projected_features = self.mm_projector(image_features)
        return projected_features
    
    def generate_caption(self, images, prompt=None, max_length=50, **generate_kwargs):
        """Generate a caption for the given images"""
        # Process the images
        with torch.no_grad():
            image_features = self.encode_images(images)
        
        # Create prompt with special tokens if not provided
        if prompt is None:
            prompt = f"{self.image_start_token} {self.add_caption_token}"
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(image_features.device)
        
        # Generate caption
        generate_kwargs.update({"max_length": max_length})
        outputs = self.language_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=(image_features, None),  # Pass image features as conditioning
            **generate_kwargs
        )
        
        # Decode the generated outputs
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions
    
    def forward(self, images, input_ids=None, labels=None, attention_mask=None):
        """Forward pass for training"""
        # Process images
        image_features = self.encode_images(images)
        
        # If no input is provided, just return the image features
        if input_ids is None:
            return image_features
            
        # Pass image features and input to the language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=(image_features, None)  # Pass image features as conditioning
        )
        
        return outputs 