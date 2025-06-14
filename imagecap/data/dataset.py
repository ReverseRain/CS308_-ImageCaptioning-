import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

from ..mm_utils import process_images
from ..constants import DEFAULT_IMAGE_TOKEN
from ..conversation import get_default_conv_template

class ImageCaptionDataset(Dataset):
    """
    Dataset for image captioning training.
    """
    def __init__(self, data_path, tokenizer, image_processor=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the JSON file containing image-text pairs.
            tokenizer: Tokenizer for encoding text.
            image_processor: Processor for processing images.
        """
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.conv_template = get_default_conv_template()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item["image"]
        caption = item["caption"]
        
        # Load and process image
        image = Image.open(image_path).convert("RGB")
        if self.image_processor:
            image_tensor = self.image_processor(image, return_tensors="pt")
        else:
            # Simple resize if no processor provided
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image).unsqueeze(0)
        
        # Prepare conversation with image token and caption
        conv = self.conv_template
        conv.messages = []  # Reset conversation
        conv.add_message("user", f"{DEFAULT_IMAGE_TOKEN} Please describe this image.")
        conv.add_message("assistant", caption)
        
        prompt = conv.get_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512)
        
        # Create target tokens for causal language modeling
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        
        # Find the assistant's response and apply loss only on that part
        # Set -100 to parts that we don't want to include in loss calculation
        user_prompt = f"user: {DEFAULT_IMAGE_TOKEN} Please describe this image."
        user_prompt_ids = self.tokenizer(user_prompt, return_tensors="pt")["input_ids"][0]
        labels[:len(user_prompt_ids)] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "labels": labels,
            "images": image_tensor[0] if isinstance(image_tensor, list) else image_tensor
        } 