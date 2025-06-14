import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
from transformers import AutoTokenizer


class COCOCaptionDataset(Dataset):
    """
    Dataset for loading COCO captioning data
    """
    def __init__(self, annotation_file, image_dir, tokenizer=None, transform=None, max_length=77, split='train'):
        """
        Args:
            annotation_file (string): Path to the annotation json file
            image_dir (string): Directory with all the images
            tokenizer: Tokenizer for processing text
            transform: Transform to apply to images
            max_length (int): Maximum length of tokenized caption
            split (string): 'train', 'val', or 'test'
        """
        self.image_dir = image_dir
        self.transform = transform
        self.max_length = max_length
        self.split = split
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Process annotations
        if 'annotations' in self.annotations:
            self.captions = self.annotations['annotations']
        else:
            self.captions = self.annotations['images']
        
        # Create image_id to filename mapping
        self.id_to_filename = {}
        for img in self.annotations['images']:
            self.id_to_filename[img['id']] = img['file_name']
            
        # Setup tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
        else:
            self.tokenizer = tokenizer
            
        # Special tokens for image captioning
        self.image_start_token = "<image>"
        self.image_end_token = "</image>"
        self.add_caption_token = "<caption>"
        
        # Add special tokens to tokenizer if they don't exist
        special_tokens = []
        for token in [self.image_start_token, self.image_end_token, self.add_caption_token]:
            if token not in self.tokenizer.vocab:
                special_tokens.append(token)
                
        if special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Processed image
            input_ids: Tokenized caption
            attention_mask: Attention mask for tokenized caption
            labels: Labels for the caption (same as input_ids but with -100 for tokens we don't want to predict)
        """
        caption_info = self.captions[idx]
        
        # Get image
        if 'image_id' in caption_info:
            image_id = caption_info['image_id']
        else:
            image_id = caption_info['id']
            
        image_filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformation if provided
        if self.transform is not None:
            image = self.transform(image)
            
        # Get caption
        if 'caption' in caption_info:
            caption = caption_info['caption']
        else:
            # For files with multiple captions per image, select one randomly
            caption = random.choice(caption_info['sentences'])['raw']
            
        # Format with special tokens
        formatted_caption = f"{self.image_start_token} {self.add_caption_token} {caption} {self.image_end_token}"
        
        # Tokenize caption
        tokenized = self.tokenizer(
            formatted_caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create labels (same as input_ids but with -100 for tokens we don't want to predict)
        labels = tokenized.input_ids.clone()
        
        # Set up labels to only predict the caption part (after the image_start and add_caption tokens)
        # Find the position of the add_caption_token
        add_caption_token_id = self.tokenizer.convert_tokens_to_ids(self.add_caption_token)
        add_caption_pos = (labels == add_caption_token_id).nonzero(as_tuple=True)[1][0]
        
        # Set labels before add_caption_token to -100 (don't predict these)
        labels[0, :add_caption_pos+1] = -100
        
        return {
            'image': image,
            'input_ids': tokenized.input_ids.squeeze(0),
            'attention_mask': tokenized.attention_mask.squeeze(0),
            'labels': labels.squeeze(0)
        }


def create_coco_dataloaders(annotation_train_file, annotation_val_file, image_dir, 
                          tokenizer, transform, batch_size=16, max_length=77):
    """
    Create dataloaders for training and validation
    """
    train_dataset = COCOCaptionDataset(
        annotation_file=annotation_train_file,
        image_dir=image_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        split='train'
    )
    
    val_dataset = COCOCaptionDataset(
        annotation_file=annotation_val_file,
        image_dir=image_dir,
        tokenizer=tokenizer,
        transform=transform,
        max_length=max_length,
        split='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader 