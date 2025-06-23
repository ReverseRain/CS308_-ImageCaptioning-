"""
COCO dataset module for image captioning
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTImageProcessor,AutoImageProcessor
from tqdm import tqdm  # 导入 tqdm 进度条库

class COCOCaptionDataset(Dataset):
    """
    COCO dataset for image captioning
    """
    
    def __init__(self, image_dir, annotation_file, image_processor_path, max_length=77, max_samples=None):
        """
        Args:
            image_dir: directory containing image files
            annotation_file: COCO annotation file
            image_processor_path: path to the ViT image processor
            max_length: maximum length of captions
            max_samples: maximum number of samples to use (None for all)
        """
        self.image_dir = image_dir
        self.max_length = max_length
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Process annotations
        self.data = []

        image_id_to_file = {img['id']: img['file_name'] for img in self.annotations['images']}
        
        annotations = self.annotations['annotations']
        if max_samples is not None and max_samples < len(annotations):
            annotations = annotations[:max_samples]
            print(f"Using {max_samples} samples out of {len(self.annotations['annotations'])}")
        cnt=0
        for ann in tqdm(annotations, desc="Processing images"):
            cnt+=1
            if(cnt>11000):
                break
            image_id = ann['image_id']
            caption = ann['caption']
            image_filename = image_id_to_file.get(image_id)  # O(1)查询
            
            if image_filename:
                self.data.append({
                    'image_id': image_id,
                    'image_file': os.path.join(self.image_dir, image_filename),
                    'caption': caption
                })
        
        # Image processor for ViT
        print("path of process ",image_processor_path)
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_path,use_fast=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and preprocess image
        try:
            image = Image.open(item['image_file']).convert('RGB')
            image_inputs = self.image_processor(images=image, return_tensors="pt")
            image_tensor = image_inputs.pixel_values.squeeze(0)  # Remove batch dim
        except Exception as e:
            # If image loading fails, return a simple error indicator
            print(f"Error loading image {item['image_file']}: {e}")
            # Return zeros with proper shape (3 channels x height x width)
            image_tensor = torch.zeros((3, 224, 224))
        
        return {
            'image': image_tensor,
            'caption': item['caption'],
            'image_id': item['image_id']
        }


def collate_fn(batch):
    """
    Collate function for DataLoader
    """
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'captions': captions,
        'image_ids': image_ids
    } 