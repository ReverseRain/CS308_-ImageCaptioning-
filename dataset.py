from model import VisionCaptionModel
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
import requests
import zipfile
from tqdm import tqdm
from datasets import load_dataset

def download_coco_images(data_dir):
    """下载COCO2014数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    files = [
        ("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip"),
        ("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip")
    ]
    
    # 下载并解压文件
    for url, filename in files:
        filepath = os.path.join(data_dir, filename)
        unzip_dir = os.path.join(data_dir, filename.replace(".zip", ""))
        
        # 如果文件已存在且解压目录存在，则跳过
        if os.path.exists(unzip_dir) and os.path.isdir(unzip_dir):
            print(f"{unzip_dir} already exists. Skipping download.")
            continue
            
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024), 
                            total=total_size//1024, 
                            unit='KB', 
                            unit_scale=True):
                f.write(data)
                
        print(f"Unzipping {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
            
        os.remove(filepath)

class ImageCaptionDataset(Dataset):
    """自定义图像字幕数据集"""
    def __init__(self, image_dir, captions_data, transform=None, split="train"):
        """
        参数:
            image_dir: 图像目录路径
            captions_data: Hugging Face数据集对象
            transform: 图像变换
            split: 数据集分割 (train/val)
        """
        self.image_dir = image_dir
        self.captions_data = captions_data
        self.transform = transform
        self.split = split
        
        # 创建图像ID到文件名的映射
        self.id_to_filename = {}
        for item in captions_data:
            image_id = item["image_id"]
            # 根据split构造文件名
            if split == "train":
                filename = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
            else:  # val
                filename = f"COCO_val2014_{str(image_id).zfill(12)}.jpg"
            self.id_to_filename[image_id] = filename
    
    def __len__(self):
        return len(self.captions_data)
    
    def __getitem__(self, idx):
        item = self.captions_data[idx]
        image_id = item["image_id"]
        caption = item["caption"]
        
        # 获取图像路径
        filename = self.id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)
        
        # 加载图像
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            # 如果图像不存在，创建一个空白图像
            print(f"Warning: Image not found at {image_path}. Using placeholder.")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, caption

def collate_fn(batch, tokenizer):
    """自定义批处理函数"""
    images, captions = zip(*batch)
    images = torch.stack(images)
    
    # Tokenize文本
    text_inputs = tokenizer(
        captions, 
        padding='longest', 
        return_tensors='pt',
        max_length=128,
        truncation=True
    )
    
    return images, text_inputs.input_ids, text_inputs.attention_mask

import re
def clean_text(text):
    """处理特殊符号问题"""
    text = re.sub(r"\.([A-Z])", r" \1", text)  
    
    return text.replace("_", " ").replace("-", " ")

from transformers import AutoTokenizer
def text_to_input_ids(text):
    tokenizer = AutoTokenizer.from_pretrained("Qwen3")
    tokenizer.to('cuda')
    text=text.to('cuda')
    inputs = tokenizer(
        text,
        padding="max_length",    # 自动填充到统一长度
        truncation=True,          # 超过最大长度时截断
        max_length=128,           # 自定义最大长度
        return_tensors="pt"       # 返回 PyTorch 张量（可选 "tf" 为 TensorFlow）
    )
    return inputs["input_ids"]


from datasets import load_from_disk
if __name__ == "__main__":
    print("  aaaaaaaaaaaaaaaaaaqwwwwwqwi ")

    
    
    # print("aaaaaaaaaaaaaaaa")

    dataset = load_dataset(
    "parquet",
    data_files = {
    "train": r".\COCO_Images\data\train-*.parquet",
    "validation": r".\COCO_Images\data\val-*.parquet",
    "test": r".\COCO_Images\data\test-*.parquet"
},
    # features=features,
    streaming=False  # 关闭流式模式以启用完整操作
)

    # ds1 = ds1.map(
    #     lambda example: {"input_ids": text_to_input_ids(example["caption"])},
    #     batched=True  
    # )
    
    dataset.save_to_disk("./coco2014_iamges")
    print("finish")