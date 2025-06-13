import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import CLIPImageProcessor


class CocoDataset(Dataset):
    """
    COCO数据集加载器，用于图像描述生成任务
    """
    def __init__(
        self,
        data_dir,
        ann_file,
        tokenizer,
        image_processor,
        max_length=77,
        image_size=224,
        image_token="<image>",
        split="train"
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        self.image_token = image_token
        
        # 加载注释
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # 创建图像ID到文件名的映射
        self.id_to_filename = {}
        for image in coco_data['images']:
            self.id_to_filename[image['id']] = image['file_name']
            
        # 根据分割选择数据
        if split == "train":
            self.annotations = [a for a in coco_data['annotations'] if a['image_id'] % 5 != 0]
        elif split == "val":
            self.annotations = [a for a in coco_data['annotations'] if a['image_id'] % 5 == 0]
        else:
            self.annotations = coco_data['annotations']
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_id = annotation['image_id']
        caption = annotation['caption']
        
        # 构建图像路径
        image_path = os.path.join(self.data_dir, self.id_to_filename[image_id])
        
        # 加载并处理图像
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"][0]
        
        # 构建提示和标签
        # 格式: "请为这张图片生成描述: <image> 这是一只猫在草地上玩耍。"
        prompt = "请为这张图片生成描述："
        text_with_image = f"{prompt} {self.image_token} {caption}"
        
        # Tokenize文本
        tokenized = self.tokenizer(
            text_with_image,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokenized.input_ids[0]
        attention_mask = tokenized.attention_mask[0]
        
        # 创建标签 (用于计算损失)
        # 将提示部分的标签设为-100，仅在caption部分计算损失
        labels = input_ids.clone()
        
        # 找到image token的位置
        image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        image_token_pos = torch.where(input_ids == image_token_id)[0]
        
        if len(image_token_pos) > 0:
            # 将图像令牌之前的部分（提示）设为-100
            labels[:image_token_pos[0]+1] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def prepare_image_text_pair(image, text, image_processor, tokenizer, max_length, image_token):
    """
    准备单个图像-文本对用于推理
    
    Args:
        image: PIL图像
        text: 文本提示
        image_processor: 图像处理器
        tokenizer: 分词器
        max_length: 最大序列长度
        image_token: 图像标记字符串
        
    Returns:
        处理后的图像和文本对
    """
    # 处理图像
    pixel_values = image_processor(image, return_tensors="pt")["pixel_values"]
    
    # 处理文本
    text_with_image = f"{text} {image_token}"
    tokenized = tokenizer(
        text_with_image,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return {
        "pixel_values": pixel_values,
        "input_ids": tokenized.input_ids,
        "attention_mask": tokenized.attention_mask
    } 