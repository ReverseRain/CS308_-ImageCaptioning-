import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from .processor import prepare_training_inputs

class COCOCaptionDataset(Dataset):
    """
    COCO图像标注数据集
    """
    def __init__(self, image_folder, annotation_file, processor, max_length=512):
        """
        初始化COCO数据集
        
        Args:
            image_folder: 图像文件夹路径
            annotation_file: 标注文件路径
            processor: 处理器
            max_length: 最大序列长度
        """
        self.image_folder = image_folder
        self.processor = processor
        self.max_length = max_length
        
        # 加载注释
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建图像ID到文件名的映射
        self.image_id_to_filename = {}
        for img in data['images']:
            self.image_id_to_filename[img['id']] = img['file_name']
        
        # 提取图像和标注
        print("Processing annotations...")
        self.examples = []
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id in self.image_id_to_filename:
                file_name = self.image_id_to_filename[image_id]
                image_path = os.path.join(image_folder, file_name)
                
                # 只添加存在的图像
                if os.path.exists(image_path):
                    self.examples.append({
                        'image_id': image_id,
                        'image_path': image_path,
                        'caption': caption
                    })
        
        print(f"Loaded {len(self.examples)} examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一项
        
        Args:
            idx: 索引
            
        Returns:
            item: 处理后的数据项
        """
        example = self.examples[idx]
        image_path = example['image_path']
        caption = example['caption']
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 设置提示
            prompt = "请为这张图片生成描述："
            
            # 准备输入
            inputs = prepare_training_inputs(
                text=prompt,
                caption=caption,
                image=image,
                processor=self.processor,
                max_length=self.max_length
            )
            
            return inputs
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # 返回数据集中的下一项（如果有的话）
            if idx + 1 < len(self):
                return self[idx + 1]
            else:
                # 如果没有更多项，返回第一项
                return self[0]

class COCOEvalDataset(Dataset):
    """
    COCO评估数据集
    """
    def __init__(self, image_folder, annotation_file, processor):
        """
        初始化COCO评估数据集
        
        Args:
            image_folder: 图像文件夹路径
            annotation_file: 标注文件路径
            processor: 处理器
        """
        self.image_folder = image_folder
        self.processor = processor
        
        # 加载注释
        print(f"Loading annotations from {annotation_file}...")
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 创建图像ID到文件名的映射
        self.image_id_to_filename = {}
        for img in data['images']:
            self.image_id_to_filename[img['id']] = img['file_name']
        
        # 整理标注数据
        self.examples = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            caption = ann['caption']
            
            if image_id in self.image_id_to_filename:
                file_name = self.image_id_to_filename[image_id]
                image_path = os.path.join(image_folder, file_name)
                
                # 只添加存在的图像
                if os.path.exists(image_path):
                    if image_id not in self.examples:
                        self.examples[image_id] = {
                            'image_id': image_id,
                            'image_path': image_path,
                            'captions': []
                        }
                    self.examples[image_id]['captions'].append(caption)
        
        # 转换为列表
        self.examples = list(self.examples.values())
        print(f"Loaded {len(self.examples)} examples for evaluation")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一项
        
        Args:
            idx: 索引
            
        Returns:
            item: 处理后的数据项
        """
        example = self.examples[idx]
        image_path = example['image_path']
        captions = example['captions']
        image_id = example['image_id']
        
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 处理图像
            pixel_values = self.processor.image_processor(image, return_tensors="pt")["pixel_values"][0]
            
            return {
                'image_id': image_id,
                'pixel_values': pixel_values,
                'image': image,
                'reference_captions': captions
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # 返回数据集中的下一项（如果有的话）
            if idx + 1 < len(self):
                return self[idx + 1]
            else:
                # 如果没有更多项，返回第一项
                return self[0] 