import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(nn.Module):
    """
    视觉编码器，使用transformer-based CLIP模型
    """
    def __init__(self, vision_model_name="openai/clip-vit-base-patch16", select_layer=-1):
        super().__init__()
        
        self.vision_model_name = vision_model_name
        self.select_layer = select_layer
        self.is_loaded = False
        
        # 直接加载模型
        self.load_model()
    
    def load_model(self):
        if self.is_loaded:
            print(f'{self.vision_model_name} 已加载，跳过。')
            return
            
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(self.vision_model_name)
        self.vision_model.requires_grad_(False)  # 冻结视觉编码器参数
        
        self.is_loaded = True
    
    def feature_select(self, image_forward_outs):
        """选择特定层的特征"""
        if hasattr(image_forward_outs, 'hidden_states'):
            # 如果模型输出了隐藏状态，选择指定层
            image_features = image_forward_outs.hidden_states[self.select_layer]
            # 只选择patch特征，忽略[CLS]标记
            return image_features[:, 1:]
        else:
            # 否则直接使用最后一层的输出
            return image_forward_outs.last_hidden_state[:, 1:]
    
    def forward(self, pixel_values):
        """
        前向传播，处理图像并提取特征
        
        Args:
            pixel_values: 输入图像的像素值 [batch_size, 3, height, width]
            
        Returns:
            image_features: 图像特征
        """
        # 确保模型已加载
        if not self.is_loaded:
            self.load_model()
            
        # 处理图像获取特征
        image_forward_outs = self.vision_model(
            pixel_values, 
            output_hidden_states=True
        )
        
        # 选择特定层的特征
        image_features = self.feature_select(image_forward_outs)
        
        return image_features
    
    @property
    def hidden_size(self):
        """获取模型隐藏层大小"""
        return self.vision_model.config.hidden_size
    
    def process_images(self, images):
        """处理图像，转换为模型输入格式"""
        if isinstance(images, list):
            # 处理图像列表
            processed = self.image_processor(images, return_tensors="pt")
        else:
            # 处理单个图像
            processed = self.image_processor(images, return_tensors="pt")
        
        return processed.pixel_values 