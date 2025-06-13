import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vision_encoder import VisionEncoder
from .projector import build_projector
from .config import ModelConfig

# 定义常量
IGNORE_INDEX = -100  # 用于忽略损失计算中的某些位置

class ImageCaptioningModel(nn.Module):
    """
    视觉-语言模型，用于图像标注
    基于LLaVA架构但更加简化
    """
    def __init__(self, config=None):
        super().__init__()
        
        if config is None:
            # 默认配置
            config = ModelConfig(
                vision_model_name="openai/clip-vit-base-patch16",
                language_model_name="Qwen/Qwen3-0.6B",  # 已替换为 Qwen3-0.6B
                vision_select_layer=-1,
                projector_type="mlp"
            )
        
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载视觉编码器
        self.vision_encoder = VisionEncoder(
            vision_model_name=config.vision_model_name,
            select_layer=config.vision_select_layer
        ).to(self.device)
        
        # 加载语言模型
        self.language_model = AutoModelForCausalLM.from_pretrained(
            config.language_model_name
        ).to(self.device)
        
        # 获取模型隐藏层大小
        vision_hidden_size = self.vision_encoder.hidden_size
        text_hidden_size = self.language_model.config.hidden_size
        
        # 更新配置中的隐藏层大小
        self.config.update_sizes(vision_hidden_size, text_hidden_size)
        
        # 构建连接器
        self.projector = build_projector(self.config).to(self.device)
        
        # 加载tokenizer - 检查是否有单独的tokenizer路径
        tokenizer_path = getattr(config, "tokenizer_path", None)
        if tokenizer_path and os.path.exists(tokenizer_path):
            # 如果有tokenizer_path且目录存在，从该路径加载
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            # 否则从语言模型路径加载
            self.tokenizer = AutoTokenizer.from_pretrained(config.language_model_name)
        
        # 添加图像标记
        self.image_token = config.image_token
        if self.image_token not in self.tokenizer.get_vocab():
            self.image_token_id = len(self.tokenizer)
            self.tokenizer.add_tokens([self.image_token])
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        
        Args:
            pixel_values: 输入图像的像素值 [batch_size, 3, height, width]
            input_ids: 输入文本的token ids [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            labels: 用于计算损失的标签 [batch_size, seq_len]
            
        Returns:
            outputs: 语言模型的输出
        """
        # 将输入移到设备
        pixel_values = pixel_values.to(self.device)
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        # 处理图像
        image_features = self.vision_encoder(pixel_values)  # [batch_size, num_patches, vision_hidden_size]
        
        # 投影到文本空间
        projected_features = self.projector(image_features)  # [batch_size, num_patches, text_hidden_size]
        
        # 准备文本嵌入
        batch_size = input_ids.shape[0]
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 找到<image>token的位置
        for batch_idx in range(batch_size):
            image_positions = torch.where(input_ids[batch_idx] == self.image_token_id)[0]
            for pos in image_positions:
                # 替换<image>位置的嵌入为图像特征的平均值
                # 对于简化版本，我们直接使用所有图像patch的平均特征
                text_embeds[batch_idx, pos] = projected_features[batch_idx].mean(dim=0)
        
        # 通过语言模型
        outputs = self.language_model(
            inputs_embeds=text_embeds, 
            attention_mask=attention_mask, 
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate_caption(self, image, prompt="请为这张图片生成描述：", max_length=50):
        """
        为图像生成描述
        
        Args:
            image: PIL图像或已处理的图像张量
            prompt: 提示文本
            max_length: 生成的最大长度
            
        Returns:
            caption: 生成的图像描述
        """
        # 预处理图像
        if not isinstance(image, torch.Tensor):
            # 如果输入是PIL图像，进行处理
            pixel_values = self.vision_encoder.process_images(image)
        else:
            pixel_values = image
            
        pixel_values = pixel_values.to(self.device)
        
        # 预处理文本
        prompt_with_image = f"{prompt} {self.image_token}"
        inputs = self.tokenizer(prompt_with_image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # 处理图像
        image_features = self.vision_encoder(pixel_values)
        projected_features = self.projector(image_features)
        
        # 构建输入嵌入
        text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # 替换图像标记
        image_positions = torch.where(input_ids == self.image_token_id)[0]
        for pos in image_positions:
            text_embeds[0, pos] = projected_features[0].mean(dim=0)
        
        # 确保tokenizer有pad_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 生成文本
        with torch.no_grad():
            outputs = self.language_model.generate(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                max_length=max_length,  # 只使用max_length参数
                num_beams=5,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 移除提示部分
        if prompt in caption:
            caption = caption[len(prompt):].strip()
        
        return caption

    def save_pretrained(self, save_dir):
        """
        保存预训练模型
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存配置
        import json
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # 保存视觉编码器
        vision_encoder_dir = os.path.join(save_dir, "vision_encoder")
        os.makedirs(vision_encoder_dir, exist_ok=True)
        self.vision_encoder.vision_model.save_pretrained(vision_encoder_dir)
        
        # 保存图像处理器
        self.vision_encoder.image_processor.save_pretrained(vision_encoder_dir)
        
        # 保存语言模型
        self.language_model.save_pretrained(os.path.join(save_dir, "language_model"))
        
        # 保存投影器
        torch.save(self.projector.state_dict(), os.path.join(save_dir, "projector.pt"))
        
        # 保存tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_dir, "tokenizer"))
        
    @classmethod
    def from_pretrained(cls, model_path, device=None):
        """
        从预训练模型加载
        
        Args:
            model_path: 模型路径
            device: 设备
            
        Returns:
            model: 加载的模型
        """
        import json
        
        # 加载配置
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # 创建配置
        config = ModelConfig.from_dict(config_dict)
        
        # 更新模型路径
        config.vision_model_name = os.path.join(model_path, "vision_encoder")
        config.language_model_name = os.path.join(model_path, "language_model")
        config.tokenizer_path = os.path.join(model_path, "tokenizer")
        
        # 创建模型
        model = cls(config)
        
        # 加载投影器
        projector_path = os.path.join(model_path, "projector.pt")
        if os.path.exists(projector_path):
            model.projector.load_state_dict(torch.load(projector_path, map_location=model.device))
        
        return model 