"""
Image captioning model combining vision encoder, projector and language model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vision_encoder import VisionEncoder
from .projector import MLPProjector
from peft import LoraConfig, get_peft_model


class ImageCaptioningModel(nn.Module):
    """
    Image captioning model combining vision encoder, projector and language model
    """
    
    def __init__(
        self,
        vision_encoder_path,
        language_model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(vision_encoder_path)

        # Language model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_path)

        vision_dim = self.vision_encoder.hidden_size
        llm_dim = self.language_model.config.hidden_size
        
        self.projector = self.build_connector(vision_dim, llm_dim)
        # Special tokens for image representation
        self.image_token = "[IMAGE]"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        
        # Move to device
        self.device = device
        self.to(device)
    
    def build_connector(self, in_dim, out_dim):
        """构建视觉-语言连接器"""
        
        return nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.ReLU(),
            nn.Linear(out_dim * 2, out_dim)
        )
        
    
    def forward(self, image, captions=None):
        # 视觉特征提取
        vis_features = self.vision_encoder(image)  # 假设返回形状为 [batch_size, num_tokens, feat_dim]

        # 文本处理
        tokenized_captions = self.tokenizer(captions, padding="longest", truncation=True, return_tensors="pt").to(self.device)
        input_ids = tokenized_captions.input_ids
        attention_mask = tokenized_captions.attention_mask

        # 对视觉特征进行投影，得到与语言模型嵌入维度匹配的特征
        projected_vis = self.projector(vis_features)  # [batch_size, num_tokens, hidden_size]

        # 获取文本嵌入
        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 拼接图像特征和文本嵌入
        inputs_embeds = torch.cat([projected_vis, text_embeds], dim=1)

        # 构造新的 attention_mask，对应图像特征部分全为 1（有效），文本部分保持原样
        new_attention_mask = torch.ones(
            (attention_mask.shape[0], projected_vis.shape[1] + attention_mask.shape[1]),
            device=attention_mask.device
        )
        new_attention_mask[:, projected_vis.shape[1]:] = attention_mask

        # 扩展标签：在开头添加对应图像特征数量的 -100 列
        labels = torch.full(
            (input_ids.shape[0], input_ids.shape[1] + projected_vis.shape[1]),
            -100,
            device=input_ids.device
        )
        labels[:, projected_vis.shape[1]:] = input_ids

        # 输入语言模型
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=labels,
            return_dict=True
        )

        return outputs