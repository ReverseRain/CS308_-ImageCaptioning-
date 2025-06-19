import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, vit_b_16
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModel


class CNNEncoder(nn.Module):
    def __init__(self, embed_size,model_path):
        super().__init__()
        # 使用预训练的Vision Transformer模型
        self.vit = AutoModel.from_pretrained(model_path)
        # 移除分类头
        self.feature_dim = self.vit.config.hidden_size
        # self.vit.heads = nn.Identity()
        # 特征映射层
        self.linear = nn.Linear(self.feature_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.vit(images).last_hidden_state[-1]   # [batch, feature_dim]
        
        features = self.linear(features)
        
        features = self.bn(features)
        return features


class MLPConnector(nn.Module):
    def __init__(self, vision_embed_size, language_embed_size, hidden_size=None):
        super().__init__()
        if hidden_size is None:
            hidden_size = language_embed_size * 2
            
        # 简单的两层MLP作为连接器
        self.mlp = nn.Sequential(
            nn.Linear(vision_embed_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, language_embed_size)
        )
    
    def forward(self, features):
        # 输入视觉特征，输出语言模型的嵌入
        return self.mlp(features)


class VLMModel(nn.Module):
    def __init__(self, vision_embed_size, language_model_name="Qwen/Qwen3-0.6B",vision_model_path="swinTransformer"):
        super().__init__()
        # 视觉编码器
        self.encoder = CNNEncoder(vision_embed_size,vision_model_path)
        
        # 加载预训练的Qwen3-0.6B语言模型
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_name)
        language_embed_size = self.language_model.config.hidden_size
        
        # MLP连接器
        self.connector = MLPConnector(vision_embed_size, language_embed_size)
        
        # 将连接器输出投影到语言模型的词嵌入空间
        self.to_vocab = nn.Linear(language_embed_size, self.language_model.config.vocab_size)
        
        # 添加视觉条件投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(language_embed_size, language_embed_size),
            nn.Tanh()
        )
        
    def forward(self, images, text_input_ids=None):
        # 从图像中提取特征
        vision_features = self.encoder(images)  # [batch_size, vision_embed_size]
        
        # 通过MLP连接器转换特征
        vision_language_features = self.connector(vision_features)  # [batch_size, language_embed_size]
        
        if text_input_ids is not None:
            # 训练模式：将视觉特征与文本特征结合，预测下一个token
            batch_size, seq_length = text_input_ids.shape
            
            # 使用语言模型获取输入文本的嵌入
            with torch.no_grad():  # 冻结语言模型参数
                text_outputs = self.language_model(input_ids=text_input_ids, 
                                                  output_hidden_states=True)
                text_features = text_outputs.hidden_states[-1]  # [batch_size, seq_length, language_embed_size]
            
            # 准备视觉特征
            vision_projection = self.visual_projection(vision_language_features)  # [batch_size, language_embed_size]
            
            # 对于每个样本，将视觉特征添加到每个文本特征中
            # 扩展视觉特征以匹配文本特征的形状
            vision_projection = vision_projection.unsqueeze(0)  # [batch_size, 1, language_embed_size]
            
            # 将视觉特征添加到每个位置的文本特征
            # 此处使用加法而非拼接，保持维度不变
            fused_features = text_features + vision_projection  # [batch_size, seq_length, language_embed_size]
            
            # 生成对词汇表的logits
            logits = self.to_vocab(fused_features)  # [batch_size, seq_length, vocab_size]
            
            return logits
        else:
            # 推理模式：仅从视觉特征生成文本
            # 将视觉特征转换为对词汇表的logits
            logits = self.to_vocab(vision_language_features.unsqueeze(1))  # [batch_size, 1, vocab_size]
            return logits