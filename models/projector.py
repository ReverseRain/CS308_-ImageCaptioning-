import torch
import torch.nn as nn
import re


class MLPProjector(nn.Module):
    """
    简单MLP连接器，用于连接视觉特征和语言模型
    """
    def __init__(self, vision_hidden_size, text_hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(vision_hidden_size, text_hidden_size)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(text_hidden_size, text_hidden_size)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


def build_projector(config):
    """
    根据配置构建连接器
    
    Args:
        config: 配置参数，包含projector_type, vision_hidden_size, text_hidden_size
    
    Returns:
        projector: 构建好的连接器模块
    """
    projector_type = getattr(config, "projector_type", "mlp")
    vision_hidden_size = config.vision_hidden_size
    text_hidden_size = config.text_hidden_size
    
    if projector_type == "mlp":
        return MLPProjector(vision_hidden_size, text_hidden_size)
    elif projector_type == "linear":
        return nn.Linear(vision_hidden_size, text_hidden_size)
    elif projector_type == "identity":
        assert vision_hidden_size == text_hidden_size, "identity映射要求视觉和文本隐藏层大小相同"
        return nn.Identity()
    else:
        raise ValueError(f"未知的连接器类型: {projector_type}") 