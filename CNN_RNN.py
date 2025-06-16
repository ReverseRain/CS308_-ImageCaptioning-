import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ViT_B_16_Weights, vit_b_16


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        # 使用预训练的Vision Transformer模型
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # 移除分类头
        self.feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        # 特征映射层
        self.linear = nn.Linear(self.feature_dim, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.vit(images)  # [batch, feature_dim]
        features = self.linear(features)
        features = self.bn(features)
        return features


class RNNDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs

