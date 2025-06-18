import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModelForCausalLM, AutoTokenizer


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.view(features.size(0), -1)  # 保证是[batch, features]
        features = self.linear(features)
        features = self.bn(features)
        return features


class QwenDecoder(nn.Module):
    def __init__(self, embed_size, model_path="pretrained_models/Qwen3-0.6B"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 改回FP32
            device_map="auto"
        )
        # 启用梯度检查点
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False  # 禁用KV缓存以配合梯度检查点
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.projection = nn.Linear(embed_size, self.model.config.hidden_size)
        
    def forward(self, features, captions=None):
        # 将图像特征投影到Qwen的隐藏维度
        projected_features = self.projection(features)
        
        if self.training:
            # 训练模式
            assert captions is not None
            # 将caption转换为token ids
            inputs = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(features.device)
            attention_mask = inputs.attention_mask.to(features.device)
            
            # 将图像特征和文本嵌入拼接
            text_embeds = self.model.get_input_embeddings()(input_ids)
            combined_embeds = torch.cat([projected_features.unsqueeze(1), text_embeds], dim=1)
            
            # 更新attention mask以包含图像特征
            image_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)
            combined_mask = torch.cat([image_mask, attention_mask], dim=1)
            
            # 准备标签：在开头添加-100以对应图像特征位置
            label_pad = torch.full((input_ids.shape[0], 1), -100, device=input_ids.device)
            labels = torch.cat([label_pad, input_ids], dim=1)
            
            # 使用拼接后的嵌入进行前向传播
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                labels=labels  # 使用修改后的标签
            )
            return outputs
        else:
            # 推理模式
            batch_size = features.size(0)
            # 使用图像特征生成文本
            generated_ids = self.model.generate(
                inputs_embeds=projected_features.unsqueeze(1),
                max_length=50,
                num_beams=4,
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            return generated_ids


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

