import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenDecoder(nn.Module):
    def __init__(self, embed_size, model_path="pretrained_models/Qwen3-0.6B"):
        super().__init__()
        # 加载模型时使用更多的内存优化选项
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # 使用float32
            device_map="auto",
            low_cpu_mem_usage=True,
            offload_folder="offload",  # 设置模型权重卸载目录
            offload_state_dict=True,  # 启用状态字典卸载
        )
        # 启用梯度检查点以节省显存
        self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False  # 禁用KV缓存以配合梯度检查点
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            legacy=False,
            trust_remote_code=True
        )
        # 投影层使用float32
        self.projection = nn.Linear(embed_size, self.model.config.hidden_size)
        
    def forward(self, features, captions=None):
        # 确保输入是float32
        features = features.float()
        # 将图像特征投影到Qwen的隐藏维度
        projected_features = self.projection(features)
        
        if self.training:
            # 训练模式
            assert captions is not None
            # 将caption转换为token ids
            inputs = self.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128  # 限制最大长度
            )
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
                labels=labels,  # 使用修改后的标签
                use_cache=False  # 禁用KV缓存以节省显存
            )
            return outputs
        else:
            # 推理模式
            # 使用图像特征生成文本
            generated_ids = self.model.generate(
                inputs_embeds=projected_features.unsqueeze(1),
                max_length=30,  # 减小最大长度
                num_beams=2,    # 减小beam search的beam数
                no_repeat_ngram_size=3,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # 在推理时启用KV缓存以加速生成
            )
            return generated_ids

    def decode(self, token_ids):
        """解码token ids为文本"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True) 