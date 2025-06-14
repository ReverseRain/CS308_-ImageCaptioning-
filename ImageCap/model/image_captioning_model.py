"""
Image captioning model combining vision encoder, projector and language model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .vision_encoder import VisionEncoder
from .projector import MLPProjector


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
        
        # Multi-modal projector
        self.projector = MLPProjector(
            vision_hidden_size=self.vision_encoder.hidden_size,
            language_hidden_size=self.language_model.config.hidden_size
        )
        
        # Special tokens for image representation
        self.image_token = "[IMAGE]"
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        
        # Move to device
        self.device = device
        self.to(device)
    
    def forward(self, images, captions=None):
        """
        Args:
            images: Tensor of images, shape [batch_size, channels, height, width]
            captions: Optional tensor of caption token ids, shape [batch_size, seq_len]
        Returns:
            outputs: CausalLM outputs with loss if captions is provided
        """
        batch_size = images.shape[0]
        
        # Get vision features
        vision_features = self.vision_encoder(images)  # [batch_size, num_patches, vision_hidden_size]
        
        # Project vision features to language model space
        projected_features = self.projector(vision_features)  # [batch_size, num_patches, language_hidden_size]
        
        if captions is None:
            # Generation mode (inference)
            # 确保使用与训练相同的输入格式
            # 准备提示词前缀
            prompt = self.image_token + " Caption: "
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
            # 获取提示词的嵌入
            prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids[0])
            
            # 构建输入嵌入，模拟训练时的格式
            input_embeds = []
            
            # 对每个批次样本构建嵌入
            for b in range(batch_size):
                # 拼接：[IMAGE]标记嵌入 + 视觉特征 + "Caption: "的嵌入
                full_embeds = torch.cat([
                    prompt_embeds[:1],  # [IMAGE]标记嵌入
                    projected_features[b],  # 图像视觉特征
                    prompt_embeds[1:],  # "Caption: "的嵌入
                ], dim=0)
                
                input_embeds.append(full_embeds)
            
            # 转换为批次张量
            input_embeds = torch.stack(input_embeds)
            
            # 创建注意力掩码
            attention_mask = torch.ones(
                (batch_size, input_embeds.shape[1]), 
                dtype=torch.long, 
                device=self.device
            )
            
            # 使用transformers的generate方法生成
            gen_tokens = self.language_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=30,
                num_beams=3,
                repetition_penalty=1.5,
                no_repeat_ngram_size=2,
                temperature=0.9,
                do_sample=True,
                top_p=0.9,
                early_stopping=True
            )
            
            # 返回生成的token
            return gen_tokens
        
        # Training mode
        # Prepare inputs: [IMAGE] Caption: {caption}
        prompt = self.image_token + " Caption: "
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
        # Tokenize captions
        tokenized_captions = self.tokenizer(captions, padding="longest", truncation=True, return_tensors="pt").to(self.device)
        caption_input_ids = tokenized_captions.input_ids
        caption_attention_mask = tokenized_captions.attention_mask
        
        # Create inputs for language model
        batch_size = images.shape[0]
        input_embeds = []
        
        for b in range(batch_size):
            # Get the language model embeddings for the prompt
            prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids.squeeze(0))
            
            # Concatenate with the projected vision features
            full_embeds = torch.cat([
                prompt_embeds[:1],  # [IMAGE] token embedding
                projected_features[b],  # Vision features
                prompt_embeds[1:],  # Caption: token embeddings
                self.language_model.get_input_embeddings()(caption_input_ids[b])  # Caption embeddings
            ], dim=0)
            
            input_embeds.append(full_embeds)
            
        # Pad input embeddings to longest in batch
        max_len = max([embed.shape[0] for embed in input_embeds])
        padded_input_embeds = []
        attention_mask = []
        
        for embed in input_embeds:
            attention = torch.ones(max_len, dtype=torch.bool, device=self.device)
            if embed.shape[0] < max_len:
                padding_len = max_len - embed.shape[0]
                padding = torch.zeros(padding_len, embed.shape[1], device=embed.device)
                padded_embed = torch.cat([embed, padding], dim=0)
                attention[embed.shape[0]:] = 0
            else:
                padded_embed = embed
            
            padded_input_embeds.append(padded_embed)
            attention_mask.append(attention)
            
        input_embeds = torch.stack(padded_input_embeds)
        attention_mask = torch.stack(attention_mask)
        
        # Create labels - shift right by one position
        labels = torch.full_like(attention_mask, -100, dtype=torch.long)  # -100 is ignored in CrossEntropyLoss
        for b in range(batch_size):
            caption_len = caption_attention_mask[b].sum()
            offset = padded_input_embeds[b].shape[0] - caption_len
            labels[b, offset:offset+caption_len-1] = caption_input_ids[b, 1:caption_len]
        
        # Forward pass through language model with inputs_embeds instead of input_ids
        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs 