"""
Image captioning model combining vision encoder, projector and language model
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer,BertConfig,BertLMHeadModel

from .vision_encoder import VisionEncoder
from .projector import MLPProjector
from peft import LoraConfig, get_peft_model


class QFormerConnector(nn.Module):
    def __init__(self, in_dim, out_dim, num_query_tokens=32, cross_attn_freq=2):
        super().__init__()
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = in_dim 
        encoder_config.is_decoder = True
        encoder_config.add_cross_attention = True
        encoder_config.output_hidden_states = True
        encoder_config.cross_attention_freq = cross_attn_freq
        encoder_config.query_length = num_query_tokens
        
        self.qformer = BertLMHeadModel(config=encoder_config)
        
        # 可学习的查询向量（信息提取的核心）[1,6](@ref)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, encoder_config.hidden_size)
        )
        nn.init.normal_(self.query_tokens, std=encoder_config.initializer_range)
        
        # 输出投影层（对齐LLM输入维度）[1,2](@ref)
        self.proj = nn.Linear(encoder_config.hidden_size, out_dim)
        self.num_query_tokens = num_query_tokens

    def forward(self, image_embeds):
        """
        image_embeds: [batch, seq_len, in_dim] 视觉特征
        返回: [batch, num_query_tokens, out_dim] 适配语言模型的视觉表示
        """
        batch_size = image_embeds.size(0)
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        
        outputs = self.qformer(
            inputs_embeds=query_embeds,
            encoder_hidden_states=image_embeds,
            return_dict=True
        )
        
        # print("Output keys:", list(outputs.keys()))
        last_hidden = outputs.hidden_states[-1]  # [batch, num_query, hidden]
        return self.proj(last_hidden)  # [batch, num_query, out_dim]


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
        # for param in self.vision_encoder.parameters():
        #     param.requires_grad = False 
        # vision_lora_config = LoraConfig(
        #     r=8,                    # 秩维度（推荐4-16）[6,7](@ref)
        #     lora_alpha=32,           # 缩放因子（通常设为2*r）
        #     target_modules=["model.encoder.layers.*.blocks.*.attention.self.query",
        # "model.encoder.layers.*.blocks.*.attention.self.key"],  # 注意力层关键模块[3,7](@ref)
        #     lora_dropout=0.05,       # 防过拟合
        #     bias="none",             # 不更新偏置
        #     task_type="FEATURE_EXTRACTION"
        # )
        # self.vision_encoder = get_peft_model(self.vision_encoder, vision_lora_config)

        # Language model
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model = AutoModelForCausalLM.from_pretrained(language_model_path)
        # for param in self.language_model.parameters():
        #     param.requires_grad = False 
        # text_lora_config = LoraConfig(
        #     r=16,                   # 文本任务需更高秩（推荐8-32）[5,9](@ref)
        #     lora_alpha=64,
        #     target_modules=["q_proj", "k_proj"],  # 注意力+MLP层[2,11](@ref)
        #     lora_dropout=0.1, 
        #     task_type="CAUSAL_LM"
        # )
        # print(self.language_model)
        # self.language_model = get_peft_model(self.language_model, text_lora_config)
        
        # Multi-modal projector
        # self.projector = MLPProjector(
        #     vision_hidden_size=self.vision_encoder.hidden_size,
        #     language_hidden_size=self.language_model.config.hidden_size
        # )
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
        
        # return nn.Sequential(
        #     nn.Linear(in_dim, out_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(out_dim * 2, out_dim)
        # )
        return QFormerConnector(in_dim,out_dim)
    
    # def forward(self, images, captions=None):
    #     """
    #     Args:
    #         images: Tensor of images, shape [batch_size, channels, height, width]
    #         captions: Optional tensor of caption token ids, shape [batch_size, seq_len]
    #     Returns:
    #         outputs: CausalLM outputs with loss if captions is provided
    #     """
    #     batch_size = images.shape[0]
        
    #     # Get vision features
    #     vision_features = self.vision_encoder(images)  # [batch_size, num_patches, vision_hidden_size]
        
    #     # Project vision features to language model space
    #     projected_features = self.projector(vision_features)  # [batch_size, num_patches, language_hidden_size]
        
    #     if captions is None:
    #         # Generation mode (inference)
    #         # 确保使用与训练相同的输入格式
    #         # 准备提示词前缀
    #         prompt = self.image_token + " Caption: "
    #         prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
    #         # 获取提示词的嵌入
    #         prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids[0])
            
    #         # 构建输入嵌入，模拟训练时的格式
    #         input_embeds = []
            
    #         # 对每个批次样本构建嵌入
    #         for b in range(batch_size):
    #             # 拼接：[IMAGE]标记嵌入 + 视觉特征 + "Caption: "的嵌入
    #             full_embeds = torch.cat([
    #                 prompt_embeds[:1],  # [IMAGE]标记嵌入
    #                 projected_features[b],  # 图像视觉特征
    #                 prompt_embeds[1:],  # "Caption: "的嵌入
    #             ], dim=0)
                
    #             input_embeds.append(full_embeds)
            
    #         # 转换为批次张量
    #         input_embeds = torch.stack(input_embeds)
            
    #         # 创建注意力掩码
    #         attention_mask = torch.ones(
    #             (batch_size, input_embeds.shape[1]), 
    #             dtype=torch.long, 
    #             device=self.device
    #         )
            
    #         # 使用transformers的generate方法生成
    #         gen_tokens = self.language_model.generate(
    #             inputs_embeds=input_embeds,
    #             attention_mask=attention_mask,
    #             max_new_tokens=30,
    #             num_beams=3,
    #             repetition_penalty=1.5,
    #             no_repeat_ngram_size=2,
    #             temperature=0.9,
    #             do_sample=True,
    #             top_p=0.9,
    #             early_stopping=True
    #         )
            
    #         # 返回生成的token
    #         return gen_tokens
        
    #     # Training mode
    #     # Prepare inputs: [IMAGE] Caption: {caption}
    #     prompt = self.image_token + " Caption: "
    #     prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        
    #     # Tokenize captions
    #     tokenized_captions = self.tokenizer(captions, padding="longest", truncation=True, return_tensors="pt").to(self.device)
    #     caption_input_ids = tokenized_captions.input_ids
    #     caption_attention_mask = tokenized_captions.attention_mask
        
    #     # Create inputs for language model
    #     batch_size = images.shape[0]
    #     input_embeds = []
        
    #     for b in range(batch_size):
    #         # Get the language model embeddings for the prompt
    #         prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids.squeeze(0))
            
    #         # Concatenate with the projected vision features
    #         full_embeds = torch.cat([
    #             prompt_embeds[:1],  # [IMAGE] token embedding
    #             projected_features[b],  # Vision features
    #             prompt_embeds[1:],  # Caption: token embeddings
    #             self.language_model.get_input_embeddings()(caption_input_ids[b])  # Caption embeddings
    #         ], dim=0)
            
    #         input_embeds.append(full_embeds)
            
    #     # Pad input embeddings to longest in batch
    #     max_len = max([embed.shape[0] for embed in input_embeds])
    #     padded_input_embeds = []
    #     attention_mask = []
        
    #     for embed in input_embeds:
    #         attention = torch.ones(max_len, dtype=torch.bool, device=self.device)
    #         if embed.shape[0] < max_len:
    #             padding_len = max_len - embed.shape[0]
    #             padding = torch.zeros(padding_len, embed.shape[1], device=embed.device)
    #             padded_embed = torch.cat([embed, padding], dim=0)
    #             attention[embed.shape[0]:] = 0
    #         else:
    #             padded_embed = embed
            
    #         padded_input_embeds.append(padded_embed)
    #         attention_mask.append(attention)
            
    #     input_embeds = torch.stack(padded_input_embeds)
    #     attention_mask = torch.stack(attention_mask)
        
    #     # Create labels - shift right by one position
    #     labels = torch.full_like(attention_mask, -100, dtype=torch.long)  # -100 is ignored in CrossEntropyLoss
    #     for b in range(batch_size):
    #         caption_len = caption_attention_mask[b].sum()
    #         offset = padded_input_embeds[b].shape[0] - caption_len
    #         labels[b, offset:offset+caption_len-1] = caption_input_ids[b, 1:caption_len]
        
    #     # Forward pass through language model with inputs_embeds instead of input_ids
    #     outputs = self.language_model(
    #         inputs_embeds=input_embeds,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         return_dict=True
    #     )
        
    #     return outputs 


    # def forward(self, image, captions=None):
    #     """
    #     前向传播
        
    #     参数:
    #         images: 输入图像 (batch_size, C, H, W)
        
    #     返回:
    #         语言模型输出caption
    #     """
    #     vis_features = self.vision_encoder(image)
    #     print("shape ",vis_features.shape)
    #     vis_features = vis_features[:, 0,:] 
    #     print("shape ",vis_features.shape)
        
       
    #     mapped_vis = self.projector(vis_features)
    #     print("ajosjijai")

    #     tokenized_captions = self.tokenizer(captions, padding="longest", truncation=True, return_tensors="pt").to(self.device)
    #     input_ids = tokenized_captions.input_ids
    #     text_embeds = self.language_model.get_input_embeddings()(input_ids)
        
    #     inputs_embeds = torch.cat([mapped_vis.unsqueeze(1), text_embeds], dim=1)
    #     attention_mask = None
        
    #     outputs = self.language_model(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         labels=input_ids,
    #         return_dict=True
    #     )
        
    #     return outputs
    
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