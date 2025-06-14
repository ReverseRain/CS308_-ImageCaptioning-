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
            prompt = self.image_token + " Caption: "
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
            
            # Create input embeddings - similar to training mode
            # Get the language model embeddings for the prompt
            prompt_embeds = self.language_model.get_input_embeddings()(prompt_ids.squeeze(0))
            
            # Concatenate with the projected vision features for batch[0]
            full_embeds = torch.cat([
                prompt_embeds[:1],  # [IMAGE] token embedding
                projected_features[0],  # Vision features for first image
                prompt_embeds[1:],  # Caption: token embeddings
            ], dim=0)
            
            # Create attention mask
            attention_mask = torch.ones(full_embeds.size(0), dtype=torch.bool, device=self.device)
            
            # Forward pass through language model with inputs_embeds and improved generation parameters
            # 修改生成参数，使用组束搜索
            outputs = self.language_model.generate(
                inputs_embeds=full_embeds.unsqueeze(0),
                attention_mask=attention_mask.unsqueeze(0),
                max_new_tokens=50,
                num_beams=6,  # 修改为6，确保能被num_beam_groups=2整除
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                diversity_penalty=1.5,
                num_beam_groups=2,
                do_sample=False,
                early_stopping=True,
                use_cache=True
                )
            
            return outputs
        
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