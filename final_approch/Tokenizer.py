from transformers import AutoTokenizer
import torch


class Tokenizer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_len=20):
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 确保tokenizer有正确的特殊token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_len = max_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
        self.eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        # 为图像描述任务准备系统提示
        self.system_prompt = "你是一个图像描述助手。请为图像提供简短、准确的描述。"
        
        print(f"Special tokens: BOS={self.bos_token_id}, EOS={self.eos_token_id}, PAD={self.pad_token_id}")

    def prepare_caption_input(self, caption=None):
        """准备用于训练的带指令的文本格式"""
        if caption:
            # 训练模式：使用真实描述
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "请描述这张图片。"},
                {"role": "assistant", "content": caption}
            ]
        else:
            # 推理模式：只有用户请求
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": "请描述这张图片。"}
            ]
        
        return self.apply_chat_template(messages)

    def encode(self, text, max_len=None):
        max_len = max_len or self.max_len
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        return tokens.squeeze(0)

    def decode(self, tokens):
        tokens = tokens.tolist() if isinstance(tokens, torch.Tensor) else tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def batch_encode(self, captions, max_len=None, use_instruction=True):
        max_len = max_len or self.max_len
        
        # 使用带指令的格式准备文本
        if use_instruction:
            formatted_texts = [self.prepare_caption_input(caption) for caption in captions]
        else:
            formatted_texts = captions
            
        return self.tokenizer(
            formatted_texts,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    def apply_chat_template(self, messages, **kwargs):
        """使用Qwen3的聊天模板来格式化消息"""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
