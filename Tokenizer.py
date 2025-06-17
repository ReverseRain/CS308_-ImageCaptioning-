from transformers import AutoTokenizer
import torch


class Tokenizer:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", max_len=20):
        print(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_len = max_len
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = getattr(self.tokenizer, 'bos_token_id', None)
        self.eos_token_id = getattr(self.tokenizer, 'eos_token_id', None)

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

    def batch_encode(self, texts, max_len=None):
        max_len = max_len or self.max_len
        return self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=max_len,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

    def apply_chat_template(self, messages, **kwargs):
        """使用Qwen3的聊天模板来格式化消息"""
        return self.tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
