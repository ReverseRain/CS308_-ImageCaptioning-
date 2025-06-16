from transformers import AutoTokenizer
import torch


class Tokenizer:
    def __init__(self, captions=None, max_len=20):
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

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
