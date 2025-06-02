import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPImageProcessor, AutoTokenizer

class CombinedProcessor:
    """
    组合图像处理器和文本处理器
    """
    def __init__(self, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __call__(self, text=None, images=None, **kwargs):
        """
        处理文本和图像输入
        
        Args:
            text: 文本输入
            images: 图像输入
            kwargs: 其他参数
            
        Returns:
            处理后的输入
        """
        if images is not None and text is None:
            return self.image_processor(images=images, **kwargs)
        elif text is not None and images is None:
            return self.tokenizer(text, **kwargs)
        elif text is not None and images is not None:
            text_inputs = self.tokenizer(text, **kwargs)
            vision_inputs = self.image_processor(images=images, **kwargs)
            
            text_inputs["pixel_values"] = vision_inputs["pixel_values"]
            return text_inputs
        else:
            raise ValueError("Either text or images or both must be provided")

def get_processors(vision_model_name="openai/clip-vit-base-patch16", 
                  language_model_name="Qwen/Qwen1.5-0.5B"):
    """
    获取处理器
    
    Args:
        vision_model_name: 视觉模型名称
        language_model_name: 语言模型名称
        
    Returns:
        processor: 组合处理器
    """
    tokenizer = AutoTokenizer.from_pretrained(language_model_name)
    image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
    
    # 添加特殊标记
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"])
    
    return CombinedProcessor(tokenizer, image_processor)

def prepare_inputs(text, image, processor, max_length=512):
    """
    准备输入
    
    Args:
        text: 输入文本
        image: 输入图像
        processor: 处理器
        max_length: 最大长度
        
    Returns:
        inputs: 模型输入
    """
    inputs = processor(
        text=text,
        images=image,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return inputs

def prepare_training_inputs(text, caption, image, processor, max_length=512):
    """
    准备训练输入
    
    Args:
        text: 提示文本
        caption: 图像描述
        image: 输入图像
        processor: 处理器
        max_length: 最大长度
        
    Returns:
        inputs: 模型输入，包含像素值、input_ids、attention_mask和labels
    """
    # 构建完整文本
    full_text = f"{text} <image> {caption}"
    
    # 处理输入
    inputs = processor(
        text=full_text,
        images=image,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 创建labels（复制input_ids）
    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]
    labels = input_ids.clone()
    
    # 计算提示部分的长度
    prompt_len = len(processor.tokenizer(f"{text} <image>", return_tensors="pt")["input_ids"][0])
    
    # 将提示部分的labels设为-100（忽略损失计算）
    labels[:prompt_len] = -100
    
    return {
        "pixel_values": inputs["pixel_values"][0],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    } 