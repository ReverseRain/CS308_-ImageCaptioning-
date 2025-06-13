import os
import torch
import sys
import argparse
from transformers import CLIPImageProcessor, AutoTokenizer

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import ImageCaptioningModel
from models.config import ModelConfig

def parse_args():
    parser = argparse.ArgumentParser(description="修复模型的预处理器和tokenizer问题")
    parser.add_argument("--model_path", type=str, required=True, help="已训练模型的路径")
    parser.add_argument("--output_path", type=str, default=None, help="修复后模型的保存路径（默认覆盖原模型）")
    parser.add_argument("--vision_model", type=str, default="openai/clip-vit-base-patch16", help="原始视觉模型名称")
    parser.add_argument("--language_model", type=str, default="Qwen/Qwen1.5-0.5B", help="原始语言模型名称")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 如果未指定输出路径，则默认覆盖原模型
    output_path = args.output_path if args.output_path else args.model_path
    
    print(f"正在修复模型: {args.model_path}")
    print(f"将保存到: {output_path}")
    
    # 从原始预训练模型加载图像处理器
    print(f"从 {args.vision_model} 加载图像处理器")
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)
    
    # 保存到模型的视觉编码器目录
    vision_encoder_dir = os.path.join(output_path, "vision_encoder")
    os.makedirs(vision_encoder_dir, exist_ok=True)
    print(f"保存图像处理器到 {vision_encoder_dir}")
    image_processor.save_pretrained(vision_encoder_dir)
    
    # 从原始预训练模型加载tokenizer
    print(f"从 {args.language_model} 加载tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    
    # 保存到模型的tokenizer目录
    tokenizer_dir = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    print(f"保存tokenizer到 {tokenizer_dir}")
    tokenizer.save_pretrained(tokenizer_dir)
    
    print("修复完成！")
    print(f"现在可以使用以下命令测试模型:")
    print(f'python scripts/demo.py --model_path "{output_path}" --image_path "您的图片路径" --prompt "请为这张图片生成描述："')

if __name__ == "__main__":
    main()