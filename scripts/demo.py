import os
import sys
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录到路径，方便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ImageCaptioningModel

def parse_args():
    parser = argparse.ArgumentParser(description="图像描述生成演示")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="图像路径")
    parser.add_argument("--prompt", type=str, default="请为这张图片生成描述：", help="生成提示")
    parser.add_argument("--max_length", type=int, default=50, help="生成的最大长度")
    parser.add_argument("--output_path", type=str, default=None, help="输出图像路径（可选）")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    model = ImageCaptioningModel.from_pretrained(args.model_path)
    model.eval()
    
    # 加载图像
    image = Image.open(args.image_path).convert("RGB")
    
    # 生成描述
    print("生成描述...")
    with torch.no_grad():
        caption = model.generate_caption(
            image=image,
            prompt=args.prompt,
            max_length=args.max_length
        )
    
    # 打印结果
    print(f"生成的描述: {caption}")
    
    # 显示结果
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(caption, fontsize=12)
    plt.axis('off')
    
    # 保存结果
    if args.output_path:
        plt.savefig(args.output_path, bbox_inches='tight', pad_inches=0.1)
        print(f"结果已保存到: {args.output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    main() 