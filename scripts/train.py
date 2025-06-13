import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import sys
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import ImageCaptioningModel
from models.config import ModelConfig
from utils.data import CocoDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="训练图像描述生成模型")
    
    # 模型参数
    parser.add_argument("--vision_model", type=str, default="openai/clip-vit-base-patch16", help="视觉模型名称")
    parser.add_argument("--language_model", type=str, default="Qwen/Qwen3-0.6B", help="语言模型名称")
    parser.add_argument("--projector_type", type=str, default="mlp", help="连接器类型: mlp, linear, identity")
    
    # 数据集参数
    parser.add_argument("--data_dir", type=str, required=True, help="COCO数据集目录")
    parser.add_argument("--ann_file", type=str, required=True, help="COCO数据集注释文件")
    parser.add_argument("--batch_size", type=int, default=8, help="批量大小")
    parser.add_argument("--max_length", type=int, default=77, help="最大文本长度")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--max_samples", type=int, default=None, help="最大样本数，用于小规模测试")
    
    # 训练参数
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=500, help="预热步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--eval_steps", type=int, default=500, help="评估间隔步数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存模型间隔步数")
    parser.add_argument("--log_steps", type=int, default=100, help="日志记录间隔步数")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="设备")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建模型配置
    config = ModelConfig(
        vision_model_name=args.vision_model,
        language_model_name=args.language_model,
        projector_type=args.projector_type
    )
    
    # 创建模型
    model = ImageCaptioningModel(config)
    
    # 准备训练数据
    train_dataset = CocoDataset(
        data_dir=args.data_dir,
        ann_file=args.ann_file,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.image_processor,
        max_length=args.max_length,
        image_size=args.image_size,
        image_token=model.image_token,
        split="train",
        max_samples=args.max_samples
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 准备验证数据
    val_dataset = CocoDataset(
        data_dir=args.data_dir,
        ann_file=args.ann_file,
        tokenizer=model.tokenizer,
        image_processor=model.vision_encoder.image_processor,
        max_length=args.max_length,
        image_size=args.image_size,
        image_token=model.image_token,
        split="val",
        max_samples=args.max_samples // 5 if args.max_samples else None  # 验证集样本数量为训练集的1/5
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 设置优化器
    # 冻结视觉编码器，只训练连接器和语言模型
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    
    # 获取需要优化的参数
    optimizer_grouped_parameters = [
        {"params": model.projector.parameters(), "lr": args.lr},
        {"params": model.language_model.parameters(), "lr": args.lr * 0.1}
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 设置学习率调度器
    total_steps = len(train_dataloader) * args.epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练模型
    model.train()
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        epoch_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # 移动数据到设备
            pixel_values = batch["pixel_values"].to(model.device)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # 前向传播
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / args.gradient_accumulation_steps
            epoch_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度累积
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 记录日志
                if global_step % args.log_steps == 0:
                    print(f"Step {global_step}, Loss: {epoch_loss / args.log_steps:.4f}")
                    epoch_loss = 0.0
                
                # 保存模型
                if global_step % args.save_steps == 0:
                    model_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(model_save_path)
                    print(f"Saved model checkpoint to {model_save_path}")
                
                # 评估模型
                if global_step % args.eval_steps == 0:
                    eval_loss = evaluate(model, val_dataloader)
                    print(f"Step {global_step}, Eval Loss: {eval_loss:.4f}")
                    
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_model_path = os.path.join(args.output_dir, "best_model")
                        model.save_pretrained(best_model_path)
                        print(f"Saved best model to {best_model_path}")
                    
                    model.train()
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    print(f"Saved final model to {final_model_path}")

def evaluate(model, dataloader):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 移动数据到设备
            pixel_values = batch["pixel_values"].to(model.device)
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # 前向传播
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            total_steps += 1
    
    return total_loss / total_steps

if __name__ == "__main__":
    main() 