#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ImageCap: 图像描述生成系统
命令行入口脚本
"""

import os
import sys
import argparse
from imagecap.run_imagecap import main as run_imagecap
from imagecap.train.train import train as train_imagecap

def main():
    parser = argparse.ArgumentParser(description="ImageCap: Image Captioning")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Run model parser
    run_parser = subparsers.add_parser("run", help="Run inference with ImageCap model")
    run_parser.add_argument("--model-path", type=str, default="checkpoints/imagecap-model")
    run_parser.add_argument("--image-file", type=str, required=True)
    run_parser.add_argument("--device", type=str, default="cuda")
    
    # Train model parser
    train_parser = subparsers.add_parser("train", help="Train ImageCap model")
    train_parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    train_parser.add_argument("--language-model", type=str, default="Qwen/Qwen-0.6B")
    train_parser.add_argument("--train-data", type=str, required=True)
    train_parser.add_argument("--output-dir", type=str, default="./checkpoints/imagecap-model")
    train_parser.add_argument("--num-train-epochs", type=int, default=3)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-5)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--freeze-vision-tower", action="store_true", default=True)
    train_parser.add_argument("--freeze-language-model", action="store_true", default=False)
    
    # Parse arguments and run appropriate function
    args = parser.parse_args()
    
    if args.command == "run":
        sys.argv = [sys.argv[0]] + ["--model-path", args.model_path, 
                                   "--image-file", args.image_file, 
                                   "--device", args.device]
        run_imagecap()
    elif args.command == "train":
        sys.argv = [sys.argv[0]] + ["--vision-tower", args.vision_tower,
                                   "--language-model", args.language_model,
                                   "--train-data", args.train_data,
                                   "--output-dir", args.output_dir,
                                   "--num-train-epochs", str(args.num_train_epochs),
                                   "--per-device-train-batch-size", str(args.per_device_train_batch_size),
                                   "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
                                   "--learning-rate", str(args.learning_rate),
                                   "--seed", str(args.seed)]
        if args.freeze_vision_tower:
            sys.argv.append("--freeze-vision-tower")
        if args.freeze_language_model:
            sys.argv.append("--freeze-language-model")
        train_imagecap()
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 