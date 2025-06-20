"""
Main training script for image captioning model
"""

import os
import torch
import argparse

from ImageCap.data.coco_dataset import COCOCaptionDataset, collate_fn
from ImageCap.model.image_captioning_model import ImageCaptioningModel
from ImageCap.train.trainer import ImageCaptioningTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train image captioning model")
    
    parser.add_argument("--vision_encoder_path", type=str, default="ImageCap/models/vit-base-patch16-224",
                        help="Path to vision encoder model")
    parser.add_argument("--language_model_path", type=str, default="ImageCap/models/qwen3-0.6b",
                        help="Path to language model")
    parser.add_argument("--image_dir", type=str, default="ImageCap/data/coco/train2014",
                        help="Directory containing training images")
    parser.add_argument("--annotation_file", type=str, default="ImageCap/data/coco/annotations/captions_train2014.json",
                        help="Path to annotation file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="ImageCap/checkpoints", help="Directory to save model checkpoints")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to use for training (None for all)")
    parser.add_argument("--use_multi_gpus", action="store_true", 
                        help="Whether to use multiple GPUs for training if available")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create dataset
    print("Creating dataset...")
    train_dataset = COCOCaptionDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        image_processor_path=args.vision_encoder_path,
        max_samples=args.max_samples
    )
    
    # Create model
    print("Creating model...")
    model = ImageCaptioningModel(
        vision_encoder_path=args.vision_encoder_path,
        language_model_path=args.language_model_path,
        device=args.device
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = ImageCaptioningTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=None,  # No validation dataset for now
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        collate_fn=collate_fn,
        log_interval=args.log_interval,
        device=args.device,
        use_multi_gpus=args.use_multi_gpus
    )
    
    # Train model
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    main() 