import os
import torch
import argparse
from transformers import TrainingArguments, set_seed
from torch.utils.data import DataLoader

from ..model import build_imagecap_model
from ..data.dataset import ImageCaptionDataset
from .trainer import ImageCapTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-tower", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--language-model", type=str, default="Qwen/Qwen-0.6B")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/imagecap-model")
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-vision-tower", action="store_true", default=True)
    parser.add_argument("--freeze-language-model", action="store_true", default=False)
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Build model
    model = build_imagecap_model(
        vision_tower_name=args.vision_tower,
        language_model_name=args.language_model,
        freeze_vision_tower=args.freeze_vision_tower,
        freeze_language_model=args.freeze_language_model
    )
    
    # Create dataset
    train_dataset = ImageCaptionDataset(args.train_data, model.tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        remove_unused_columns=False,
        save_strategy="epoch",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # Initialize trainer
    trainer = ImageCapTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model.tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    model.tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    train() 