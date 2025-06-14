import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from dataclasses import dataclass, field
from typing import Optional
import transformers
from transformers import HfArgumentParser

from ..model import ImageCaptioningModel
from .trainer import ImageCapTrainer
from ..data import create_coco_dataloaders


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config we are going to fine-tune"""
    vision_tower: str = field(
        default="google/vit-base-patch16-224", 
        metadata={"help": "Path to the vision encoder model"}
    )
    language_model_path: str = field(
        default="Qwen/Qwen1.5-0.6B", 
        metadata={"help": "Path to the language model"}
    )
    mm_projector_type: str = field(
        default="mlp",
        metadata={"help": "Type of multi-modal projector to use. Options: linear, mlp, mlp{n}x_gelu"}
    )
    mm_vision_select_layer: int = field(
        default=-1,
        metadata={"help": "Which layer of the vision encoder to use (-1 means last layer)"}
    )
    mm_vision_select_feature: str = field(
        default="patch",
        metadata={"help": "Which feature of the vision encoder to use: patch or cls"}
    )


@dataclass
class DataArguments:
    """Arguments pertaining to the data"""
    train_annotation_file: str = field(
        default=None,
        metadata={"help": "Path to the training annotation file"}
    )
    val_annotation_file: str = field(
        default=None,
        metadata={"help": "Path to the validation annotation file"}
    )
    image_dir: str = field(
        default=None,
        metadata={"help": "Directory with the images"}
    )
    max_length: int = field(
        default=77,
        metadata={"help": "Maximum length of tokenized caption"}
    )
    image_size: int = field(
        default=224,
        metadata={"help": "Size of the images"}
    )


@dataclass
class TrainingArguments:
    """Arguments pertaining to training"""
    output_dir: str = field(
        default="./outputs",
        metadata={"help": "Output directory"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for training"}
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay"}
    )
    epochs: int = field(
        default=10,
        metadata={"help": "Number of epochs"}
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of steps for warmup"}
    )
    save_every: int = field(
        default=1,
        metadata={"help": "Save checkpoint every n epochs"}
    )
    train_mm_mlp_adapter: bool = field(
        default=True,
        metadata={"help": "Train the multi-modal MLP adapter"}
    )
    train_lm_head: bool = field(
        default=True,
        metadata={"help": "Train the language model's output layer"}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model"}
    )
    use_fp16: bool = field(
        default=False,
        metadata={"help": "Use mixed precision"}
    )
    save_language_model: bool = field(
        default=False,
        metadata={"help": "Save the language model separately"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for training"}
    )


def train():
    """Main training function"""
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up device
    device = torch.device(training_args.device)
    
    # Create output directory
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    # Create model
    print("Creating model...")
    model_config = argparse.Namespace(**vars(model_args), **vars(training_args))
    model_config.language_model_path = model_args.language_model_path
    
    model = ImageCaptioningModel(model_config)
    
    # Create image transforms
    transform = transforms.Compose([
        transforms.Resize((data_args.image_size, data_args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_coco_dataloaders(
        annotation_train_file=data_args.train_annotation_file,
        annotation_val_file=data_args.val_annotation_file,
        image_dir=data_args.image_dir,
        tokenizer=model.tokenizer,
        transform=transform,
        batch_size=training_args.batch_size,
        max_length=data_args.max_length
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = ImageCapTrainer(model, model_config)
    
    # Train model
    print("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=training_args.epochs
    )
    
    print("Training complete!")


if __name__ == "__main__":
    train() 