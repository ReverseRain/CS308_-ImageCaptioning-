"""
Script to enhance and improve an existing image captioning model
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ImageCap.model.image_captioning_model import ImageCaptioningModel
from ImageCap.data.coco_dataset import COCOCaptionDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance an existing image captioning model")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file (.pt)")
    parser.add_argument("--vision_encoder_path", type=str, default="ImageCap/models/vit-base-patch16-224",
                        help="Path to vision encoder model")
    parser.add_argument("--language_model_path", type=str, default="ImageCap/models/qwen3-0.6b",
                        help="Path to language model")
    parser.add_argument("--image_dir", type=str, default="ImageCap/data/coco/train2014",
                        help="Directory containing training images")
    parser.add_argument("--annotation_file", type=str, default="ImageCap/data/coco/annotations/captions_train2014.json",
                        help="Path to annotation file")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of enhancement epochs")
    parser.add_argument("--save_dir", type=str, default="ImageCap/enhanced_checkpoints", 
                        help="Directory to save enhanced model checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run training on")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum number of samples to use for enhancement")
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, vision_encoder_path, language_model_path, device):
    """
    Load model from checkpoint
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Initialize model
    model = ImageCaptioningModel(
        vision_encoder_path=vision_encoder_path,
        language_model_path=language_model_path,
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights
    model.projector.load_state_dict(checkpoint["projector_state_dict"])
    model.language_model.get_input_embeddings().load_state_dict(checkpoint["embedding_state_dict"])
    
    model.to(device)
    return model


def enhance_model(model, train_dataset, batch_size, learning_rate, num_epochs, save_dir, device):
    """
    Enhance the model with additional training
    """
    # Create DataLoader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )
    
    # Set up optimizer with weight decay
    # This time we'll also fine-tune a few layers of the language model for better adaptation
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
        
    # Unfreeze the last 2 layers of the language model
    for name, param in model.language_model.named_parameters():
        # For Qwen model, check for specific layer patterns
        if "layers" in name and any(f"layers.{i}" in name for i in [-1, -2]):
            param.requires_grad = True
            print(f"Unfreezing: {name}")
        else:
            param.requires_grad = False
    
    trainable_params = [
        {"params": model.projector.parameters(), "weight_decay": 0.01},
        {"params": [p for n, p in model.language_model.named_parameters() if p.requires_grad], "weight_decay": 0.01},
        {"params": model.language_model.get_input_embeddings().parameters(), "weight_decay": 0.01}
    ]
    
    optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    # Create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Training loop
    best_loss = float('inf')
    model.train()
    
    print(f"Starting enhancement training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        total_loss = 0
        
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Enhancement Epoch {epoch + 1}/{num_epochs}"
        )
        
        for step, batch in progress_bar:
            images = batch["images"].to(device)
            captions = batch["captions"]
            
            optimizer.zero_grad()
            outputs = model(images, captions)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            progress_bar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / (step + 1),
                "lr": scheduler.get_last_lr()[0]
            })
            
            # Free memory
            del images, outputs, loss
            if step % 10 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"enhanced_model_epoch_{epoch + 1}.pt")
        
        # Get the state dict for the unfrozen layers
        language_model_partial_state_dict = {}
        for name, param in model.language_model.named_parameters():
            if param.requires_grad:
                language_model_partial_state_dict[name] = param.data.clone()
        
        checkpoint = {
            "projector_state_dict": model.projector.state_dict(),
            "language_model_partial_state_dict": language_model_partial_state_dict,
            "embedding_state_dict": model.language_model.get_input_embeddings().state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "tokenizer_path": model.tokenizer.save_pretrained(os.path.dirname(checkpoint_path) + "/tokenizer")
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_dir, "best_enhanced_model.pt")
            torch.save(checkpoint, best_model_path)
            print(f"Best model saved to {best_model_path}")
    
    return model


def main():
    args = parse_args()
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(
        args.checkpoint_path,
        args.vision_encoder_path,
        args.language_model_path,
        args.device
    )
    
    # Load dataset with limited samples for focused enhancement
    train_dataset = COCOCaptionDataset(
        image_dir=args.image_dir,
        annotation_file=args.annotation_file,
        image_processor_path=args.vision_encoder_path,
        max_samples=args.max_samples
    )
    
    # Enhance model
    enhanced_model = enhance_model(
        model,
        train_dataset,
        args.batch_size,
        args.learning_rate,
        args.num_epochs,
        args.save_dir,
        args.device
    )
    
    print("Model enhancement complete!")
    print(f"Enhanced model saved to {args.save_dir}/best_enhanced_model.pt")


if __name__ == "__main__":
    main() 