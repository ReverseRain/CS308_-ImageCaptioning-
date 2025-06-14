import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class ImageCapTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device if hasattr(config, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        # We optimize only the projector and the LM head by default
        trainable_params = []
        if getattr(self.config, 'train_mm_mlp_adapter', True):
            # Train the MLP projector
            trainable_params.extend(self.model.mm_projector.parameters())
        
        if getattr(self.config, 'train_lm_head', True):
            # Train the language model's output layer
            for name, param in self.model.language_model.named_parameters():
                if 'lm_head' in name or 'output_projection' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
        
        if getattr(self.config, 'full_finetune', False):
            # Full fine-tuning
            trainable_params = self.model.parameters()
            
        # Set up the optimizer
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=getattr(self.config, 'learning_rate', 2e-5),
            weight_decay=getattr(self.config, 'weight_decay', 0.0),
        )
        
        # Set up loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
    def train(self, train_dataloader, val_dataloader=None, epochs=10):
        """Train the model for the specified number of epochs"""
        # Create learning rate scheduler
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * getattr(self.config, 'warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss = 0
            
            with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}/{epochs}") as pbar:
                for batch in train_dataloader:
                    images = batch['images'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    outputs = self.model(
                        images=images,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    # Calculate loss and backpropagate
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), getattr(self.config, 'max_grad_norm', 1.0))
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    # Update progress bar
                    train_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()})
            
            # Calculate average training loss for this epoch
            avg_train_loss = train_loss / len(train_dataloader)
            print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(os.path.join(self.config.output_dir, "best_model"))
            
            # Save checkpoint
            if epoch % getattr(self.config, 'save_every', 1) == 0:
                self.save_model(os.path.join(self.config.output_dir, f"checkpoint-{epoch}"))
                
        # Save final model
        self.save_model(os.path.join(self.config.output_dir, "final_model"))
    
    def validate(self, val_dataloader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    images=images,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
        
        return val_loss / len(val_dataloader)
    
    def save_model(self, output_path):
        """Save the model"""
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Save model state
        model_state = {
            'vision_tower': self.model.vision_tower.state_dict() if hasattr(self.model, 'vision_tower') else None,
            'mm_projector': self.model.mm_projector.state_dict() if hasattr(self.model, 'mm_projector') else None,
        }
        
        # Save language model separately if it's not too large
        if getattr(self.config, 'save_language_model', False):
            self.model.language_model.save_pretrained(os.path.join(output_path, "language_model"))
            self.model.tokenizer.save_pretrained(os.path.join(output_path, "language_model"))
        
        # Save the model configuration
        self.config.save_pretrained(output_path) if hasattr(self.config, 'save_pretrained') else None
        
        # Save the model weights
        torch.save(model_state, os.path.join(output_path, "model_state.pth"))
        print(f"Model saved to {output_path}")
    
    def generate_caption(self, images, max_length=50, **generate_kwargs):
        """Generate captions for images"""
        self.model.eval()
        with torch.no_grad():
            return self.model.generate_caption(images, max_length=max_length, **generate_kwargs) 