"""
Trainer module for image captioning model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class ImageCaptioningTrainer:
    """
    Trainer for image captioning model
    """
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset=None,
        batch_size=4,
        learning_rate=5e-5,
        num_epochs=1,
        save_dir="./checkpoints",
        collate_fn=None,
        log_interval=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_multi_gpus=True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.collate_fn = collate_fn
        self.log_interval = log_interval
        self.device = device
        self.use_multi_gpus = use_multi_gpus
        
        # Create DataLoader
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=7,
            collate_fn=collate_fn,
            pin_memory=True if device == "cuda" else False,
        )
        
        if val_dataset:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=7,
                collate_fn=collate_fn,
                pin_memory=True if device == "cuda" else False,
            )
        else:
            self.val_dataloader = None
        
        # Prepare model
        self.model.to(device)
        # Setup multi-GPU if available and requested
        if self.use_multi_gpus and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            self.model = nn.DataParallel(self.model)
        
        # Set up optimizer - only optimize projector and special token embeddings
        # Freeze vision encoder and language model parameters
        # for param in self.model.module.vision_encoder.parameters() if isinstance(self.model, nn.DataParallel) else self.model.vision_encoder.parameters():
        #     param.requires_grad = False
            
        # for param in self.model.module.language_model.parameters() if isinstance(self.model, nn.DataParallel) else self.model.language_model.parameters():
        #     param.requires_grad = False
            
        # Only train projector and token embeddings
        # if isinstance(self.model, nn.DataParallel):
        #     trainable_params = [
        #         {"params": self.model.module.projector.parameters()}
        #         # {"params": self.model.module.language_model.get_input_embeddings().parameters()}
        #     ]
        # else:
        #     trainable_params = [
        #         {"params": self.model.projector.parameters()}
        #         # {"params": self.model.language_model.get_input_embeddings().parameters()}
        #     ]
        trainable_params = [
                {"params": self.model.projector.parameters()}
                # {"params": self.model.language_model.parameters()}
                # {"params": self.model.vision_encoder.parameters()}
            ]

        self.optimizer = optim.AdamW(trainable_params, lr=learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Create save directory if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def train(self):
        """
        Train the model
        """
        best_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            # Training loop
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )
            
            for step, batch in progress_bar:
                images = batch["images"].to(self.device)
                captions = batch["captions"]
                
                self.optimizer.zero_grad()
                
                outputs = self.model(images, captions)
                
                # Fix for DataParallel: make sure loss is scalar
                if isinstance(outputs.loss, torch.Tensor) and outputs.loss.numel() > 1:
                    loss = outputs.loss.mean()  # Take mean of losses from different GPUs
                else:
                    loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if step % self.log_interval == 0:
                    progress_bar.set_postfix({
                        "loss": loss.item(),
                        "avg_loss": total_loss / (step + 1)
                    })
            
            avg_train_loss = total_loss / len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs} - Average Training Loss: {avg_train_loss:.4f}")
            
            self.scheduler.step()
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            self.save_checkpoint(checkpoint_path)
            
            # Save best model
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_path = os.path.join(self.save_dir, "best_model.pt")
                self.save_checkpoint(best_model_path)
                print(f"Best model saved to {best_model_path}")
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint
        """
        if isinstance(self.model, nn.DataParallel):
            projector_state_dict = self.model.module.projector.state_dict()
            embedding_state_dict = self.model.module.language_model.get_input_embeddings().state_dict()
            tokenizer = self.model.module.tokenizer
        else:
            projector_state_dict = self.model.projector.state_dict()
            embedding_state_dict = self.model.language_model.get_input_embeddings().state_dict()
            tokenizer = self.model.tokenizer
            
        checkpoint = {
            "projector_state_dict": projector_state_dict,
            "embedding_state_dict": embedding_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "tokenizer_path": tokenizer.save_pretrained(os.path.dirname(path) + "/tokenizer")
        }
        
        torch.save(checkpoint, path) 