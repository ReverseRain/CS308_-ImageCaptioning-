import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from typing import Optional, Dict, List, Union, Any

class ImageCapTrainer(Trainer):
    """
    Trainer for ImageCap models.
    Extends Hugging Face's Trainer class with additional functionality for handling image-text pairs.
    """
    
    def __init__(self, model=None, args=None, data_collator=None, train_dataset=None,
                 eval_dataset=None, tokenizer=None, **kwargs):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for image-text tasks.
        """
        # Extract inputs
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        images = inputs.get("images")
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            images=images,
        )
        
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss 