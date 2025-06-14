import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def prepare_qwen_for_imagecap(model_name):
    """
    Prepare a Qwen model for use in ImageCap.
    
    Args:
        model_name: Name or path of the Qwen model.
        
    Returns:
        tuple: (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        device_map="auto"
    )
    
    # Add special tokens for image handling
    special_tokens = {
        "additional_special_tokens": [
            "<image>",
            "<im_patch>",
            "<im_start>",
            "<im_end>"
        ]
    }
    
    # Update tokenizer
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    # Resize token embeddings
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    return tokenizer, model 