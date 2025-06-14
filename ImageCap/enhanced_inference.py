"""
Improved inference script for enhanced image captioning model
"""

import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import traceback
import sys
from transformers import ViTImageProcessor, AutoTokenizer

from ImageCap.model.image_captioning_model import ImageCaptioningModel


def parse_args():
    parser = argparse.ArgumentParser(description="Generate caption for an image using enhanced model")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the enhanced checkpoint file (.pt)")
    parser.add_argument("--vision_encoder_path", type=str, default="ImageCap/models/vit-base-patch16-224",
                        help="Path to vision encoder model")
    parser.add_argument("--language_model_path", type=str, default="ImageCap/models/qwen3-0.6b",
                        help="Path to language model")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--show_image", action="store_true",
                        help="Display the image with its caption")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with verbose output")
    
    return parser.parse_args()


def load_enhanced_model(checkpoint_path, vision_encoder_path, language_model_path, device, debug=False):
    """
    Load the enhanced model with trained weights
    """
    if debug:
        print("Debug info: Starting enhanced model loading...")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Vision encoder path: {vision_encoder_path}")
        print(f"Language model path: {language_model_path}")
        print(f"Device: {device}")
    
    try:
        print("Initializing model components...")
        # Initialize model
        model = ImageCaptioningModel(
            vision_encoder_path=vision_encoder_path,
            language_model_path=language_model_path,
            device=device
        )
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if debug:
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        print("Loading weights into model...")
        # Load trained weights
        model.projector.load_state_dict(checkpoint["projector_state_dict"])
        model.language_model.get_input_embeddings().load_state_dict(checkpoint["embedding_state_dict"])
        
        # Load fine-tuned language model layers if available
        if "language_model_partial_state_dict" in checkpoint and checkpoint["language_model_partial_state_dict"]:
            print("Loading fine-tuned language model layers...")
            for name, param in checkpoint["language_model_partial_state_dict"].items():
                if name in dict(model.language_model.named_parameters()):
                    if debug:
                        print(f"Loading language model parameter: {name}")
                    dict(model.language_model.named_parameters())[name].data.copy_(param)
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        if debug:
            print("Detailed error:")
            traceback.print_exc()
        raise


def generate_caption(model, image_path, device, debug=False):
    """
    Generate caption for an image
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load and preprocess image
        print(f"Loading image processor...")
        image_processor = ViTImageProcessor.from_pretrained(model.vision_encoder.model.config._name_or_path)
        
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        if debug:
            print(f"Image size: {image.size}")
        
        image_inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        if debug:
            print(f"Image tensor shape: {image_inputs.pixel_values.shape}")
        
        # Generate caption
        print("Generating caption...")
        with torch.no_grad():
            outputs = model(image_inputs.pixel_values)
            
            if debug:
                print(f"Output tokens shape: {outputs.shape}")
                print(f"Output tokens: {outputs[0].tolist()}")
            
            # Decode the output tokens to text
            caption = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if debug:
                print(f"Raw caption: {caption}")
            
            # Extract only the caption part (remove the prompt)
            prompt = model.image_token + " Caption: "
            if prompt in caption:
                caption = caption[caption.find(prompt) + len(prompt):].strip()
                
            # Clean up repetitive text - remove consecutive repeated words
            words = caption.split()
            if len(words) > 1:
                clean_words = []
                for i, word in enumerate(words):
                    if i == 0 or word != words[i-1]:
                        clean_words.append(word)
                caption = " ".join(clean_words)
        
        if not caption:
            print("Warning: Generated caption is empty!")
        
        return caption, image
    
    except Exception as e:
        print(f"Error generating caption: {e}")
        if debug:
            print("Detailed error:")
            traceback.print_exc()
        raise


def main():
    args = parse_args()
    
    try:
        print("\n" + "="*50)
        print("ENHANCED IMAGE CAPTIONING INFERENCE")
        print("="*50)
        
        # Load model
        model = load_enhanced_model(
            args.checkpoint_path,
            args.vision_encoder_path,
            args.language_model_path,
            args.device,
            args.debug
        )
        
        # Generate caption
        caption, image = generate_caption(model, args.image_path, args.device, args.debug)
        
        # Print results
        print("\n" + "="*50)
        print(f"IMAGE PATH: {args.image_path}")
        print(f"GENERATED CAPTION: {caption}")
        print("="*50 + "\n")
        
        # Display image with caption if requested
        if args.show_image:
            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.title(caption)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 