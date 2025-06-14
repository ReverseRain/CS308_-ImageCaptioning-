import argparse
import torch
import os
from PIL import Image

from imagecap.model import ImageCapModel
from imagecap.mm_utils import process_images
from imagecap.conversation import get_default_conv_template
from imagecap.constants import DEFAULT_IMAGE_TOKEN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/imagecap-model")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize model
    model = ImageCapModel.from_pretrained(args.model_path)
    model.to(args.device)
    model.eval()
    
    # Load and process image
    image = Image.open(args.image_file).convert('RGB')
    image_tensor = process_images(image, model.image_processor)
    
    # Generate caption
    conv = get_default_conv_template()
    conv.add_message("user", f"{DEFAULT_IMAGE_TOKEN} Please describe this image.")
    prompt = conv.get_prompt()
    
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_tensor.to(args.device),
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True
        )
    
    output = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = output.split("assistant:")[-1].strip()
    
    print(f"\nImage: {args.image_file}")
    print(f"Caption: {response}")

if __name__ == "__main__":
    main() 