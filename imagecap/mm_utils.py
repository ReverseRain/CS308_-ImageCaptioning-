import torch
import base64
from io import BytesIO
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

def load_image_from_base64(image_str):
    """Load image from base64 string."""
    return Image.open(BytesIO(base64.b64decode(image_str)))

def process_images(images, image_processor, return_tensors="pt"):
    """Process images for the model using the image processor."""
    if not isinstance(images, list):
        images = [images]
    
    processed_images = []
    for image in images:
        if isinstance(image, str):
            # Check if it's a base64 string
            try:
                image = load_image_from_base64(image)
            except:
                # If not base64, assume it's a file path
                image = Image.open(image).convert('RGB')
        
        # Apply processing
        processed_image = image_processor(image, return_tensors=return_tensors)
        processed_images.append(processed_image)
    
    if len(processed_images) == 1:
        return processed_images[0]
    return processed_images 