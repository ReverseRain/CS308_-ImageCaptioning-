import os
import json
import torch
import argparse
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from transformers import HfArgumentParser

from ..model import ImageCaptioningModel
from ..data import COCOCaptionDataset

# Try to import pycocoevalcap, with a helpful error if not available
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
except ImportError:
    print("""
    Please install the pycocoevalcap package to use the evaluation:
    pip install pycocoevalcap
    """)


@dataclass
class EvaluationArguments:
    """Arguments for evaluation"""
    model_path: str = field(
        default=None,
        metadata={"help": "Path to the saved model state"}
    )
    vision_tower: str = field(
        default="google/vit-base-patch16-224", 
        metadata={"help": "Path to the vision encoder model"}
    )
    language_model_path: str = field(
        default="Qwen/Qwen1.5-0.6B", 
        metadata={"help": "Path to the language model"}
    )
    annotation_file: str = field(
        default=None,
        metadata={"help": "Path to the validation annotation file"}
    )
    image_dir: str = field(
        default=None,
        metadata={"help": "Directory with the images"}
    )
    output_file: str = field(
        default="predictions.json",
        metadata={"help": "Path to save the predictions"}
    )
    batch_size: int = field(
        default=16,
        metadata={"help": "Batch size for evaluation"}
    )
    max_length: int = field(
        default=30,
        metadata={"help": "Maximum length of generated caption"}
    )
    image_size: int = field(
        default=224,
        metadata={"help": "Size of the images"}
    )
    beam_size: int = field(
        default=4,
        metadata={"help": "Beam size for generation"}
    )
    device: str = field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        metadata={"help": "Device to use for evaluation"}
    )


def evaluate_coco_captioning():
    """Evaluate a trained image captioning model on COCO"""
    # Parse arguments
    parser = HfArgumentParser(EvaluationArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set up device
    device = torch.device(args.device)
    
    # Create image transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create model config
    model_config = argparse.Namespace(
        vision_tower=args.vision_tower,
        language_model_path=args.language_model_path,
        use_fp16=False,
        device=args.device
    )
    
    # Create model
    print("Loading model...")
    model = ImageCaptioningModel(model_config)
    
    # Load saved model weights if provided
    if args.model_path:
        print(f"Loading model weights from {args.model_path}")
        model_state = torch.load(os.path.join(args.model_path, "model_state.pth"), map_location=device)
        
        if model_state['vision_tower'] is not None:
            model.vision_tower.load_state_dict(model_state['vision_tower'])
            
        if model_state['mm_projector'] is not None:
            model.mm_projector.load_state_dict(model_state['mm_projector'])
    
    model.to(device)
    model.eval()
    
    # Load COCO dataset
    print("Loading validation data...")
    with open(args.annotation_file, 'r') as f:
        annotations = json.load(f)
    
    # Create dictionary to store results
    predictions = {}
    ground_truth = {}
    
    # Image id to filename mapping
    id_to_filename = {}
    for img in annotations['images']:
        id_to_filename[img['id']] = img['file_name']
    
    # Process all images in the validation set
    image_ids = list(id_to_filename.keys())
    
    print(f"Generating captions for {len(image_ids)} images...")
    for i in tqdm(range(0, len(image_ids), args.batch_size)):
        batch_ids = image_ids[i:i+args.batch_size]
        batch_images = []
        
        for img_id in batch_ids:
            # Get image path
            image_filename = id_to_filename[img_id]
            image_path = os.path.join(args.image_dir, image_filename)
            
            # Load and transform image
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            batch_images.append(image_tensor)
        
        # Combine batch of images
        batch_images = torch.cat(batch_images, dim=0)
        
        # Generate captions
        with torch.no_grad():
            captions = model.generate_caption(
                batch_images, 
                max_length=args.max_length,
                num_beams=args.beam_size
            )
        
        # Store predictions
        for img_id, caption in zip(batch_ids, captions):
            predictions[img_id] = [caption]
    
    # Create ground truth dictionary
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        caption = ann['caption'].strip()
        
        if img_id in ground_truth:
            ground_truth[img_id].append(caption)
        else:
            ground_truth[img_id] = [caption]
    
    # Evaluate using pycocoevalcap
    print("Computing evaluation metrics...")
    
    # Make sure ground_truth and predictions have the same image ids
    common_ids = set(ground_truth.keys()) & set(predictions.keys())
    
    # Convert to the format required by pycocoevalcap
    gts = {img_id: ground_truth[img_id] for img_id in common_ids}
    res = {img_id: predictions[img_id] for img_id in common_ids}
    
    # Calculate scores
    metrics = {}
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                metrics[m] = sc
        else:
            metrics[method] = score
    
    # Print results
    print("Evaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    
    # Save predictions
    print(f"Saving predictions to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump({
            "predictions": predictions,
            "metrics": metrics
        }, f, indent=4)
    
    return metrics


if __name__ == "__main__":
    evaluate_coco_captioning() 