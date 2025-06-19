"""
Evaluation script for image captioning model
"""

import os
import json
import argparse
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocotools.coco import COCO
from transformers import ViTImageProcessor,AutoImageProcessor

from ImageCap.model.image_captioning_model import ImageCaptioningModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate image captioning model on COCO test set")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file (.pt)")
    parser.add_argument("--vision_encoder_path", type=str, default="ImageCap/models/vit-base-patch16-224",
                        help="Path to vision encoder model")
    parser.add_argument("--language_model_path", type=str, default="ImageCap/models/qwen3-0.6b",
                        help="Path to language model")
    parser.add_argument("--coco_dir", type=str, required=True,
                        help="Path to COCO dataset directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit evaluation to this many images (for debugging)")
    parser.add_argument("--output_file", type=str, default="evaluation_results.json",
                        help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode")
    
    return parser.parse_args()


def load_model(args):
    """
    Load the model with trained weights
    """
    print(f"Loading model from {args.checkpoint_path}...")
    
    # Initialize model
    model = ImageCaptioningModel(
        vision_encoder_path=args.vision_encoder_path,
        language_model_path=args.language_model_path,
        device=args.device
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    
    # Load trained weights
    model.projector.load_state_dict(checkpoint["projector_state_dict"])
    model.language_model.get_input_embeddings().load_state_dict(checkpoint["embedding_state_dict"])
    
    model.to(args.device)
    model.eval()
    
    return model


def get_image_processor(model):
    """
    Get image processor for the vision encoder
    """
    # return ViTImageProcessor.from_pretrained(model.vision_encoder.model.config._name_or_path)
    return AutoImageProcessor.from_pretrained(model.vision_encoder.model.config._name_or_path,use_fast=True)


def prepare_coco_data(args):
    """
    Prepare COCO dataset for evaluation
    """
    # Determine the annotation file and image folder based on the split
    if args.split == "val":
        ann_file = os.path.join(args.coco_dir, "annotations", "captions_val2014.json")
        image_folder = os.path.join(args.coco_dir, "val2014")
        print("Evaluating on validation set")
    else:
        ann_file = os.path.join(args.coco_dir, "annotations", "captions_test2014.json")
        image_folder = os.path.join(args.coco_dir, "test2014")
        print("Evaluating on test set")
    
    # Load COCO annotations
    coco = COCO(ann_file)
    
    # Get image IDs
    image_ids = list(coco.getImgIds())
    if args.limit is not None:
        image_ids = image_ids[:args.limit]
        print(f"Limiting evaluation to {args.limit} images")
    
    print(f"Found {len(image_ids)} images for evaluation")
    
    return coco, image_ids, image_folder


def generate_captions(model, image_processor, coco, image_ids, image_folder, args):
    """
    Generate captions for images in COCO dataset
    """
    model.eval()
    results = {}
    
    # Process images in batches
    for i in tqdm(range(0, len(image_ids), args.batch_size), desc="Generating captions"):
        batch_image_ids = image_ids[i:i + args.batch_size]
        batch_images = []
        
        # Load images and preprocess
        for img_id in batch_image_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(image_folder, img_info["file_name"])
            
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                batch_images.append(Image.new("RGB", (224, 224), color="black"))  # placeholder for failed images
        
        # # Preprocess images
        # image_inputs = image_processor(images=batch_images, return_tensors="pt").to(args.device)
        
        # # Generate captions
        # with torch.no_grad():
        #     outputs = model(image_inputs.pixel_values)
        
        # Process generated captions
        prompt = " "
        input_ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        for j, img_id in enumerate(batch_image_ids):
            # Determine the offset for decoding (196 patches + prompt length)
            # prompt = model.image_token + " Caption: "
            # prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            # prompt_len = len(prompt_ids)
            # offset = model.vision_encoder(image_inputs.pixel_values).shape[1] + prompt_len - 1
            
            # # Decode the caption
            # if offset < outputs.shape[1]:
            #     caption = model.tokenizer.decode(outputs[j][offset:], skip_special_tokens=True)
            # else:
            #     # Fallback to decoding the whole sequence
            #     caption = model.tokenizer.decode(outputs[j], skip_special_tokens=True)
            #     if model.image_token in caption:
            #         caption = caption.split(model.image_token)[-1]
            #     if "Caption:" in caption:
            #         caption = caption.split("Caption:")[-1]
            
            image_tensor = image_processor(batch_images[j], return_tensors="pt").pixel_values.to(args.device)
            # print("type",type(image_tensor),"   jusa ",type(batch_images[j]))

            vis_features = model.vision_encoder(image_tensor)
            # vis_features = vis_features[:, 0, :].unsqueeze(0)
            # print(vis_features.shape)

            mapped_vis = model.projector(vis_features)

            text_outputs = model.language_model(input_ids=input_ids, output_hidden_states=True)
            text_features = text_outputs.hidden_states[-1]

            text_embeds = model.language_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([mapped_vis, text_embeds], dim=1)

            # input_embeds = mapped_vis
            # print(input_embeds.shape)
            # break


            caption_tokens = model.language_model.generate(
                inputs_embeds=inputs_embeds,
                max_length=70,
                pad_token_id=model.tokenizer.pad_token_id,
                eos_token_id=model.tokenizer.eos_token_id,
                attention_mask=torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=inputs_embeds.device)
            )
            caption = model.tokenizer.decode(caption_tokens[0], skip_special_tokens=True)

            # Clean up the caption
            caption = caption.strip()
            
            # Save the result
            results[img_id] = caption
            
            if args.debug and j < 5:
                # Print some examples for debugging
                print(f"Image ID: {img_id}, Caption: {caption}")
    
    return results

# def generate_caption2(self, image, max_length=50):
#         """为图像生成字幕"""
#         image_tensor = self.preprocess_image(image).to(next(self.parameters()).device)

#         vis_features = self.vision_encoder(image_tensor).last_hidden_state
#         vis_features = vis_features[:, 0, :] 

#         mapped_vis = self.connector(vis_features)

#         input_embeds = mapped_vis.unsqueeze(0)

#         caption_tokens = self.llm.generate(
#             inputs_embeds=input_embeds,
#             max_length=max_length,
#             pad_token_id=self.tokenizer.pad_token_id,
#             eos_token_id=self.tokenizer.eos_token_id,
#             attention_mask=torch.ones(input_embeds.shape[:-1], dtype=torch.long, device=input_embeds.device)
#         )
        
#         caption = self.tokenizer.decode(caption_tokens[0], skip_special_tokens=True)
#         return caption

def evaluate_captions(results, coco, args):
    """
    Evaluate generated captions using BLEU and CIDEr metrics
    """
    # Prepare ground truth captions
    gts = {}
    for img_id in results.keys():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        gts[img_id] = [ann["caption"] for ann in annotations]
    
    # Prepare generated captions
    res = {img_id: [caption] for img_id, caption in results.items()}
    
    # Initialize scorers
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]
    
    # Calculate scores
    scores = {}
    print("\nCalculating evaluation scores:")
    for scorer, method in scorers:
        score, scores_per_image = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                scores[m] = sc
                print(f"{m}: {sc:.4f}")
        else:
            scores[method] = score
            print(f"{method}: {score:.4f}")
    
    return scores, gts, res


def save_results(results, scores, args):
    """
    Save evaluation results to a JSON file
    """
    output = {
        "model_checkpoint": args.checkpoint_path,
        "dataset_split": args.split,
        "number_of_images": len(results),
        "scores": scores,
        "predictions": results
    }
    
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


def save_examples(results, gts, coco, image_folder, args):
    """
    Save a few examples with image, ground truth, and generated caption
    """
    import matplotlib.pyplot as plt
    import random
    
    # Select a few random examples
    num_examples = min(5, len(results))
    example_img_ids = random.sample(list(results.keys()), num_examples)
    
    # Create a directory for examples
    examples_dir = os.path.join(os.path.dirname(args.output_file), "examples")
    os.makedirs(examples_dir, exist_ok=True)
    
    # Save examples
    for img_id in example_img_ids:
        # Load image
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_folder, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        
        # Get captions
        gen_caption = results[img_id]
        gt_captions = gts[img_id]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.title(f"Generated: {gen_caption}")
        plt.figtext(0.5, 0.01, f"Ground Truth: {gt_captions[0]}", wrap=True, horizontalalignment="center", fontsize=10)
        plt.axis("off")
        
        # Save figure
        plt.savefig(os.path.join(examples_dir, f"example_{img_id}.png"), bbox_inches="tight")
        plt.close()
    
    print(f"Example images saved to {examples_dir}")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("IMAGE CAPTIONING EVALUATION")
    print("=" * 80)
    
    # Load model
    model = load_model(args)
    
    # Get image processor
    image_processor = get_image_processor(model)
    
    # Prepare COCO data
    coco, image_ids, image_folder = prepare_coco_data(args)
    
    # Generate captions
    results = generate_captions(model, image_processor, coco, image_ids, image_folder, args)
    
    # Evaluate captions
    scores, gts, res = evaluate_captions(results, coco, args)
    
    # Save results
    save_results(results, scores, args)
    
    # Save some examples (if not limited to validation set)
    if not args.debug and args.split != "test":
        try:
            save_examples(results, gts, coco, image_folder, args)
        except Exception as e:
            print(f"Error saving examples: {e}")
    
    print("=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main() 