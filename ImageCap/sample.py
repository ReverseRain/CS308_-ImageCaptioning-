from ImageCap.evaluate import *
import time

def parse_args_sample():
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
    parser.add_argument("--img_path",type=str)
    
    return parser.parse_args()


def generate_caption(model, image_processor,  img_path, args):
    """
    Generate captions for images in COCO dataset
    """
    model.eval()
    results = {}
    image = Image.open(img_path).convert("RGB")
    
    prompt = " "
    input_ids = model.tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

    image_tensor = image_processor(image, return_tensors="pt").pixel_values.to(args.device)

    vis_features = model.vision_encoder(image_tensor)

    mapped_vis = model.projector(vis_features)

    text_outputs = model.language_model(input_ids=input_ids, output_hidden_states=True)
    text_features = text_outputs.hidden_states[-1]

    text_embeds = model.language_model.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([mapped_vis, text_embeds], dim=1)


    start_time = time.time()
    caption_tokens = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        max_length=70,
        pad_token_id=model.tokenizer.pad_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        attention_mask=torch.ones(inputs_embeds.shape[:-1], dtype=torch.long, device=inputs_embeds.device)
    )
    total_time = time.time() - start_time
    print("total time is ",total_time)
    caption = model.tokenizer.decode(caption_tokens[0], skip_special_tokens=True)

    caption = caption.strip()
    caption = split_caption_at_dot(caption)

    return caption


def main():
    print("+" * 80)
    args = parse_args_sample()
    
    print("=" * 80)
    print("IMAGE CAPTIONING EVALUATION")
    print("=" * 80)
    
    # Load model
    model = load_model(args)
    
    # Get image processor
    image_processor = get_image_processor(model)
    
    
    # Generate captions
    caption = generate_caption(model, image_processor,  args.img_path, args)
    
    print(caption)
if __name__ == "__main__":
    print("++++++" * 30)
    main()