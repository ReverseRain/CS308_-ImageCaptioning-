"""
Inference script for image captioning model
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
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the checkpoint file (.pt)")
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


def load_model(checkpoint_path, vision_encoder_path, language_model_path, device, debug=False):
    """
    Load the model with trained weights
    """
    if debug:
        print("Debug info: Starting model loading...")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Vision encoder path: {vision_encoder_path}")
        print(f"Language model path: {language_model_path}")
        print(f"Device: {device}")
    
    try:
        print("Initializing model components...")
        # Initialize model components
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
            if 'projector_state_dict' in checkpoint:
                print(f"Projector keys: {list(checkpoint['projector_state_dict'].keys())}")
                # 打印投影器第一层权重的统计信息
                if 'linear1.weight' in checkpoint['projector_state_dict']:
                    weight = checkpoint['projector_state_dict']['linear1.weight']
                    print(f"Projector linear1.weight stats: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
            
            if 'embedding_state_dict' in checkpoint:
                print(f"Embedding keys: {list(checkpoint['embedding_state_dict'].keys())}")
                # 打印嵌入权重的统计信息
                if 'weight' in checkpoint['embedding_state_dict']:
                    weight = checkpoint['embedding_state_dict']['weight']
                    print(f"Embedding weight stats: shape={weight.shape}, mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
        
        print("Loading weights into model...")
        # 确保模型处于eval模式以避免dropout等
        model.eval()
        
        # Load trained weights
        if 'projector_state_dict' in checkpoint:
            # 打印加载前的投影器状态
            if debug:
                for name, param in model.projector.named_parameters():
                    print(f"Before loading - Projector {name} stats: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            
            # 加载投影器权重
            model.projector.load_state_dict(checkpoint["projector_state_dict"])
            
            # 验证权重是否正确加载
            if debug:
                for name, param in model.projector.named_parameters():
                    print(f"After loading - Projector {name} stats: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                    # 确认参数已更新（不应与初始值相同）
                    print(f"  Parameter requires gradient: {param.requires_grad}")
        else:
            raise KeyError("Checkpoint is missing 'projector_state_dict'")
        
        # 加载嵌入层权重
        if 'embedding_state_dict' in checkpoint:
            # 获取嵌入层
            embedding_layer = model.language_model.get_input_embeddings()
            
            # 打印加载前的嵌入层状态
            if debug:
                for name, param in embedding_layer.named_parameters():
                    print(f"Before loading - Embedding {name} stats: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
            
            # 确认现有嵌入层与检查点兼容
            checkpoint_embedding = checkpoint["embedding_state_dict"]['weight']
            current_embedding = embedding_layer.weight
            if checkpoint_embedding.shape != current_embedding.shape:
                print(f"WARNING: Embedding shape mismatch! Checkpoint: {checkpoint_embedding.shape}, Current: {current_embedding.shape}")
                # 处理嵌入大小不匹配的情况
                if checkpoint_embedding.shape[1] == current_embedding.shape[1]:  # 如果嵌入维度匹配但词汇表大小不同
                    # 只加载词汇表重叠部分
                    min_vocab = min(checkpoint_embedding.shape[0], current_embedding.shape[0])
                    print(f"Loading first {min_vocab} token embeddings only")
                    temp_dict = {'weight': checkpoint_embedding[:min_vocab]}
                    embedding_layer.load_state_dict(temp_dict, strict=False)
                else:
                    print("Cannot load embeddings due to dimension mismatch!")
            else:
                # 尺寸匹配，正常加载
                embedding_layer.load_state_dict(checkpoint["embedding_state_dict"])
            
            # 验证权重是否正确加载
            if debug:
                for name, param in embedding_layer.named_parameters():
                    print(f"After loading - Embedding {name} stats: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
        else:
            raise KeyError("Checkpoint is missing 'embedding_state_dict'")
        
        # 移至指定设备
        model.to(device)
        # 确保模型处于评估模式
        model.eval()
        print("Model loaded successfully!")
        
        # 测试模型生成一些随机token以确保一切正常
        if debug:
            print("\nPerforming quick model test...")
            # 创建一个小的随机图像张量进行测试
            test_image = torch.randn(1, 3, 224, 224).to(device)
            with torch.no_grad():
                # 只运行视觉编码器+投影器部分
                vis_features = model.vision_encoder(test_image)
                proj_features = model.projector(vis_features)
                print(f"Test: Vision features shape: {vis_features.shape}")
                print(f"Test: Projected features shape: {proj_features.shape}")
                print(f"Test: Projected features stats: mean={proj_features.mean().item():.6f}, std={proj_features.std().item():.6f}")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        if debug:
            print("Detailed error:")
            traceback.print_exc()
        raise


def generate_caption(model, image_path, device, debug=False):
    """
    Generate caption for an image with advanced post-processing
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
        
        # 应用标准预处理并确保正确的设备位置
        image_inputs = image_processor(images=image, return_tensors="pt").to(device)
        
        if debug:
            print(f"Image tensor shape: {image_inputs.pixel_values.shape}")
        
        # Generate caption
        print("Generating caption...")
        
        # 设置推理模式以避免梯度计算
        with torch.no_grad():
            # 将模型设置为评估模式以确保一致性
            model.eval()
            
            # 传递图像到模型
            gen_tokens = model(image_inputs.pixel_values)
            
            if debug:
                print(f"Output type: {type(gen_tokens)}")
                print(f"Output shape: {gen_tokens.shape}")
                print(f"First few tokens: {gen_tokens[0][:20].tolist()}")
            
            # 计算提示词的长度，以便跳过它们
            prompt = model.image_token + " Caption: "
            prompt_ids = model.tokenizer(prompt, add_special_tokens=False).input_ids
            prompt_len = len(prompt_ids)
            
            if debug:
                print(f"Prompt tokens: {prompt_ids}")
                print(f"Prompt length: {prompt_len}")
            
            offset = model.vision_encoder(image_inputs.pixel_values).shape[1] + prompt_len - 1
            
            if debug:
                print(f"Calculated offset for decoding: {offset}")
                if offset < gen_tokens.shape[1]:
                    print(f"Tokens to decode: {gen_tokens[0][offset:offset+20].tolist()}")
            
            # 从该偏移位置解码token (如果有)
            if offset < gen_tokens.shape[1]:
                caption = model.tokenizer.decode(gen_tokens[0][offset:], skip_special_tokens=True)
            else:
                # 如果偏移超出范围，尝试直接解码全部内容，然后处理
                caption = model.tokenizer.decode(gen_tokens[0], skip_special_tokens=True)
                # 尝试去除提示部分
                if model.image_token in caption:
                    caption = caption.split(model.image_token)[-1]
                if "Caption:" in caption:
                    caption = caption.split("Caption:")[-1]
            
            if debug:
                print(f"Raw caption: {caption}")
            
            # 基本清理
            caption = caption.strip()
            
            # 如果caption为空，尝试不同的解码方法
            if not caption:
                print("Warning: Initial caption is empty, trying alternative decoding...")
                # 尝试解码生成序列的不同部分
                caption = model.tokenizer.decode(gen_tokens[0][1:], skip_special_tokens=True)
            
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
        print("IMAGE CAPTIONING INFERENCE")
        print("="*50)
        
        # Load model
        model = load_model(
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