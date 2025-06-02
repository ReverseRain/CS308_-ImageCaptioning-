import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import logging
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from PIL import Image

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ImageCaptioningModel
from data import COCOEvalDataset, get_processors

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="评估图像描述生成模型")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--data_dir", type=str, required=True, help="COCO数据集图像目录")
    parser.add_argument("--ann_file", type=str, required=True, help="COCO数据集注释文件")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--max_length", type=int, default=50, help="生成的最大长度")
    parser.add_argument("--prompt", type=str, default="请为这张图片生成描述：", help="生成提示")
    
    return parser.parse_args()

def load_coco_annotations(ann_file):
    """加载COCO数据集注释"""
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # 创建图像ID到文件名的映射
    id_to_filename = {}
    for image in data['images']:
        id_to_filename[image['id']] = image['file_name']
    
    # 创建图像ID到caption的映射
    id_to_captions = {}
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        if image_id not in id_to_captions:
            id_to_captions[image_id] = []
        id_to_captions[image_id].append(annotation['caption'])
    
    return id_to_filename, id_to_captions

def evaluate(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = ImageCaptioningModel.from_pretrained(args.model_path)
    model.eval()
    
    # 获取处理器
    processor = model.processor if hasattr(model, 'processor') else get_processors(
        vision_model_name=model.vision_encoder.config._name_or_path,
        language_model_name=model.language_model.config._name_or_path
    )
    
    # 加载数据集
    logger.info(f"加载数据集: {args.ann_file}")
    id_to_filename, id_to_captions = load_coco_annotations(args.ann_file)
    
    # 初始化评估指标
    bleu_scorer = Bleu(n=4)
    cider_scorer = Cider()
    
    # 用于评估的数据
    all_image_ids = list(id_to_captions.keys())
    test_image_ids = [img_id for img_id in all_image_ids if img_id % 5 == 0]  # 简单起见使用图像ID模5为0的作为测试集
    
    # 生成描述
    logger.info("生成图像描述...")
    results = {}
    gts = {}
    
    for i, image_id in enumerate(tqdm(test_image_ids)):
        # 加载图像
        image_path = os.path.join(args.data_dir, id_to_filename[image_id])
        image = Image.open(image_path).convert("RGB")
        
        # 生成描述
        with torch.no_grad():
            caption = model.generate_caption(image, prompt=args.prompt, max_length=args.max_length)
        
        # 保存结果
        results[str(image_id)] = [caption]
        gts[str(image_id)] = id_to_captions[image_id]
    
    # 计算BLEU和CIDEr分数
    logger.info("计算评估指标...")
    bleu_scores, _ = bleu_scorer.compute_score(gts, results)
    cider_score, _ = cider_scorer.compute_score(gts, results)
    
    # 打印结果
    logger.info("评估结果:")
    logger.info(f"BLEU-1: {bleu_scores[0]:.4f}")
    logger.info(f"BLEU-2: {bleu_scores[1]:.4f}")
    logger.info(f"BLEU-3: {bleu_scores[2]:.4f}")
    logger.info(f"BLEU-4: {bleu_scores[3]:.4f}")
    logger.info(f"CIDEr: {cider_score:.4f}")
    
    # 保存结果
    result_file = os.path.join(args.output_dir, "eval_results.json")
    with open(result_file, 'w') as f:
        json.dump({
            "bleu1": float(bleu_scores[0]),
            "bleu2": float(bleu_scores[1]),
            "bleu3": float(bleu_scores[2]),
            "bleu4": float(bleu_scores[3]),
            "cider": float(cider_score)
        }, f, indent=2)
    
    logger.info(f"结果已保存到: {result_file}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args) 