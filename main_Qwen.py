import os
# 设置PyTorch内存分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from CNN_RNN import CNNEncoder
from QwenDecoder import QwenDecoder
from CocoCaptionDataset import CocoCaptionDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from pycocoevalcap.cider.cider import Cider

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt...")
    nltk.download('punkt')


def train(model, dataloader, optimizer, device):
    model['encoder'].train()
    model['decoder'].train()
    total_loss = 0
    optimizer.zero_grad()
    
    for idx, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        features = model['encoder'](images)
        outputs = model['decoder'](features, captions)
        loss = outputs.loss / config['accumulation_steps']  # 根据梯度累积步数缩放损失
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (idx + 1) % config['accumulation_steps'] == 0:
            # 应用梯度裁剪
            torch.nn.utils.clip_grad_norm_(model['encoder'].parameters(), config['gradient_clip'])
            torch.nn.utils.clip_grad_norm_(model['decoder'].parameters(), config['gradient_clip'])
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True可以节省内存
        
        total_loss += loss.item() * config['accumulation_steps']  # 恢复原始损失大小
        
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model['encoder'].eval()
    model['decoder'].eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            features = model['encoder'](images)
            outputs = model['decoder'](features, captions)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    config = {
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'batch_size': 4,  # 减小batch size
        'epochs': 20,
        'embed_size': 1024,
        'lr': 5e-5,
        'gradient_clip': 1.0,
        'accumulation_steps': 4  # 添加梯度累积步数
    }

    # 创建offload目录
    os.makedirs('offload', exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")
    
    # 设置空闲GPU
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 构建训练集和验证集
    train_dataset = CocoCaptionDataset(config['train_img_dir'], config['train_ann_file'], transform)
    val_dataset = CocoCaptionDataset(config['val_img_dir'], config['val_ann_file'], transform)
    print(f"Training Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    # 构建模型 - 确保所有参数都是float32
    encoder = CNNEncoder(config['embed_size']).to(device)
    decoder = QwenDecoder(config['embed_size']).to(device)
    
    # 确保所有参数都是float32
    for param in encoder.parameters():
        param.data = param.data.float()
            
    model = {'encoder': encoder, 'decoder': decoder}

    # 优化器 - 使用不同的学习率
    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config['lr']},
        {'params': decoder_params, 'lr': config['lr'] * 0.1}  # 对预训练模型使用更小的学习率
    ])

    print(f"Start training...")
    # 训练
    history = []
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'encoder_best.pth'))
            torch.save(decoder.state_dict(), os.path.join(ckpt_dir, 'decoder_best.pth'))
        
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

    # 保存最终模型
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'encoder_final.pth'))
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, 'decoder_final.pth'))

    # 保存训练历史
    with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training finished...")

    # 测试/生成示例
    encoder.eval()
    decoder.eval()
    results = []
    all_references = []
    all_hypotheses = []
    
    with torch.no_grad():
        for images, captions in tqdm(val_loader):
            images = images.to(device)
            features = encoder(images)
            generated_ids = decoder(features)
            generated_texts = decoder.decode(generated_ids)
            
            for i in range(len(generated_texts)):
                gt = captions[i]
                pred = generated_texts[i]
                print('GT:', gt)
                print('Pred:', pred)
                
                # 为BLEU评分准备数据
                reference = nltk.word_tokenize(gt.lower())
                hypothesis = nltk.word_tokenize(pred.lower())
                all_references.append([reference])
                all_hypotheses.append(hypothesis)
                
                results.append({
                    'gt': gt,
                    'pred': pred
                })

    # 计算BLEU分数
    bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # 计算CIDEr分数
    scorer = Cider()
    # 准备CIDEr评分所需的格式
    gts = {i: [results[i]['gt']] for i in range(len(results))}
    res = {i: [results[i]['pred']] for i in range(len(results))}
    cider_score, _ = scorer.compute_score(gts, res)

    # 添加评分到结果中
    metrics = {
        'bleu1': float(bleu1),
        'bleu2': float(bleu2),
        'bleu3': float(bleu3),
        'bleu4': float(bleu4),
        'cider': float(cider_score)
    }
    
    # 保存生成示例和评分
    final_results = {
        'metrics': metrics,
        'samples': results
    }
    
    with open(os.path.join(ckpt_dir, 'sample_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print("\nEvaluation Metrics:")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"CIDEr: {cider_score:.4f}")