from CNN_RNN import CNNEncoder, QwenDecoder
from CocoCaptionDataset import CocoCaptionDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from torch.amp import autocast, GradScaler


def train(model, dataloader, criterion, optimizer, device, scaler):
    model['encoder'].train()
    model['decoder'].train()
    total_loss = 0
    for images, captions in tqdm(dataloader):
        images = images.to(device)
        optimizer.zero_grad()
        
        # 使用混合精度训练
        with autocast(device_type='cuda', dtype=torch.float16):
            features = model['encoder'](images)
            outputs = model['decoder'](features, captions)
            loss = outputs.loss

        # 使用scaler来处理梯度
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model['encoder'].parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model['decoder'].parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model['encoder'].eval()
    model['decoder'].eval()
    total_loss = 0
    with torch.no_grad(), autocast(device_type='cuda', dtype=torch.float16):
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            features = model['encoder'](images)
            outputs = model['decoder'](features, captions)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(dataloader)


def calculate_scores(gts, res):
    """计算BLEU和CIDEr分数
    Args:
        gts: 字典，key为图片id，value为参考描述列表
        res: 字典，key为图片id，value为生成的描述列表
    Returns:
        scores: 字典，包含各项评分
    """
    # 初始化评分器
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]
    
    # 计算所有得分
    scores = {}
    for scorer, method in scorers:
        score, scores_per_caption = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                scores[m] = sc
        else:
            scores[method] = score
    
    return scores


if __name__ == "__main__":
    config = {
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'batch_size': 4,  # 进一步减小batch size
        'epochs': 2,
        'embed_size': 1024,  # 调整为与Qwen模型匹配的维度
        'lr': 5e-5,  # 降低学习率以适应预训练模型的微调
        'gradient_clip': 1.0  # 添加梯度裁剪
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")
    
    # 设置PyTorch内存分配器配置
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])

    # 构建训练集和验证集
    train_dataset = CocoCaptionDataset(config['train_img_dir'], config['train_ann_file'], transform)
    val_dataset = CocoCaptionDataset(config['val_img_dir'], config['val_ann_file'], transform)
    print(f"Training Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    # 构建模型
    encoder = CNNEncoder(config['embed_size']).to(device)
    decoder = QwenDecoder(config['embed_size']).to(device)
    model = {'encoder': encoder, 'decoder': decoder}
    
    # 优化器 - 使用不同的学习率
    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': config['lr']},
        {'params': decoder_params, 'lr': config['lr'] * 0.1}  # 对预训练模型使用更小的学习率
    ])
    
    # 创建梯度缩放器用于混合精度训练
    scaler = GradScaler(enabled=True)
    
    print(f"Start training...")
    
    # 训练
    history = []
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, None, optimizer, device, scaler)
        val_loss = validate(model, val_loader, None, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'encoder_best.pth'))
            torch.save(decoder.state_dict(), os.path.join(ckpt_dir, 'decoder_best.pth'))
        
        history.append({
            'epoch': epoch+1, 
            'train_loss': train_loss, 
            'val_loss': val_loss
        })

        # 每个epoch都保存一次
        torch.save(encoder.state_dict(), os.path.join(ckpt_dir, f'encoder_epoch_{epoch+1}.pth'))
        torch.save(decoder.state_dict(), os.path.join(ckpt_dir, f'decoder_epoch_{epoch+1}.pth'))

    # 保存最终模型
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'encoder_final.pth'))
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, 'decoder_final.pth'))

    # 保存训练历史
    import json
    with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training finished...")

    # 测试/生成示例
    encoder.eval()
    decoder.eval()
    results = []
    gts = {}  # 用于存储参考描述
    res = {}  # 用于存储生成的描述
    
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            features = encoder(images)
            generated_ids = decoder(features)
            
            # 解码生成的文本
            generated_captions = decoder.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i in range(images.size(0)):
                image_id = f"{batch_idx}_{i}"  # 创建唯一的图片ID
                gt = captions[i]
                pred = generated_captions[i]
                
                # 存储结果用于计算指标
                gts[image_id] = [gt]  # 参考描述需要是列表格式
                res[image_id] = [pred]
                
                results.append({
                    'image_id': image_id,
                    'gt': gt,
                    'pred': pred
                })
                
                if len(results) >= 100:  # 限制评估样本数量
                    break
            
            if len(results) >= 100:
                break
    
    # 计算评估指标
    eval_scores = calculate_scores(gts, res)
    print("\nEvaluation Scores:")
    for metric, score in eval_scores.items():
        print(f"{metric}: {score:.4f}")
    
    # 保存结果和评估指标
    final_results = {
        'predictions': results,
        'metrics': eval_scores
    }
    
    with open(os.path.join(ckpt_dir, 'sample_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2) 