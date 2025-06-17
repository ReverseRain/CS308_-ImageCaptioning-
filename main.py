from CNN_RNN import VLMModel
from CocoCaptionDataset import CocoCaptionDataset
from Tokenizer import Tokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time


def train(model, dataloader, criterion, optimizer, tokenizer, device):
    model.train()
    total_loss = 0
    for images, captions in tqdm(dataloader):
        images = images.to(device)
        
        # 为每个caption创建输入和目标
        text_inputs = tokenizer.batch_encode(captions, max_len=50)
        input_ids = text_inputs.input_ids[:, :-1].to(device)  # 除了最后一个token
        target_ids = text_inputs.input_ids[:, 1:].to(device)  # 从第二个token开始
        
        # 前向传播
        logits = model(images, input_ids)
        
        # 确保logits和target_ids的形状匹配
        # logits形状: [batch_size, seq_length, vocab_size]
        # 我们需要reshape为 [batch_size*seq_length, vocab_size]
        logits_flat = logits.reshape(-1, logits.size(-1))
        # target_ids形状: [batch_size, seq_length]
        # 我们需要reshape为 [batch_size*seq_length]
        target_ids_flat = target_ids.reshape(-1)
        
        # 计算损失（只考虑非填充token）
        loss = criterion(logits_flat, target_ids_flat)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            
            # 为每个caption创建输入和目标
            text_inputs = tokenizer.batch_encode(captions, max_len=50)
            input_ids = text_inputs.input_ids[:, :-1].to(device)  # 除了最后一个token
            target_ids = text_inputs.input_ids[:, 1:].to(device)  # 从第二个token开始
            
            # 前向传播
            logits = model(images, input_ids)
            
            # 确保logits和target_ids的形状匹配
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_ids_flat = target_ids.reshape(-1)
            
            # 计算损失（只考虑非填充token）
            loss = criterion(logits_flat, target_ids_flat)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def generate_caption(model, image, tokenizer, max_length=50, device='cpu'):
    model.eval()
    with torch.no_grad():
        # 1. 准备图像特征，并计算出只执行一次的“视觉印记”
        image = image.unsqueeze(0).to(device)
        vision_features = model.encoder(image)
        vision_language_features = model.connector(vision_features)
        # vision_projection 的形状是 [1, 1, language_embed_size]，它将作为贯穿始终的图像上下文
        vision_projection = model.visual_projection(vision_language_features).unsqueeze(1)

        # 2. 准备初始输入
        # 我们使用BOS (Beginning of Sentence) token作为生成的起点
        # 对于没有BOS token的分词器，可以使用一个特殊token或pad_token作为替代
        start_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
        input_ids = torch.tensor([[start_token_id]], dtype=torch.long).to(device)

        generated_ids = []

        # 3. 自回归生成循环
        for _ in range(max_length):
            # a. 获取当前输入序列的文本特征 (hidden states)
            # 注意这里我们直接调用底层的 language_model
            text_outputs = model.language_model(input_ids=input_ids, output_hidden_states=True)
            text_features = text_outputs.hidden_states[-1]

            # b. 将“视觉印记”加到最后一个词的特征上，进行信息融合
            # 这是最关键的一步，确保模型在每一步都“看”着图片
            last_token_feature = text_features[:, -1:, :]
            fused_feature = last_token_feature + vision_projection

            # c. 使用您模型中定义的 to_vocab 层来预测下一个词的概率
            logits = model.to_vocab(fused_feature)
            
            # d. 选择概率最高的词作为下一个词
            next_token_id = torch.argmax(logits, dim=-1).item()
            
            # e. 如果生成了结束符，则停止
            if next_token_id == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token_id)

            # f. 将新生成的词添加到输入序列中，为下一次循环做准备
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

        # 解码生成的ID序列，得到最终的文本描述
        return tokenizer.decode(generated_ids)


if __name__ == "__main__":
    config = {
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'batch_size': 8,  # 减小批量大小以适应更大的模型
        'epochs': 1,
        'vision_embed_size': 768,  # 视觉特征的嵌入维度
        'lr': 5e-5  # 降低学习率以适应预训练模型
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")
    
    # 设置设备
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 图像预处理
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

    # 初始化tokenizer
    tokenizer = Tokenizer(model_name="Qwen/Qwen3-0.6B", max_len=50)
    print(f"Tokenizer initialized.")

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, 
                            drop_last=True)

    # 构建VLM模型
    model = VLMModel(vision_embed_size=config['vision_embed_size'], 
                    language_model_name="Qwen/Qwen3-0.6B").to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # 仅训练连接器和投影层，保持视觉编码器和语言模型冻结
    optimizer = optim.Adam([
        {'params': model.connector.parameters()},
        {'params': model.to_vocab.parameters()},
        {'params': model.visual_projection.parameters()},
    ], lr=config['lr'])
    
    print(f"Starting training...")
    
    # 训练
    history = []
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, tokenizer, device)
        val_loss = validate(model, val_loader, criterion, tokenizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(ckpt_dir, 'vlm_model_final.pth'))

    # 保存训练历史
    import json
    with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training finished. Model saved to {ckpt_dir}")

    # 测试/生成示例
    model.eval()
    results = []
    for i, (images, captions) in enumerate(val_loader):
        if i >= 5:  # 只测试5个批次
            break
            
        images = images.to(device)
        for j in range(images.size(0)):
            gt = captions[j]
            pred = generate_caption(model, images[j], tokenizer, max_length=20, device=device)
            print('GT:', gt)
            print('Pred:', pred)
            print('---')
            results.append({'gt': gt, 'pred': pred})

    # 保存生成示例
    with open(os.path.join(ckpt_dir, 'sample_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
