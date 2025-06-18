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


def train(model, dataloader, criterion, optimizer, tokenizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()  # 初始化梯度
    for i, (images, captions) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        
        # 为每个caption创建输入和目标，使用带指令的格式
        text_inputs = tokenizer.batch_encode(captions, max_len=50, use_instruction=True)
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
        # 缩放损失值
        loss = loss / accumulation_steps
        loss.backward()

        # 每accumulation_steps步更新一次参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps    
    # 处理最后不足accumulation_steps的部分    
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            
            # 为每个caption创建输入和目标，使用带指令的格式
            text_inputs = tokenizer.batch_encode(captions, max_len=50, use_instruction=True)
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


def generate_caption(model, image, tokenizer, max_length=50, device='cpu', temperature=0.7, top_p=0.9):
    model.eval()
    with torch.no_grad():
        # 1. 准备图像特征，并计算出只执行一次的"视觉印记"
        image = image.unsqueeze(0).to(device)
        vision_features = model.encoder(image)
        vision_language_features = model.connector(vision_features)
        # vision_projection 的形状是 [1, 1, language_embed_size]，它将作为贯穿始终的图像上下文
        vision_projection = model.visual_projection(vision_language_features).unsqueeze(1)

        # 2. 准备初始输入 - 使用指令提示
        # 为Qwen3模型准备带有指令的输入
        prompt = tokenizer.prepare_caption_input(caption=None)  # 推理模式，无真实描述
        input_ids = tokenizer.tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        # 保存生成过程中需要的token
        generated_ids = []
        stop_token_id = tokenizer.eos_token_id

        # 3. 自回归生成循环
        for _ in range(max_length):
            # a. 获取当前输入序列的文本特征 (hidden states)
            text_outputs = model.language_model(input_ids=input_ids, output_hidden_states=True)
            text_features = text_outputs.hidden_states[-1]

            # b. 将"视觉印记"加到最后一个词的特征上，进行信息融合
            last_token_feature = text_features[:, -1:, :]
            fused_feature = last_token_feature + vision_projection

            # c. 使用模型的to_vocab层来预测下一个词的概率
            logits = model.to_vocab(fused_feature).squeeze(1)
            
            # d. 使用温度参数调整概率分布
            logits = logits / temperature
            
            # e. 应用top-p采样（核采样）以增加多样性
            # 对logits进行softmax得到概率
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # 按概率从大到小排序
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            # 计算累积概率
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 创建mask，标记累积概率小于top_p的位置
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # 将第一个元素设为False，确保至少保留一个token
            sorted_indices_to_remove[..., 0] = False
            
            # 应用mask
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            filtered_logits = logits.clone()
            filtered_logits[indices_to_remove] = float('-inf')
            
            # 采样下一个token
            next_token_id = torch.multinomial(torch.nn.functional.softmax(filtered_logits, dim=-1), num_samples=1).item()
            
            # f. 检查是否应该停止生成
            if next_token_id == stop_token_id:
                break
            
            generated_ids.append(next_token_id)

            # g. 将新生成的词添加到输入序列中，为下一次循环做准备
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            
            # 限制输入长度以避免内存溢出
            if input_ids.size(1) > 512:
                input_ids = input_ids[:, -512:]

        # 解码生成的ID序列，得到最终的文本描述
        if not generated_ids:
            return "无法生成描述"
            
        # 只解码模型生成的部分，不包括原始的指令提示
        full_response = tokenizer.decode(generated_ids)

        # 清理生成的文本，移除不必要的前缀
        if "assistant" in full_response:
            # 移除"assistant\n<think>\n\n</think>\n\n"前缀
            # 查找最后一个\n\n后的内容
            try:
                clean_response = full_response.split("</think>\n\n", 1)[1]
            except IndexError:
                # 如果无法按预期分割，尝试其他方式
                if "\n\n" in full_response:
                    clean_response = full_response.split("\n\n", 1)[1]
                else:
                    clean_response = full_response
        else:
            clean_response = full_response
        
        # 清理生成的文本，移除不必要的标记等
        return clean_response.strip()

if __name__ == "__main__":
    config = {
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'batch_size': 12,  # 减小批量大小以适应更大的模型
        'epochs': 2,      # 增加训练轮数
        'vision_embed_size': 768,  # 视觉特征的嵌入维度
        'lr': 2e-5,       # 调整学习率
        'max_length': 50, # 最大生成长度
        'temperature': 0.7, # 生成温度
        'top_p': 0.9,      # 核采样参数
        'train_subset_ratio': 1  # 训练时使用的数据集比例，1.0表示使用全部
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")
    
    # 设置设备
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
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
    
    # 如果需要，创建训练数据集的子集
    if config['train_subset_ratio'] < 1.0:
        train_size = int(len(train_dataset) * config['train_subset_ratio'])
        # 使用PyTorch的random_split来创建子集
        from torch.utils.data import random_split
        train_dataset, _ = random_split(
            train_dataset, 
            [train_size, len(train_dataset) - train_size],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子，确保可重复性
        )
        print(f"Using {train_size} samples ({config['train_subset_ratio']*100:.1f}%) of training data")
    
    print(f"Training Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    # 初始化tokenizer
    tokenizer = Tokenizer(model_name="Qwen/Qwen3-0.6B", max_len=config['max_length'])
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
    
    # 优化所有需要训练的参数
    optimizer = optim.AdamW([
        {'params': model.connector.parameters()},
        {'params': model.to_vocab.parameters()},
        {'params': model.visual_projection.parameters()},
        # 添加顶层transformer层参数，使用较小学习率
        {'params': model.language_model.model.layers[-2:].parameters(), 'lr': config['lr']/10},  
    ], lr=config['lr'], weight_decay=0.01)
    
    print(f"Starting training...")
    
    # 添加学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

    # 训练
    history = []
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, tokenizer, device)
        val_loss = validate(model, val_loader, criterion, tokenizer, device)
        scheduler.step()  # 更新学习率
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
        
        # 每个epoch保存一次检查点
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f'vlm_model_epoch_{epoch+1}.pth'))

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
        # if i >= 5:  # 只测试5个批次
        #     break
            
        images = images.to(device)
        for j in range(images.size(0)):
            gt = captions[j]
            # 使用改进的生成函数，带有温度和top-p采样
            pred = generate_caption(
                model, 
                images[j], 
                tokenizer, 
                max_length=config['max_length'],
                device=device,
                temperature=config['temperature'],
                top_p=config['top_p']
            )
            print('GT:', gt)
            print('Pred:', pred)
            print('---')
            results.append({'gt': gt, 'pred': pred})

    # 保存生成示例
    with open(os.path.join(ckpt_dir, 'sample_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
