from CNN_RNN import CNNEncoder, RNNDecoder
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
    model['encoder'].train()
    model['decoder'].train()
    total_loss = 0
    for images, captions in tqdm(dataloader):
        images = images.to(device)
        targets = torch.stack([tokenizer.encode(c) for c in captions]).to(device)
        features = model['encoder'](images)
        outputs = model['decoder'](features, targets[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, tokenizer, device):
    model['encoder'].eval()
    model['decoder'].eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions in tqdm(dataloader):
            images = images.to(device)
            targets = torch.stack([tokenizer.encode(c) for c in captions]).to(device)
            features = model['encoder'](images)
            outputs = model['decoder'](features, targets[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    config = {
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'batch_size': 32,
        'epochs': 20,
        'embed_size': 256,
        'hidden_size': 512,
        'lr': 1e-3
    }

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    ckpt_dir = os.path.join('checkpoints', timestamp)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints directory: {ckpt_dir}")
    device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
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

    # 构建分词器（使用训练集的caption构建）
    tokenizer = Tokenizer([c for _, c in train_dataset.captions])
    print(f"Tokenizer done.")

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    # 构建模型
    encoder = CNNEncoder(config['embed_size']).to(device)
    decoder = RNNDecoder(config['embed_size'], config['hidden_size'], len(tokenizer.tokenizer.vocab)).to(device)
    model = {'encoder': encoder, 'decoder': decoder}
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config['lr'])
    print(f"Start training...")
    # 训练
    history = []
    for epoch in range(config['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer, tokenizer, device)
        val_loss = validate(model, val_loader, criterion, tokenizer, device)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})

    # 保存最终模型
    torch.save(encoder.state_dict(), os.path.join(ckpt_dir, 'encoder_final.pth'))
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, 'decoder_final.pth'))

    # 保存训练历史/评估报告
    import json

    with open(os.path.join(ckpt_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training finished...")

    # 测试/生成示例
    encoder.eval()
    decoder.eval()
    results = []
    for images, captions in val_loader:
        images = images.to(device)
        features = encoder(images)
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(20):
            hiddens, states = decoder.lstm(inputs, states)
            outputs = decoder.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = decoder.embed(predicted).unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)
        for i in range(images.size(0)):
            gt = captions[i]
            pred = tokenizer.decode(sampled_ids[i].cpu().numpy())
            print('GT:', gt)
            print('Pred:', pred)
            results.append({'gt': gt, 'pred': pred})

    # 保存生成示例
    with open(os.path.join(ckpt_dir, 'sample_results.json'), 'w') as f:
        json.dump(results, f, indent=2)