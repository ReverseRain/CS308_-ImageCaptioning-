from CNN_RNN import CNNEncoder, RNNDecoder
from Tokenizer import Tokenizer
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


def load_model(checkpoint_dir, embed_size, hidden_size, vocab_size, device):
    encoder = CNNEncoder(embed_size).to(device)
    decoder = RNNDecoder(embed_size, hidden_size, vocab_size).to(device)
    
    # 使用map_location来控制模型加载到指定设备
    encoder.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'encoder_final.pth'), 
                  map_location=device)
    )
    decoder.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'decoder_final.pth'), 
                  map_location=device)
    )
    
    return {'encoder': encoder, 'decoder': decoder}


def predict_caption(model, image_tensor, tokenizer, device, max_length=20):
    model['encoder'].eval()
    model['decoder'].eval()
    
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 添加batch维度
        features = model['encoder'](image_tensor)
        
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        
        for _ in range(max_length):
            hiddens, states = model['decoder'].lstm(inputs, states)
            outputs = model['decoder'].linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            inputs = model['decoder'].embed(predicted).unsqueeze(1)
            
            # 清理不需要的中间变量
            del hiddens, outputs
            torch.cuda.empty_cache()
        
        sampled_ids = torch.stack(sampled_ids, 1)
        predicted_caption = tokenizer.decode(sampled_ids[0].cpu().numpy().tolist())
        
        # 清理GPU内存
        del features, inputs, states, sampled_ids
        torch.cuda.empty_cache()
        
        return predicted_caption


def main():
    # 配置参数
    config = {
        'image_path': 'test/test4.jpg',  # 需要预测的图片路径
        'checkpoint_dir': 'checkpoints/20250618_124426',  # 模型检查点目录
        'device': 'cuda:4' if torch.cuda.is_available() else 'cpu',  # 运行设备
        'embed_size': 256,  # 嵌入维度
        'hidden_size': 512,  # 隐藏层维度
    }
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    # 设置设备
    device = torch.device(config['device'])
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 加载图像
    try:
        image = Image.open(config['image_path']).convert('RGB')
        image_tensor = transform(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print("Loading tokenizer...")
    tokenizer = torch.load(os.path.join(config['checkpoint_dir'], 'tokenizer.pth'))
    
    print("Loading model...")
    model = load_model(config['checkpoint_dir'], config['embed_size'], config['hidden_size'],
                      len(tokenizer), device)
    
    print("Generating caption...")
    caption = predict_caption(model, image_tensor, tokenizer, device)
    
    # 清理GPU内存
    del model
    torch.cuda.empty_cache()
    
    print("\nResults:")
    print(f"Image: {config['image_path']}")
    print(f"Generated Caption: {caption}")


if __name__ == "__main__":
    main() 