from final_approch.CNN_RNN import VLMModel
from final_approch.Tokenizer import Tokenizer
from final_approch.CocoCaptionDataset import CocoCaptionDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import os
from tqdm import tqdm

# 配置参数
config = {
    'val_img_dir': 'coco2014/val2014',
    'val_ann_file': 'coco2014/annotations/captions_val2014.json',
    'batch_size': 12,
    'vision_embed_size': 768,
    'max_length': 50,
    'temperature': 0.7,
    'top_p': 0.9
}

# 加载已训练模型的路径
# 替换为您的模型保存路径，例如：20250617_063723
checkpoint_dir = 'checkpoints'
model_path = os.path.join(checkpoint_dir, 'vlm_model_final.pth')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 构建验证集
val_dataset = CocoCaptionDataset(config['val_img_dir'], config['val_ann_file'], transform)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# 初始化tokenizer
tokenizer = Tokenizer(model_name="ImageCap/Qwen3", max_len=config['max_length'])

# 初始化模型
model = VLMModel(vision_embed_size=config['vision_embed_size'], 
                language_model_name="ImageCap/Qwen3",vision_model_path="ImageCap/swinTransformer").to(device)

# 加载模型权重
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # 设置为评估模式

print(f"模型已从 {model_path} 加载")

# 使用与main.py中相同的generate_caption函数
def generate_caption(model, image, tokenizer, max_length=50, device='cpu', temperature=0.7, top_p=0.9):
    model.eval()
    with torch.no_grad():
        # 1. 准备图像特征，并计算出只执行一次的"视觉印记"
        image = image.unsqueeze(0).to(device)
        vision_features = model.encoder(image)
        vision_language_features = model.connector(vision_features)
        print("vision feature ",vision_language_features.shape)
        # vision_projection 的形状是 [1, 1, language_embed_size]，它将作为贯穿始终的图像上下文
        vision_projection = model.visual_projection(vision_language_features)
        print("vision projection ",model.visual_projection(vision_language_features).shape)

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
            print("last",last_token_feature.shape," vision ",vision_projection.shape)
            fused_feature = last_token_feature + vision_projection

            # c. 使用模型的to_vocab层来预测下一个词的概率
            logits = model.to_vocab(fused_feature).squeeze(1)
            print("logits shape",logits.shape,"fused_feature shape",fused_feature.shape)
            # print("no squ shape",model.to_vocab(fused_feature).shape)
            
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
            print("shape filter",filtered_logits.shape)
            
            # 采样下一个token
            # print("shape ",torch.nn.functional.softmax(filtered_logits, dim=-1).shape)
            next_token_id = torch.multinomial(torch.nn.functional.softmax(logits, dim=-1), num_samples=1).item()
            
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

# 设置最大评估样本数
max_eval_samples = 2400

# 进行评估
results = []
sample_count = 0
print("开始生成图像描述...")
for i, (images, captions) in enumerate(tqdm(val_loader)):
    if i>5:
        break
    images = images.to(device)
    for j in range(images.size(0)):
        gt = captions[j]
        pred = generate_caption(
            model, 
            images[j], 
            tokenizer, 
            max_length=config['max_length'],
            device=device,
            temperature=config['temperature'],
            top_p=config['top_p']
        )
        results.append({'gt': gt, 'pred': pred})
        sample_count += 1
        
        # 打印部分结果
        # if len(results) % 1 == 0:
        #     print('gt:', gt)
        #     print('pred:', pred)
        #     print('---')
        
        # 达到最大样本数后停止
        if sample_count >= max_eval_samples:
            print(f"已达到设定的最大评估样本数 {max_eval_samples}")
            break
    
    # 外层循环也需要检查是否应该停止
    if sample_count >= max_eval_samples:
        break

# 保存评估结果
output_path = os.path.join(checkpoint_dir, 'evaluation_results.json')
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"评估完成，结果已保存至 {output_path}")

# 计算BLEU和CIDEr评分 (需要安装pycocoevalcap)
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    
    # 准备评估格式
    gts = {}
    res = {}
    for i, item in enumerate(results):
        gts[i] = [item['gt']]
        res[i] = [item['pred']]
    
    # 计算评分
    scorer_bleu = Bleu(4)
    scorer_cider = Cider()
    
    bleu_scores, _ = scorer_bleu.compute_score(gts, res)
    cider_score, _ = scorer_cider.compute_score(gts, res)
    
    print("BLEU-1: {:.3f}".format(bleu_scores[0]))
    print("BLEU-2: {:.3f}".format(bleu_scores[1]))
    print("BLEU-3: {:.3f}".format(bleu_scores[2]))
    print("BLEU-4: {:.3f}".format(bleu_scores[3]))
    print("CIDEr: {:.3f}".format(cider_score))
    
    # 保存评分结果
    metrics = {
        'bleu1': float(bleu_scores[0]),
        'bleu2': float(bleu_scores[1]), 
        'bleu3': float(bleu_scores[2]),
        'bleu4': float(bleu_scores[3]),
        'cider': float(cider_score)
    }
    
    with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
except ImportError:
    print("注意: 未安装pycocoevalcap包，无法计算BLEU和CIDEr评分")
    print("可通过pip安装: pip install pycocoevalcap")