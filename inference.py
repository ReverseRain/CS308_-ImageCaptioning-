from CNN_RNN import CNNEncoder, RNNDecoder
from CocoCaptionDataset import CocoCaptionDataset
from Tokenizer import Tokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import json
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


def load_model(checkpoint_dir, embed_size, hidden_size, vocab_size, device):
    encoder = CNNEncoder(embed_size).to(device)
    decoder = RNNDecoder(embed_size, hidden_size, vocab_size).to(device)
    
    encoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'encoder_final.pth')))
    decoder.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'decoder_final.pth')))
    
    return {'encoder': encoder, 'decoder': decoder}


def compute_scores(results):
    gts = {}  # ground truth
    res = {}  # predictions
    
    for idx, item in enumerate(results):
        gts[idx] = [item['ground_truth']]  # ground truth needs to be a list of references
        res[idx] = [item['prediction']]     # prediction is a single string
    
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Cider(), "CIDEr")
    ]

    scores = {}
    for scorer, method in scorers:
        score, scores_per_caption = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                scores[m] = sc
        else:
            scores[method] = score

    return scores


def inference(model, dataloader, tokenizer, device, max_length=20):
    model['encoder'].eval()
    model['decoder'].eval()
    results = []
    
    with torch.no_grad():
        for images, captions in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)
            features = model['encoder'](images)
            
            sampled_ids = []
            inputs = features.unsqueeze(1)
            states = None
            
            for _ in range(max_length):
                hiddens, states = model['decoder'].lstm(inputs, states)
                outputs = model['decoder'].linear(hiddens.squeeze(1))
                _, predicted = outputs.max(1)
                sampled_ids.append(predicted)
                inputs = model['decoder'].embed(predicted).unsqueeze(1)
            
            sampled_ids = torch.stack(sampled_ids, 1)
            
            for i in range(images.size(0)):
                gt = captions[i]
                pred = tokenizer.decode(sampled_ids[i].cpu().numpy())

                results.append({
                    'ground_truth': gt,
                    'prediction': pred
                })
                
    return results


if __name__ == "__main__":
    config = {
        'checkpoint_dir': 'checkpoints/20250615_045705',  # 需要根据实际检查点目录修改
        'val_img_dir': 'coco2014/val2014',
        'val_ann_file': 'coco2014/annotations/captions_val2014.json',
        'train_img_dir': 'coco2014/train2014',
        'train_ann_file': 'coco2014/annotations/captions_train2014.json',
        'batch_size': 64,
        'embed_size': 256,
        'hidden_size': 512,
        'device': 'cuda:6'
    }
    
    output_dir = os.path.join(config['checkpoint_dir'], 'inference_result')
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    print("Loading dataset...")
    train_dataset = CocoCaptionDataset(config['train_img_dir'], config['train_ann_file'], transform)
    val_dataset = CocoCaptionDataset(config['val_img_dir'], config['val_ann_file'], transform)
    
    print("Building tokenizer...")
    tokenizer = Tokenizer([c for _, c in train_dataset.captions])
    
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    print("Loading model...")
    model = load_model(config['checkpoint_dir'], config['embed_size'], config['hidden_size'], 
                      len(tokenizer.tokenizer.vocab), device)
    
    print("Starting inference...")
    results = inference(model, val_loader, tokenizer, device)
    
    print("Computing evaluation metrics...")
    scores = compute_scores(results)
    
    output = {
        'results': results,
        'metrics': {
            'bleu1': scores['Bleu_1'],
            'bleu2': scores['Bleu_2'],
            'bleu3': scores['Bleu_3'],
            'bleu4': scores['Bleu_4'],
            'cider': scores['CIDEr']
        }
    }
    
    output_file = os.path.join(output_dir, 'inference_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(output['metrics'], f, ensure_ascii=False, indent=2)
    
    print(f"Inference completed! Results saved to: {output_file}")
    print(f"Evaluation metrics saved to: {metrics_file}")
    print("\nEvaluation Metrics:")
    for metric, value in output['metrics'].items():
        print(f"{metric}: {value:.4f}")