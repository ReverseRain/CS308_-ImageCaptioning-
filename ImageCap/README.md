# ImageCap: Image Captioning with Transformer Vision Encoder and Qwen

A project for image captioning that combines a transformer-based vision encoder with the Qwen-3-0.6B language model using an MLP connector.

## Project Structure

```
ImageCap/
├── data/
│   ├── __init__.py
│   └── coco_dataset.py         # Dataset handling for COCO
├── model/
│   ├── __init__.py
│   ├── image_captioning_model.py   # Main model implementation
│   ├── language_model/         # Qwen LLM loading utilities
│   ├── multimodal_encoder/     # Vision transformer encoder
│   └── multimodal_projector/   # MLP connector
├── train/
│   ├── __init__.py
│   ├── train.py               # Training script
│   └── trainer.py             # Trainer class
├── eval/
│   ├── __init__.py
│   └── evaluate.py            # Evaluation script
└── main.py                    # Main entry point
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- pycocoevalcap
- PIL
- tqdm

Install dependencies:

```bash
pip install torch torchvision transformers pillow tqdm pycocoevalcap
```

## Usage

### Training

To train the model on COCO dataset:

```bash
python -m ImageCap.main train \
    --vision_tower="google/vit-base-patch16-224" \
    --language_model_path="Qwen/Qwen1.5-0.6B" \
    --train_annotation_file="path/to/coco/annotations/captions_train2014.json" \
    --val_annotation_file="path/to/coco/annotations/captions_val2014.json" \
    --image_dir="path/to/coco/images" \
    --output_dir="./outputs" \
    --batch_size=16 \
    --learning_rate=2e-5 \
    --epochs=10
```

### Evaluation

To evaluate a trained model on COCO dataset:

```bash
python -m ImageCap.main eval \
    --model_path="./outputs/best_model" \
    --vision_tower="google/vit-base-patch16-224" \
    --language_model_path="Qwen/Qwen1.5-0.6B" \
    --annotation_file="path/to/coco/annotations/captions_val2014.json" \
    --image_dir="path/to/coco/images" \
    --output_file="predictions.json"
```

## Model Components

1. **Vision Encoder**: Transformer-based vision encoder (default: ViT-Base)
2. **Language Model**: Qwen-3-0.6B language model
3. **Connector**: MLP-based projector to connect the vision encoder with the language model

## Training Process

The training process follows these steps:

1. Process images through the vision encoder to extract features
2. Project these features to the language model dimension using the MLP connector
3. Feed the projected features to the language model
4. Generate captions in an autoregressive manner
5. Calculate loss on the generated captions
6. Backpropagate and update the parameters

By default, only the MLP projector and the language model's output layer are trained, while the vision encoder and most of the language model are kept frozen.

## Evaluation Metrics

The evaluation script calculates the following metrics:
- BLEU-1/2/3/4
- METEOR
- ROUGE-L
- CIDEr

## Acknowledgements

This project is inspired by the LLaVA project: https://github.com/haotian-liu/LLaVA 