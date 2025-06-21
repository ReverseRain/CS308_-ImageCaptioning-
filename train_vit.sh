#!/bin/bash

VISION_ENCODER_PATH="ImageCap/models/vit-base-patch16-224"
LANGUAGE_MODEL_PATH="ImageCap/models/qwen3-0.6b"
SAVE_DIR="ImageCap/checkpoints/vit_model"
COCO_DIR="coco2014"

# 创建保存目录
mkdir -p $SAVE_DIR

# 运行训练脚本
python -m ImageCap.train.train \
    --vision_encoder_path $VISION_ENCODER_PATH \
    --language_model_path $LANGUAGE_MODEL_PATH \
    --image_dir $COCO_DIR/train2014 \
    --annotation_file $COCO_DIR/annotations/captions_train2014.json \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --save_dir $SAVE_DIR \
    --log_interval 10 \
    --device cuda:5
    # --max_samples 140000