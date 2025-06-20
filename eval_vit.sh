#!/bin/bash

CHECKPOINT_PATH="ImageCap/checkpoints/vit_model/best_model.pt"
VISION_ENCODER_PATH="ImageCap/models/vit-base-patch16-224"
LANGUAGE_MODEL_PATH="ImageCap/models/qwen3-0.6b"
COCO_DIR="coco2014"
OUTPUT_FILE="ImageCap/checkpoints/vit_model/evaluation_results.json"

# 运行评估脚本
python -m ImageCap.evaluate \
    --checkpoint_path $CHECKPOINT_PATH \
    --vision_encoder_path $VISION_ENCODER_PATH \
    --language_model_path $LANGUAGE_MODEL_PATH \
    --coco_dir $COCO_DIR \
    --split val \
    --output_file $OUTPUT_FILE \
    --batch_size 16 \
    --limit 100 \
    --device cuda:0 \
    --debug