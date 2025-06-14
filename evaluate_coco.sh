#!/bin/bash


# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=5  # 可以根据需要更改

# 模型和数据路径
CHECKPOINT_PATH="ImageCap/checkpoints/best_model.pt"
VISION_ENCODER_PATH="ImageCap/models/vit-base-patch16-224"
LANGUAGE_MODEL_PATH="ImageCap/models/qwen3-0.6b"
COCO_DIR="ImageCap/data/coco"  # 使用正确的COCO数据集路径
OUTPUT_FILE="evaluation_results.json"

# 先尝试运行一个小批量进行调试
echo "调试模式：评估少量图片..."
python -m ImageCap.evaluate \
    --checkpoint_path $CHECKPOINT_PATH \
    --vision_encoder_path $VISION_ENCODER_PATH \
    --language_model_path $LANGUAGE_MODEL_PATH \
    --coco_dir $COCO_DIR \
    --split val \
    --limit 10 \
    --output_file "debug_evaluation_results.json" \
    --batch_size 5 \
    --debug

if [ $? -eq 0 ]; then
    echo "调试运行成功！开始完整评估..."
    
    # 运行完整评估
    python -m ImageCap.evaluate \
        --checkpoint_path $CHECKPOINT_PATH \
        --vision_encoder_path $VISION_ENCODER_PATH \
        --language_model_path $LANGUAGE_MODEL_PATH \
        --coco_dir $COCO_DIR \
        --split val \
        --output_file $OUTPUT_FILE \
        --batch_size 16
    
    # 显示结果摘要
    echo "评估完成，结果已保存至 $OUTPUT_FILE"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "评估指标摘要:"
        grep -A 5 "scores" $OUTPUT_FILE
    fi
else
    echo "调试运行失败，请检查错误信息"
fi 