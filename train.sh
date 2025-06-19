CHECKPOINT_PATH="ImageCap/checkpoints/best_model_all_feature.pt"
# CHECKPOINT_PATH="checkpoints/vlm_model_final.pth"
VISION_ENCODER_PATH="ImageCap/swinTransformer"
LANGUAGE_MODEL_PATH="ImageCap/Qwen3"
COCO_DIR="coco2014"  # 使用正确的COCO数据集路径
OUTPUT_FILE="evaluation_results_all_feature.json"

# /home/wangdx_lab/cse12210626/.conda/envs/cs308/bin/python -m ImageCap.train.train --vision_encoder_path ImageCap/swinTransformer --language_model_path ImageCap/Qwen3 --annotation_file coco2014/annotations/captions_train2014.json --image_dir coco2014/train2014 

# /home/wangdx_lab/cse12210626/.conda/envs/cs308/bin/python -m final_approch.main --vision_encoder_path ImageCap/swinTransformer --language_model_path ImageCap/Qwen3 --annotation_file coco2014/annotations/captions_train2014.json --image_dir coco2014/train2014 

python -m ImageCap.evaluate \
    --checkpoint_path $CHECKPOINT_PATH \
    --vision_encoder_path $VISION_ENCODER_PATH \
    --language_model_path $LANGUAGE_MODEL_PATH \
    --coco_dir $COCO_DIR \
    --split val \
    --output_file $OUTPUT_FILE \
    --batch_size 16 \
    --limit 100 \
    --debug

# /home/wangdx_lab/cse12210626/.conda/envs/cs308/bin/python -m final_approch.eval
    