#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan      # 作业提交的指定分区队列为titan
#SBATCH --qos=titan            # 指定作业的QOS
#SBATCH -J myFirstGPUJob       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=6    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:2 


CHECKPOINT_PATH="ImageCap/checkpoints/best_model_gpt_epoch10.pt"
VISION_ENCODER_PATH="ImageCap/swinTransformer"
LANGUAGE_MODEL_PATH="ImageCap/gpt2zh"
COCO_DIR="coco2014"  # 使用正确的COCO数据集路径
OUTPUT_FILE="evaluation_results_gpt_epoch10.json"

/home/wangdx_lab/cse12210626/.conda/envs/cs308/bin/python -m ImageCap.train.train --vision_encoder_path ImageCap/swinTransformer --language_model_path ImageCap/Qwen3 --annotation_file coco2014/annotations/captions_train2014.json --image_dir coco2014/train2014 

# python -m ImageCap.evaluate \
#     --checkpoint_path $CHECKPOINT_PATH \
#     --vision_encoder_path $VISION_ENCODER_PATH \
#     --language_model_path $LANGUAGE_MODEL_PATH \
#     --coco_dir $COCO_DIR \
#     --split val \
#     --output_file $OUTPUT_FILE \
#     --batch_size 16 \
#     --limit 10 \
#     --debug
    