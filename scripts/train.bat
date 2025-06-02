@echo off
REM 图像描述生成模型训练脚本 - Windows版本

REM 设置默认参数
set VISION_MODEL=openai/clip-vit-base-patch16
set LANGUAGE_MODEL=Qwen/Qwen1.5-0.5B
set PROJECTOR_TYPE=mlp
set DATA_DIR=data/coco/train2014
set ANN_FILE=data/coco/annotations/captions_train2014.json
set BATCH_SIZE=8
set EPOCHS=3
set LR=5e-5
set OUTPUT_DIR=outputs/caption_model

REM 解析命令行参数
:parse
if "%~1"=="" goto execute
if /i "%~1"=="--vision_model" (
    set VISION_MODEL=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--language_model" (
    set LANGUAGE_MODEL=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--projector_type" (
    set PROJECTOR_TYPE=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--data_dir" (
    set DATA_DIR=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--ann_file" (
    set ANN_FILE=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--batch_size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--lr" (
    set LR=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--output_dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto parse
)
shift
goto parse

:execute
REM 打印训练配置
echo ======================================
echo 图像描述生成模型训练
echo ======================================
echo 视觉模型: %VISION_MODEL%
echo 语言模型: %LANGUAGE_MODEL%
echo 连接器类型: %PROJECTOR_TYPE%
echo 数据目录: %DATA_DIR%
echo 注释文件: %ANN_FILE%
echo 批量大小: %BATCH_SIZE%
echo 训练轮数: %EPOCHS%
echo 学习率: %LR%
echo 输出目录: %OUTPUT_DIR%
echo ======================================

REM 运行训练脚本
python %~dp0train.py ^
    --vision_model "%VISION_MODEL%" ^
    --language_model "%LANGUAGE_MODEL%" ^
    --projector_type "%PROJECTOR_TYPE%" ^
    --data_dir "%DATA_DIR%" ^
    --ann_file "%ANN_FILE%" ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --output_dir "%OUTPUT_DIR%"

echo 训练完成!
pause 