@echo off
REM 图像描述生成模型评估脚本 - Windows版本

REM 设置默认参数
set MODEL_PATH=outputs/caption_model/best_model
set DATA_DIR=data/coco/val2014
set ANN_FILE=data/coco/annotations/captions_val2014.json
set OUTPUT_DIR=outputs/evaluation_results
set BATCH_SIZE=16
set MAX_LENGTH=50
set PROMPT=请为这张图片生成描述：

REM 解析命令行参数
:parse
if "%~1"=="" goto execute
if /i "%~1"=="--model_path" (
    set MODEL_PATH=%~2
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
if /i "%~1"=="--output_dir" (
    set OUTPUT_DIR=%~2
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
if /i "%~1"=="--max_length" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--prompt" (
    set PROMPT=%~2
    shift
    shift
    goto parse
)
shift
goto parse

:execute
REM 打印评估配置
echo ======================================
echo 图像描述生成模型评估
echo ======================================
echo 模型路径: %MODEL_PATH%
echo 数据目录: %DATA_DIR%
echo 注释文件: %ANN_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 批量大小: %BATCH_SIZE%
echo 最大长度: %MAX_LENGTH%
echo 提示文本: %PROMPT%
echo ======================================

REM 运行评估脚本
python %~dp0evaluate.py ^
    --model_path "%MODEL_PATH%" ^
    --data_dir "%DATA_DIR%" ^
    --ann_file "%ANN_FILE%" ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size %BATCH_SIZE% ^
    --max_length %MAX_LENGTH% ^
    --prompt "%PROMPT%"

echo 评估完成!
pause 