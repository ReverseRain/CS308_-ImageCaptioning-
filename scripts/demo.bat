@echo off
REM 图像描述生成模型演示脚本 - Windows版本

REM 设置默认参数
set MODEL_PATH=outputs/caption_model/best_model
set IMAGE_PATH=
set PROMPT=请为这张图片生成描述：
set MAX_LENGTH=50
set OUTPUT_PATH=

REM 解析命令行参数
:parse
if "%~1"=="" goto check
if /i "%~1"=="--model_path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--image_path" (
    set IMAGE_PATH=%~2
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
if /i "%~1"=="--max_length" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto parse
)
if /i "%~1"=="--output_path" (
    set OUTPUT_PATH=%~2
    shift
    shift
    goto parse
)
shift
goto parse

:check
REM 检查必需参数
if "%IMAGE_PATH%"=="" (
    echo 错误：请提供图像路径 (--image_path)
    goto end
)

REM 打印演示配置
echo ======================================
echo 图像描述生成模型演示
echo ======================================
echo 模型路径: %MODEL_PATH%
echo 图像路径: %IMAGE_PATH%
echo 提示文本: %PROMPT%
echo 最大长度: %MAX_LENGTH%
if not "%OUTPUT_PATH%"=="" echo 输出路径: %OUTPUT_PATH%
echo ======================================

REM 准备命令行
set CMD=python %~dp0demo.py --model_path "%MODEL_PATH%" --image_path "%IMAGE_PATH%" --prompt "%PROMPT%" --max_length %MAX_LENGTH%
if not "%OUTPUT_PATH%"=="" set CMD=%CMD% --output_path "%OUTPUT_PATH%"

REM 运行演示脚本
%CMD%

echo 演示完成!

:end
pause 