# 图像描述生成项目

本项目实现了一个基于视觉-语言模型的图像描述生成系统。该系统结合了视觉编码器和大型语言模型，能够为给定图像生成相关且准确的自然语言描述。

## 项目结构

```
├── config/               # 配置文件
├── data/                 # 数据目录
├── LLaVA/                # LLaVA参考实现
├── models/               # 模型定义
│   ├── __init__.py       
│   ├── model.py          # 主模型文件
│   ├── config.py         # 模型配置
│   ├── projector.py      # 连接器实现
│   └── vision_encoder.py # 视觉编码器
├── preprocessing/        # 数据预处理脚本
├── scripts/              # 训练和评估脚本
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   └── demo.py           # 演示脚本
├── utils/                # 工具函数
│   ├── data.py           # 数据加载
│   └── metrics.py        # 评估指标
└── requirements.txt      # 依赖项
```

## 模型架构

本项目的图像描述生成模型包括三个主要部分：

1. **视觉编码器**：使用基于Transformer的CLIP视觉模型提取图像特征
2. **连接器**：使用简单的MLP（多层感知机）将视觉特征映射到语言模型的特征空间
3. **语言模型**：使用Qwen系列语言模型生成文本描述

## 安装

1. 克隆代码库：

```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 安装评估指标工具：

```bash
pip install git+https://github.com/salaniz/pycocoevalcap.git
```

## 数据准备

本项目使用COCO数据集进行训练和评估。您需要下载COCO图像和注释文件：

1. 下载COCO图像：
   - [COCO训练集](http://images.cocodataset.org/zips/train2014.zip)
   - [COCO验证集](http://images.cocodataset.org/zips/val2014.zip)

2. 下载COCO注释文件：
   - [COCO注释文件](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)

## 训练

使用以下命令训练模型：

```bash
python scripts/train.py \
    --vision_model "openai/clip-vit-base-patch16" \
    --language_model "Qwen/Qwen1.5-0.5B" \
    --projector_type "mlp" \
    --data_dir "/path/to/coco/images" \
    --ann_file "/path/to/captions_train2014.json" \
    --batch_size 8 \
    --epochs 3 \
    --output_dir "outputs"
```

## 评估

使用以下命令评估模型：

```bash
python scripts/evaluate.py \
    --model_path "outputs/best_model" \
    --data_dir "/path/to/coco/images" \
    --ann_file "/path/to/captions_val2014.json"
```

## 演示

使用以下命令演示模型：

```bash
python scripts/demo.py \
    --model_path "outputs/best_model" \
    --image_path "path/to/image.jpg"
```

## 不同模型的对比

在本项目中，您可以尝试不同的视觉编码器和语言模型组合：

- 视觉编码器（Transformer-based）：
  - CLIP-ViT (openai/clip-vit-base-patch16)
  - CLIP-ViT-Large (openai/clip-vit-large-patch14)

- 语言模型：
  - Qwen-0.5B (Qwen/Qwen1.5-0.5B)
  - Qwen-1.8B (Qwen/Qwen1.5-1.8B)
  - Qwen-0.6B (Qwen/Qwen2-0.5B)

- 连接器：
  - MLP（默认）
  - Linear