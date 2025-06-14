# ImageCap: 图像描述生成系统

基于视觉-语言模型的图像描述生成系统，使用CLIP视觉编码器和大型语言模型。

## 项目结构

```
imagecap/
├── __init__.py             # 包初始化
├── main.py                 # 主入口
├── model/                  # 模型定义
│   ├── __init__.py
│   ├── builder.py          # 模型构建器
│   ├── imagecap_arch.py    # 主模型架构
│   ├── vision_encoder/     # 视觉编码器
│   │   ├── __init__.py
│   │   └── clip_encoder.py
│   ├── language_model/     # 语言模型
│   │   ├── __init__.py
│   │   └── utils.py
│   └── multimodal_projector/ # 多模态连接器
│       ├── __init__.py
│       └── projector.py
├── train/                  # 训练模块
│   ├── __init__.py
│   └── trainer.py
├── eval/                   # 评估模块
│   ├── __init__.py
│   └── evaluator.py
└── data/                   # 数据处理
    ├── __init__.py
    └── dataset.py
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

数据应按以下格式组织：

```
data/
├── images/                 # 图像目录
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── captions.json           # 描述文件
```

`captions.json` 文件格式：

```json
[
  {
    "image": "image1.jpg",
    "caption": "这是图像1的描述"
  },
  {
    "image": "image2.jpg",
    "caption": "这是图像2的描述"
  },
  ...
]
```

## 使用方法

### 训练模型

```bash
python run.py train --data_path ./data --output_dir ./outputs --batch_size 4 --num_epochs 3
```

主要参数：

- `--vision_model_name`: 视觉模型名称，默认为 "openai/clip-vit-base-patch16"
- `--language_model_name`: 语言模型名称，默认为 "Qwen/Qwen3-0.6B"
- `--data_path`: 数据路径
- `--output_dir`: 输出目录
- `--batch_size`: 批量大小
- `--num_epochs`: 训练轮数
- `--max_samples`: 最大样本数，用于限制训练样本数量
- `--freeze_vision_model`: 是否冻结视觉模型
- `--freeze_language_model_except_layers`: 冻结语言模型除了最后几层

### 评估模型

```bash
python run.py eval --model_path ./outputs/final --test_dir ./test_images --output_file ./eval_results.json
```

主要参数：

- `--model_path`: 模型路径
- `--test_dir`: 测试图像目录
- `--output_file`: 输出文件路径
- `--prompt`: 提示文本

### 生成描述

```bash
python run.py generate --model_path ./outputs/final --image_path ./test_images/image.jpg
```

主要参数：

- `--model_path`: 模型路径
- `--image_path`: 图像路径
- `--prompt`: 提示文本
- `--max_length`: 生成的最大长度

## 模型架构

ImageCap模型由三部分组成：

1. **视觉编码器**：使用CLIP视觉模型提取图像特征
2. **多模态投影器**：将视觉特征投影到语言空间
3. **语言模型**：生成描述文本

## 训练过程

1. 加载预训练的视觉编码器和语言模型
2. 冻结视觉编码器参数
3. 可选地冻结部分语言模型参数
4. 训练投影器和未冻结的语言模型参数 