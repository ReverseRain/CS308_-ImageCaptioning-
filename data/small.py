import json
import os

# 加载原始数据
with open("data/coco/annotations/captions_train2014.json", 'r') as f:
    data = json.load(f)

# 只保留前1000个图像和相关注释
image_ids = set([img['id'] for img in data['images'][:10]])
filtered_annotations = [a for a in data['annotations'] if a['image_id'] in image_ids]
filtered_images = [img for img in data['images'] if img['id'] in image_ids]

# 创建小数据集
small_dataset = {
    'info': data['info'],
    'licenses': data['licenses'],
    'images': filtered_images,
    'annotations': filtered_annotations
}

# 保存
os.makedirs("data/coco/annotations/small", exist_ok=True)
with open("data/coco/annotations/small/captions_train2014_small.json", 'w') as f:
    json.dump(small_dataset, f)

print(f"创建了小数据集，包含{len(filtered_images)}张图像和{len(filtered_annotations)}个注释")