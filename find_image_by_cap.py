from pycocotools.coco import COCO

# 初始化COCO API
coco = COCO('coco2014/annotations/captions_val2014.json')

# 目标描述（可以是部分或完整文本）
target_caption = "This is an open box containing four cucumbers."

# 查找包含目标描述的标注（模糊匹配）
ann_ids = coco.getAnnIds()  # 获取所有标注ID
annotations = coco.loadAnns(ann_ids)
matched_anns = [ann for ann in annotations if target_caption.lower() in ann['caption'].lower()]

if not matched_anns:
    print("未找到匹配的描述！")
else:
    # 获取对应的图片ID和文件名
    for ann in matched_anns:
        img_id = ann['image_id']
        img_info = coco.loadImgs(img_id)[0]
        print(f"描述: '{ann['caption']}'")
        print(f"文件名: {img_info['file_name']}")
        print(f"图片ID: {img_id}\n")