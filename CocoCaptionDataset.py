import json
import os
from PIL import Image
from torch.utils.data import Dataset


class CocoCaptionDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        self.images = {img['id']: img['file_name'] for img in data['images']}
        self.captions = []
        for ann in data['annotations']:
            self.captions.append((ann['image_id'], ann['caption']))

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_id, caption = self.captions[idx]
        img_path = os.path.join(self.img_dir, self.images[image_id])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, caption
