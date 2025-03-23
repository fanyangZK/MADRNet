import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2

# 自定义数据集
# 修改自定义数据集类
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        classes = [d for d in os.listdir(root_dir) if
                   os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')]
        print(classes)

        for cls_idx, cls_name in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.startswith('.'): continue
                img_path = os.path.join(cls_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    ratio = self._calculate_ratio(img)
                    self.samples.append((img_path, cls_idx, float(ratio)))
                except Exception as e:
                    print(f"Error loading {img_path}: {str(e)}")

    def _calculate_ratio(self, img):
        img_np = np.array(img)
        if img_np.size == 0:
            return 1.5
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return h / w if w != 0 else 1.5
        return 1.5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, ratio = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label), torch.tensor(ratio)