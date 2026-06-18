"""
SteelBallDataset：讀取 train/val/test.txt，套用 Albumentations transform
"""
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

NUM_CLASSES = 13
IMAGE_SIZE  = 512

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def get_train_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.CLAHE(p=0.2),
        A.OneOf([A.GaussianBlur(p=1), A.MotionBlur(p=1)], p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class SteelBallDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.samples = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                self.samples.append((parts[0], parts[1]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))   # uint8, pixel = class id

        if self.transform:
            out  = self.transform(image=img, mask=mask)
            img  = out["image"]   # (3,H,W) float tensor
            mask = out["mask"]    # (H,W) uint8 tensor

        return img, mask.long()
