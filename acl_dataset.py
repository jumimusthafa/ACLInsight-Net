# -*- coding: utf-8 -*-
"""
acl_dataset.py — ACLInsight-Net
MRI-friendly augmentations, MRNetSliceDataset, and per-plane DataLoaders.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from acl_config import (
    IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS, PLANES,
    TRAIN_DIR, VALID_DIR, TRAIN_ACL_CSV, VALID_ACL_CSV
)

# ── Augmentations (MRI-friendly) ──────────────────────────────────────────
# ImageNet mean/std — standard for pretrained models on grayscale-as-RGB input
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=8, p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
    # RandomRotate90 removed — anatomically incorrect for knee MRI
    A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=1.0),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=_MEAN, std=_STD, max_pixel_value=1.0),
    ToTensorV2()
])

print("Augmentations defined.")


# ── Dataset ───────────────────────────────────────────────────────────────
class MRNetSliceDataset(Dataset):
    """
    Slice-level dataset for MRNet .npy volumes.
    Volumes are cached in memory after the first load to avoid repeated I/O.
    Each sample: (3×H×W tensor, label, case_id).
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        transform=None,
        planes=('axial', 'coronal', 'sagittal')
    ):
        self.transform = transform
        self._volume_cache: dict = {}          # in-memory volume cache

        df = pd.read_csv(csv_path, header=None, names=['case', 'label'])
        labels = dict(zip(df['case'].astype(int), df['label'].astype(int)))

        self.samples = []
        for case, label in labels.items():
            for plane in planes:
                npy_path = os.path.join(root_dir, plane, f"{case:04d}.npy")
                if not os.path.exists(npy_path):
                    continue
                if npy_path not in self._volume_cache:
                    self._volume_cache[npy_path] = np.load(npy_path)   # (S, H, W)
                n_slices = self._volume_cache[npy_path].shape[0]
                for s in range(n_slices):
                    self.samples.append((npy_path, s, label, case))

        split = os.path.basename(root_dir)
        print(f"[{split}] {len(self.samples):,} slices | "
              f"{len(labels)} exams | {len(self._volume_cache)} volumes cached")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, slice_idx, label, case = self.samples[idx]

        # Read from cache — no disk I/O after first epoch
        img = self._volume_cache[npy_path][slice_idx].astype(np.float32) / 255.0

        # Grayscale → 3-channel for pretrained backbone
        img = np.stack([img, img, img], axis=-1)   # (H, W, 3)

        if self.transform:
            img = self.transform(image=img)['image']

        return img, torch.tensor(label, dtype=torch.float32), case


# ── Build per-plane datasets & loaders ───────────────────────────────────
def build_loaders():
    """Returns (train_loaders, val_loaders) dicts keyed by plane name."""
    train_datasets = {
        plane: MRNetSliceDataset(
            root_dir=TRAIN_DIR, csv_path=TRAIN_ACL_CSV,
            transform=train_transform, planes=(plane,)
        )
        for plane in PLANES
    }
    val_datasets = {
        plane: MRNetSliceDataset(
            root_dir=VALID_DIR, csv_path=VALID_ACL_CSV,
            transform=val_transform, planes=(plane,)
        )
        for plane in PLANES
    }

    _kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                   pin_memory=True, persistent_workers=True)

    train_loaders = {
        p: DataLoader(train_datasets[p], shuffle=True, drop_last=True, **_kwargs)
        for p in PLANES
    }
    val_loaders = {
        p: DataLoader(val_datasets[p], shuffle=False, **_kwargs)
        for p in PLANES
    }

    print("DataLoaders created per plane.")
    return train_datasets, val_datasets, train_loaders, val_loaders


if __name__ == '__main__':
    train_datasets, val_datasets, train_loaders, val_loaders = build_loaders()
