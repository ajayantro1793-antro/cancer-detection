"""
dataset.py  —  Classification Dataset
─────────────────────────────────────────────────────────────────────────────
Defines the PyTorch Dataset class for the IQ-OTH/NCCD lung cancer
classification task.

The dataset has 3 classes:
  0 = Normal    (healthy lung)
  1 = Benign    (non-cancerous nodule)
  2 = Malignant (cancerous nodule)

We apply two different augmentation pipelines:
  - train_transform:  aggressive augmentation to improve generalisation
  - val_transform:    only normalisation — no augmentation during evaluation

We also compute class weights to handle class imbalance (Normal scans
typically outnumber Malignant scans in real datasets). The weighted
cross-entropy loss uses these weights to penalise misclassification of
the minority (Malignant) class more heavily.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import yaml


# ── Class label mapping ────────────────────────────────────────────────────
CLASS_NAMES  = ["Normal", "Benign", "Malignant"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


# ── Albumentations augmentation pipelines ─────────────────────────────────

def get_train_transform(image_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline.
    Albumentations is used rather than torchvision transforms because it
    applies the same spatial transformation simultaneously to the image AND
    any bounding box annotations — critical for consistency.

    The augmentations simulate real-world CT scan variation:
      - Flips:             different patient orientations
      - Rotation ±15°:     slight positioning differences between scans
      - Brightness/contrast: variation between CT machines
      - Gaussian noise:    scanner noise at different radiation doses
      - Elastic transform: tissue deformation (simulates breathing motion)
      - Grid distortion:   simulates slight geometric artefacts
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(p=0.4),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 512) -> A.Compose:
    """
    Validation / test augmentation pipeline.
    Only resize and normalise — no random transforms. Evaluation should be
    deterministic so we can reliably compare runs.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


# ── Dataset class ──────────────────────────────────────────────────────────

class LungCancerDataset(Dataset):
    """
    PyTorch Dataset for IQ-OTH/NCCD lung cancer classification.

    Expects data_dir to contain one sub-folder per class:
        data_dir/
            Normal/
                img001.png
                img002.png
                ...
            Benign/
                ...
            Malignant/
                ...
    """

    def __init__(self,
                 image_paths: List[str],
                 labels: List[int],
                 transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Load image as RGB (cv2 loads BGR by default, so we convert)
        img_bgr = cv2.imread(self.image_paths[idx])
        if img_bgr is None:
            raise RuntimeError(f"Failed to load: {self.image_paths[idx]}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label = self.labels[idx]

        # Apply augmentation / normalisation
        if self.transform:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]  # shape: (3, H, W) float32 tensor
        else:
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        return img_tensor, label


def build_dataset_splits(data_dir: str,
                          train_split: float = 0.80,
                          val_split: float   = 0.10,
                          test_split: float  = 0.10,
                          image_size: int    = 512,
                          seed: int          = 42
                          ) -> Tuple[Dataset, Dataset, Dataset, List[float]]:
    """
    Scan data_dir, collect all image paths and labels, split into
    train / val / test sets, and return three Dataset objects plus
    class weights for the loss function.

    The stratified split ensures every class is proportionally
    represented in all three sets — critical when one class is rarer.
    """
    all_paths, all_labels = [], []

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = Path(data_dir) / class_name
        if not class_dir.exists():
            print(f"Warning: directory not found — {class_dir}")
            continue

        img_files = list(class_dir.glob("*.png")) + \
                    list(class_dir.glob("*.jpg")) + \
                    list(class_dir.glob("*.jpeg"))

        all_paths  += [str(p) for p in img_files]
        all_labels += [class_idx] * len(img_files)
        print(f"  {class_name}: {len(img_files)} images")

    # ── Stratified split: train → (val + test) → (val, test) ──────────────
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=(1 - train_split),
        stratify=all_labels,
        random_state=seed
    )
    val_ratio = val_split / (val_split + test_split)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=seed
    )

    print(f"\nSplit sizes — Train: {len(train_paths)} | "
          f"Val: {len(val_paths)} | Test: {len(test_paths)}")

    # ── Compute class weights to handle imbalance ──────────────────────────
    # Weight for class i = total_samples / (num_classes * count_of_class_i)
    # Rare classes get a higher weight so the loss penalises their
    # misclassification more heavily.
    counts = np.bincount(all_labels, minlength=len(CLASS_NAMES))
    total  = len(all_labels)
    class_weights = [total / (len(CLASS_NAMES) * c) if c > 0 else 1.0
                     for c in counts]
    print(f"Class weights: {dict(zip(CLASS_NAMES, [round(w, 3) for w in class_weights]))}")

    train_ds = LungCancerDataset(train_paths, train_labels,
                                  transform=get_train_transform(image_size))
    val_ds   = LungCancerDataset(val_paths,   val_labels,
                                  transform=get_val_transform(image_size))
    test_ds  = LungCancerDataset(test_paths,  test_labels,
                                  transform=get_val_transform(image_size))

    return train_ds, val_ds, test_ds, class_weights


def get_dataloaders(data_dir: str,
                    batch_size: int  = 16,
                    image_size: int  = 512,
                    num_workers: int = 4
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[float]]:
    """
    Convenience wrapper that returns train/val/test DataLoaders directly.

    num_workers=4 is appropriate for most Windows/Linux setups.
    On Windows, set num_workers=0 if you encounter pickling errors.
    """
    train_ds, val_ds, test_ds, class_weights = build_dataset_splits(
        data_dir, image_size=image_size
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=num_workers,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                               shuffle=False, num_workers=num_workers,
                               pin_memory=True)

    return train_loader, val_loader, test_loader, class_weights


