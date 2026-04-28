"""
Head CT Hemorrhage veri hattı: grup/hasta bazlı bölme, augmentasyon, DataLoader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df


def resolve_image_path(head_ct_dir: Path, image_id: int) -> Path:
    """Kaggle varyantlarına göre dosya adını bulur."""
    head_ct_dir = Path(head_ct_dir)
    candidates = [
        head_ct_dir / f"{image_id}.png",
        head_ct_dir / f"{image_id:03d}.png",
        head_ct_dir / f"{image_id:06d}.png",
        head_ct_dir / f"{image_id}.jpg",
        head_ct_dir / f"{image_id:03d}.jpg",
    ]
    for p in candidates:
        if p.is_file():
            return p
    # Son çare: eşleşen tek dosya
    matches = list(head_ct_dir.glob(f"{image_id}.*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"ID {image_id} için görüntü bulunamadı. Denenen örnek: {candidates[0]}"
    )


def load_labels_dataframe(
    labels_csv: Path = config.LABELS_CSV,
    head_ct_dir: Path = config.HEAD_CT_DIR,
    group_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    CSV ve head_ct klasöründen birleşik tablo üretir.

    group_column: CSV'de hasta/grup sütunu adı (ör. 'patient_id'). Yoksa her satır
    kendi grubu olarak `id` kullanılır (aynı kesitin iki sette olması engellenir).
    Çoklu kesit aynı hastadaysa CSV'ye `patient_id` ekleyip burada verin.
    """
    df = _strip_columns(pd.read_csv(labels_csv))
    if "id" not in df.columns:
        raise ValueError("labels.csv içinde 'id' sütunu gerekli.")
    label_col = "hemorrhage" if "hemorrhage" in df.columns else None
    if label_col is None:
        for c in df.columns:
            if c != "id":
                label_col = c
                break
    if label_col is None:
        raise ValueError("Etiket sütunu bulunamadı (hemorrhage veya id dışı bir sütun).")

    df["label"] = df[label_col].astype(int)
    df["group_id"] = df[group_column].astype(str) if group_column and group_column in df.columns else df["id"].astype(str)
    df["image_path"] = [resolve_image_path(head_ct_dir, int(i)) for i in df["id"]]
    return df


def split_train_val_test_by_group(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    group_col: str = "group_id",
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Grup/hasta düzeyinde stratified bölme: aynı grup yalnızca tek bir sette.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train + val + test oranları 1 olmalı.")

    groups = (
        df.groupby(group_col, as_index=False)[label_col]
        .agg("first")
        .rename(columns={label_col: "group_label"})
    )

    idx_all = np.arange(len(groups))
    tv_idx, te_idx = train_test_split(
        idx_all,
        test_size=test_ratio,
        stratify=groups["group_label"],
        random_state=random_state,
    )
    train_val = groups.iloc[tv_idx].reset_index(drop=True)
    test_ids = set(groups.iloc[te_idx][group_col])

    rel_val = val_ratio / (train_ratio + val_ratio)
    idx_tv = np.arange(len(train_val))
    tr_idx, va_idx = train_test_split(
        idx_tv,
        test_size=rel_val,
        stratify=train_val["group_label"],
        random_state=random_state,
    )
    train_ids = set(train_val.iloc[tr_idx][group_col])
    val_ids = set(train_val.iloc[va_idx][group_col])

    train_df = df[df[group_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[group_col].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df


def build_transforms(
    image_size: int = config.IMAGE_SIZE,
    train: bool = False,
) -> Callable:
    """Train: augmentasyon + resize + normalize. Val/Test: sadece resize + normalize."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((image_size + 16, image_size + 16)),
                transforms.RandomCrop(image_size),
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class HeadCTDataset(Dataset):
    """Dosya yolu + etiket; opsiyonel transform."""

    def __init__(self, frame: pd.DataFrame, transform: Optional[Callable] = None):
        self.paths = frame["image_path"].tolist()
        self.labels = frame["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = Path(self.paths[idx])
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = int(self.labels[idx])
        return image, label


def make_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 16,
    num_workers: int = 0,
    image_size: int = config.IMAGE_SIZE,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf = build_transforms(image_size, train=True)
    eval_tf = build_transforms(image_size, train=False)

    train_loader = DataLoader(
        HeadCTDataset(train_df, train_tf),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        HeadCTDataset(val_df, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        HeadCTDataset(test_df, eval_tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def build_default_split(
    group_column: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple[DataLoader, DataLoader, DataLoader]]:
    """Etiketleri yükle, böl, varsayılan batch ile DataLoader döndür."""
    full = load_labels_dataframe(group_column=group_column)
    train_df, val_df, test_df = split_train_val_test_by_group(
        full, random_state=random_state
    )
    loaders = make_dataloaders(train_df, val_df, test_df)
    return train_df, val_df, test_df, loaders
