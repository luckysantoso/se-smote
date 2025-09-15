# dataset.py
import io
import zipfile
import pickle
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class TransformedSubset(Dataset):
    """
    Membungkus torch.utils.data.Subset dan menerapkan transform pada image.
    """
    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]  # x masih PIL Image dari ImageFolder
        if self.transform is not None:
            x = self.transform(x)
        return x, y


def load_subsets_from_zip(
    zip_path: str
) -> Tuple[Subset, Subset, dict]:
    """
    Memuat train_subset & val_subset + class_to_idx dari file ZIP hasil preprocessing.
    """
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open('train_subset.pkl', 'r') as f:
            train_subset = pickle.load(f)
        with zf.open('val_subset.pkl', 'r') as f:
            val_subset = pickle.load(f)
        with zf.open('class_to_idx.pkl', 'r') as f:
            class_to_idx = pickle.load(f)
    return train_subset, val_subset, class_to_idx


def get_default_transforms(image_size: int = 64):
    """
    Menghasilkan transform train & val yang konsisten dengan pipeline kamu.
    - Output tensor dinormalisasi ke [-1,1].
    """
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # -> [-1,1]
    ])
    return train_tf, val_tf


def make_loaders_from_zip(
    zip_path: str,
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = 64,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Shortcut: load subsets + bungkus dengan TransformedSubset + buat DataLoader.
    """
    train_subset, val_subset, class_to_idx = load_subsets_from_zip(zip_path)
    train_tf, val_tf = get_default_transforms(image_size=image_size)

    ds_train = TransformedSubset(train_subset, transform=train_tf)
    ds_val = TransformedSubset(val_subset, transform=val_tf)

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader, class_to_idx
