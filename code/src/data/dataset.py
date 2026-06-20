"""PyTorch dataset + dataloaders built from the manifest produced by build_dataset.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.config import load_config, resolve_path


class ChartDataset(Dataset):
    """Reads (image, label) pairs from the manifest for one split."""

    def __init__(self, manifest: pd.DataFrame, classes: list[str], transform=None) -> None:
        self.df = manifest.reset_index(drop=True)
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(resolve_path(row["path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[row["label"]]


def build_transforms(image_size: int, train: bool, augment: str = "light"):
    """Augmentations. Two presets:
      - 'light' : tiny horizontal jitter only (original; flips/colour would break semantics).
      - 'strong': small affine (translate+scale) and gentle uniform brightness jitter.
                  No flips, no hue/saturation/contrast: candle semantics (green=up, red=down)
                  are preserved.
    """
    base = [transforms.Resize((image_size, image_size))]
    if train:
        if augment == "strong":
            base.append(transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.02), scale=(0.95, 1.05), fill=0,
            ))
            base.append(transforms.ColorJitter(brightness=0.10, contrast=0.0,
                                                saturation=0.0, hue=0.0))
        else:
            base.append(transforms.RandomAffine(degrees=0, translate=(0.03, 0.0)))
    base += [transforms.ToTensor()]
    return transforms.Compose(base)


def make_loaders(cfg: dict, image_size: int, batch_size: int = 32, num_workers: int = 2,
                 augment: str = "light"):
    """Return (train_loader, val_loader, test_loader, classes) from the manifest."""
    proc = resolve_path(cfg["paths"]["processed_dir"])
    manifest = pd.read_csv(proc / "manifest.csv")
    classes = cfg["classes"]

    loaders = {}
    for split in ["train", "val", "test"]:
        sub = manifest[manifest["split"] == split]
        ds = ChartDataset(sub, classes,
                          build_transforms(image_size, train=(split == "train"), augment=augment))
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=False,
        )
    return loaders["train"], loaders["val"], loaders["test"], classes


def compute_class_weights(cfg: dict, classes: list[str]) -> "torch.Tensor":
    """Inverse-frequency class weights for cross-entropy, normalised to mean 1."""
    import torch
    proc = resolve_path(cfg["paths"]["processed_dir"])
    manifest = pd.read_csv(proc / "manifest.csv")
    train = manifest[manifest["split"] == "train"]
    counts = train["label"].value_counts().reindex(classes).fillna(0).to_numpy()
    inv = counts.sum() / (len(classes) * counts.clip(min=1))
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)
