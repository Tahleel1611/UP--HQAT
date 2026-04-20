from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    dataset: str
    batch_size: int
    num_workers: int
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    val_split: float = 0.1
    seed: int = 42
    data_root: str = "data"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_dataset(name: str, train: bool, root: Path):
    if name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        return datasets.MNIST(root=str(root), train=train, download=True, transform=transform)

    if name == "cifar10":
        if train:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                ]
            )
        return datasets.CIFAR10(root=str(root), train=train, download=True, transform=transform)

    raise ValueError(f"Unsupported dataset: {name}")


def _maybe_subset(dataset, max_samples: int | None):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def _split_train_validation(dataset, val_split: float, seed: int):
    if val_split <= 0.0:
        return dataset, None
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0,1), got {val_split}")
    val_len = int(len(dataset) * val_split)
    val_len = max(1, val_len)
    train_len = len(dataset) - val_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    return train_set, val_set


def make_dataloaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader | None, DataLoader, int, int]:
    root = Path(cfg.data_root)
    root.mkdir(parents=True, exist_ok=True)

    train_set = _build_dataset(cfg.dataset, train=True, root=root)
    test_set = _build_dataset(cfg.dataset, train=False, root=root)
    train_set = _maybe_subset(train_set, cfg.max_train_samples)
    test_set = _maybe_subset(test_set, cfg.max_test_samples)
    train_set, val_set = _split_train_validation(train_set, cfg.val_split, cfg.seed)

    in_channels = 1 if cfg.dataset == "mnist" else 3
    num_classes = 10

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader, in_channels, num_classes
