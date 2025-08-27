import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from typing import Dict, Any


class TextDataset(Dataset):
    def __init__(self, text_ids, block_size=128):
        super(TextDataset, self).__init__()
        self.text_ids = text_ids
        self.block_size = block_size

    def __len__(self):
        return (len(self.text_ids) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        x = torch.tensor(self.text_ids[start:end], dtype=torch.long)
        y = torch.tensor(self.text_ids[start + 1 : end + 1], dtype=torch.long)
        return x, y


def init_dataloader(data_ids, block_size=128, batch_size=64, train=True, shuffle=True, **kwargs):
    train_dataset = TextDataset(data_ids, block_size)
    if torch.cuda.is_available():
        num_workers = kwargs.get("num_workers", 2)
        persistent_workers = kwargs.get("persistent_workers", True)
        pin_memory = kwargs.get("pin_memory", True)
        prefetch_factor = kwargs.get("prefetch_factor", 2)
    else:
        num_workers = 0
        persistent_workers = False
        pin_memory = False
        prefetch_factor = None

    if train:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return train_loader


class WarmupThenCosine(LRScheduler):
    def __init__(self, optimizer, warmup_steps, T_max, last_epoch=-1, eta_min=0.0):
        self.warmup_steps = max(0, warmup_steps)
        self.T_max = T_max
        self.cosine = None
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            scale = step / self.warmup_steps
            return [base_lr * scale for base_lr in self.base_lrs]
        # initialize cosine on first post-warmup call
        if self.cosine is None:
            # remaining steps after warmup
            remain = max(1, self.T_max - self.warmup_steps)
            self.cosine = CosineAnnealingLR(self.optimizer, T_max=remain, eta_min=self.eta_min)
            self.cosine.last_epoch = -1  # reset internal counter so first get_lr() is step 0
        return self.cosine.get_last_lr()

    def step(self, epoch=None):
        # Advance this scheduler
        super().step(epoch)
        # Also advance cosine if active and after warmup
        if self.cosine is not None:
            self.cosine.step(epoch if epoch is not None else None)


def set_seed(seed: int = 1337, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(state: Dict[str, Any], path: str, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path, **kwargs)


def load_checkpoint(path: str, map_location="cpu", **kwargs) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, **kwargs)
