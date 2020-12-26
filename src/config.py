from dataclasses import asdict, dataclass
from typing import Callable
import torch
import torch.nn as nn


@dataclass
class Config:
    epochs: int
    device: str
    n_workers: int
    random_seed: int
    batch_size: int
    dataset_path: str


config = Config(
    epochs=30,
    device='cuda',
    n_workers=4,
    random_seed=42,
    batch_size=32,
    dataset_path="/content/captcha_images_v2",
)