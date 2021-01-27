from dataclasses import dataclass
from torch import nn
from model import OcrModel


@dataclass
class Config:
    epochs: int
    device: str
    n_workers: int
    random_seed: int
    batch_size: int
    model: nn.Module
    shape: tuple


config = Config(
    epochs=30,
    device='cuda',
    n_workers=4,
    random_seed=42,
    batch_size=32,
    shape=(64, 512),
    model=OcrModel,
)

