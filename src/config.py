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
    dataset_paths: list
    model: nn.Module
    shape: tuple


config = Config(
    epochs=30,
    device='cuda',
    n_workers=4,
    random_seed=42,
    batch_size=32,
    shape=(512, 512),
    dataset_paths=[
        "/content/handwritten_rus/LABELED/assignments_from_pool_601263__05-12-2020.tsv",
        "/content/handwritten_rus/LABELED/assignments_from_pool_615470__05-12-2020.tsv",
    ],
    model=OcrModel,
)

