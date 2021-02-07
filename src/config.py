from dataclasses import dataclass
from torch import nn
from functools import partial
from model import *


@dataclass
class Config:
    epochs: int
    learning_rate: float
    device: str
    n_workers: int
    random_seed: int
    batch_size: int
    shape: tuple
    log_wnb: bool
    run_name: str
    model: nn.Module
    use_aug: bool
    use_padding: bool


config = Config(
    epochs = 200,
    learning_rate = 3e-4,
    device='cuda',
    n_workers = 4,
    random_seed = 42,
    batch_size = 32,
    shape = (128, 512),
    log_wnb = True,
    run_name = 'test_2g_aug',
    model = partial(OcrModel_v0),
    use_aug = True,
    use_padding = True
)