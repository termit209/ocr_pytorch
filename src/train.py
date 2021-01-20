import torch
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import metrics

from config import *

from model import OcrModel
from constants import IMAGE_HEIGHT, IMAGE_WIDTH
from train_utils import train, evaluate
from dataset import OcrDataset


def fit(model, train_loader, val_loader, config: Config):
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=2, verbose=True
    )

    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, optimizer)
        valid_loss = evaluate(model, val_loader)


if __name__ == '__main__':
    full_df = pd.DataFrame()
    for dataset_path in config.dataset_paths:
        cur_df = pd.read_csv(dataset_path, sep='\t')
        full_df = pd.concat([full_df, cur_df], axis=0, ignore_index=False)
    train_df, val_df = model_selection.train_test_split(full_df, test_size=0.1, random_state=2020)

    train_dataset = OcrDataset(
        df=train_df,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.n_workers,
        shuffle=True
    )

    val_dataset = OcrDataset(
        df=val_df,
        resize=(IMAGE_HEIGHT, IMAGE_WIDTH),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=config.n_workers,
        shuffle=False,
        collate_fn=SynthCollator()
    )

    fit()