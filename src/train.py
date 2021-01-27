import torch
import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn import metrics

from config import *

from model import OcrModel
from constants import IMAGE_FOLDERS, LABELS_LIST
from train_utils import train, evaluate
from dataset import OcrDataset
from utils import create_label_dict


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
    label_dict = create_label_dict(IMAGE_FOLDERS, LABELS_LIST)
    img_paths, labels_names = labels_dict.keys(), labels_dict.values()

    train_dataset = OcrDataset(
        df=train_df,
        resize=config.shape
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
        shuffle=False
    )

    fit()