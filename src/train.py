import torch
import numpy as np
import pandas as pd

from sklearn import model_selection, metrics
from config import *

from constants import IMAGE_FOLDERS, LABELS_LIST
from train_utils import *
from dataset import OcrDataset
from utils import *


def fit(model, train_loader, val_loader, config: Config):
    model.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=8, 
                                                           verbose=True, threshold=0.1)
    if config.log_wnb:
        wandb.init(project="ocr_russian", config=config, name=config.run_name)
        wandb.watch(model) 

    for epoch in range(config.epochs):
        train_loss = train(model, train_loader, optimizer, config.device)
        valid_preds, valid_loss = evaluate(model, val_loader, config.device)
        valid_final_preds = []
        
        for pred in valid_preds:
            cur_preds = decode_preds(pred, label_encoder)
            valid_final_preds.extend(cur_preds)
        valid_final_preds = transform_timeseries_string(valid_final_preds)
        symbol_acc = sybol_wise_accuracy_batch(test_orig_targets, valid_final_preds)
        string_acc = string_wise_accuracy_batch(test_orig_targets, valid_final_preds)
        
        wandb.log({"train loss": train_loss, "valid loss": valid_loss, "symbol acc": symbol_acc, 
                   "string acc":string_acc, "loss":(train_loss, valid_loss)})
        print("-"*90)
        print(f"Epoch: {epoch} | Train loss = {train_loss} | Valid loss = {valid_loss} | symbol acc = {symbol_acc} | string acc = {string_acc} |")
        print("-"*90)
        scheduler.step(valid_loss)


if __name__ == '__main__':
    set_seed(config.random_seed)
    labels_dict = create_label_dict(IMAGE_FOLDERS, LABELS_LIST)
    labels_names = list(labels_dict.values())
    images_path = list(labels_dict.keys())
    label_encoded, label_encoder = get_labels_encode(labels_names)
    (train_img, test_img, train_labels, test_labels, 
     train_orig_labels, test_orig_targets) = model_selection.train_test_split(images_path, label_encoded, 
                                                                              labels_names, test_size=0.1, 
                                                                              random_state=config.random_seed)
    train_dataset = OcrDataset(image_path=train_img, labels=train_labels, resize=config.shape, 
                               use_augment=config.use_aug, use_padding=config.use_padding)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, 
                                               num_workers=config.n_workers, shuffle=True)
    test_dataset = OcrDataset(image_path=test_img, labels=test_labels, resize=config.shape)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, 
                                              num_workers=config.n_workers, shuffle=False)
    model = OcrModel_effnetb0(num_characters=len(label_encoder.classes_))
    fit(model, train_loader, test_loader, config)
