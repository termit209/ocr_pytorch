import torch
import numpy as np

from sklearn import model_selection
from sklearn import metrics

from config import *
from dataset import OcrDataset
from dataset import SynthCollator

from model import OcrModel_v0, OcrModel_mobilenet, OcrModel_vgg16
from train_utils import train, evaluate
from decode_predictions import decode_preds
from pprint import pprint
import wandb

wandb.login()

def fit_with_logging():
    (
        train_img,
        test_img, 
        train_labels, 
        test_labels, 
        train_orig_labels, 
        test_orig_targets,
    )   =model_selection.train_test_split(
            IMAGES, LABELS_ENCODED, LABELS_NAMES, test_size=0.1, random_state=2020)
    
    train_dataset = OcrDataset(image_path=train_img, 
                               labels=train_labels, 
                               resize=(IMAGE_HEIGHT,IMAGE_WIDTH)
                              )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=SynthCollator()
    )
    
    
    test_dataset = OcrDataset(image_path=test_img, 
                               labels=test_labels, 
                               resize=(IMAGE_HEIGHT,IMAGE_WIDTH)
                              )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=SynthCollator()
    )
        
wandb.init(project="ocr_russian", config={
    "bb": '',
    "BATCH_SIZE": 32,
    "EPOCHS": 250,
    "DEVICE": "cuda",
    "LR": 3e-4
})


model = OcrModel_v0(num_characters=len(labels_encoded.classes_))
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
optimizer, factor=0.8, patience=2, verbose=True
    )

wandb.watch(model)  
for epoch in range(EPOCHS):
  train_loss = train(model, train_loader, optimizer)
  valid_preds, valid_loss = evaluate(model, test_loader)
  valid_final_preds = []
  
  for pred in valid_preds:
      #print(pred)
      cur_preds = decode_preds(pred, labels_encoded)
      valid_final_preds.extend(cur_preds)
  valid_final_preds = transform_timeseries_string(valid_final_preds)
  show_preds_list = list(zip(test_orig_targets, valid_final_preds))[1:4]
  symbol_acc = sybol_wise_accuracy_batch(test_orig_targets, valid_final_preds)
  string_acc = string_wise_accuracy_batch(test_orig_targets, valid_final_preds)
  print(*show_preds_list, sep='\n')
  pprint("-"*90)
  wandb.log({"epoch": epoch, "train loss": train_loss, "valid loss": valid_loss, "symbol acc": symbol_acc, "string acc":string_acc})
  pprint(f"Epoch: {epoch} | Train loss = {train_loss} | Valid loss = {valid_loss} | symbol acc = {symbol_acc} | string acc = {string_acc} |")
  pprint("-"*90)