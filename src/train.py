import torch
import numpy as np

from sklearn import model_selection
from sklearn import metrics

from config import *
from dataset import OcrDataset
from dataset import SynthCollator

from model import OcrModel_v0, OcrModel_mobilenet, OcrModel_vgg16
from train_utils import *
from decode_predictions import decode_preds
from pprint import pprint



def fit():
    (
        train_img,
        test_img,
        train_labels,
        test_labels,
        train_orig_labels,
        test_orig_targets,
    ) = model_selection.train_test_split(
        IMAGES, LABELS_ENCODED, LABELS_NAMES, test_size=0.1, random_state=2020)

    train_dataset = OcrDataset(image_path=train_img,
                               labels=train_labels,
                               resize=(IMAGE_HEIGHT, IMAGE_WIDTH)
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
                              resize=(IMAGE_HEIGHT, IMAGE_WIDTH)
                              )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=SynthCollator()
    )

    model = OcrModel_v0(num_characters=len(labels_encoded.classes_))
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=2, verbose=True
    )

    loss_data, valid_data, symb_acc_data, str_acc_data, str_1, str_2 = [], [], [], [], [], []
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer)
        valid_preds, valid_loss = evaluate(model, test_loader)
        valid_final_preds = []

        for pred in valid_preds:
            # print(pred)
            cur_preds = decode_preds(pred, labels_encoded)
            valid_final_preds.extend(cur_preds)
        valid_final_preds = transform_timeseries_string(valid_final_preds)
        show_preds_list = list(zip(test_orig_targets, valid_final_preds))[1:4]
        symbol_acc = symbol_wise_accuracy_batch(test_orig_targets, valid_final_preds)
        string_acc = string_wise_accuracy_batch(test_orig_targets, valid_final_preds)
        string_acc_1 = string_wise_accuracy_batch(test_orig_targets,
                                                  valid_final_preds,
                                                  errors_to_omit=1)
        string_acc_2 = string_wise_accuracy_batch(test_orig_targets,
                                                  valid_final_preds,
                                                  errors_to_omit=2)
        print(*show_preds_list, sep='\n')
        pprint("-" * 90)
        pprint(
            "Epoch: {} | Train loss = {:.5f} | Valid loss = {:.5f} | symbol acc = {:.2f}% | string acc = {:.2f}% |".format(
                epoch, train_loss, valid_loss, symbol_acc * 100, string_acc * 100
            ))
        pprint('String accuracy (1 symbol omitted) = {:.2f}% | String accuracy (2 symbols omitted) = {:.2f}%'.format(
            string_acc_1 * 100, string_acc_2 * 100))
        pprint("-" * 90)
        loss_data.append(train_loss)
        valid_data.append(valid_loss)
        symb_acc_data.append(symbol_acc)
        str_acc_data.append(string_acc)
        str_1.append(string_acc_1)
        str_2.append(string_acc_2)


if __name__ == '__main__':
    fit()
