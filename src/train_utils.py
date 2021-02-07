import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from config import Config


def train(model, dataloader, optimizer, device):
    model.train()
    final_loss = 0
    for data in dataloader:
        for key, value in data.items():
            data[key] = value.to(device)
        optimizer.zero_grad()
        x = model(**data)

        data['label_size']
        log_softmax_values = F.log_softmax(x, 2)
        input_lenghts = torch.full(size=(len(data['images']),),
                                   fill_value=log_softmax_values.size(0),
                                   dtype=torch.int32
                                   )

        output_lenghts = data['label_size']
        labels = data['labels']

        loss = nn.CTCLoss(blank=0, zero_infinity=True)(
            log_softmax_values,
            labels,
            input_lenghts,
            output_lenghts)

        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(dataloader)
    

def evaluate(model, dataloader, device):
    model.eval()
    final_loss = 0
    final_predictions = []
    with torch.no_grad():
        for data in dataloader:
            for key, value in data.items():
                data[key] = value.to(device)
            x = model(**data)
            log_softmax_values = F.log_softmax(x, 2)
            input_lenghts = torch.full(size=(len(data['images']),),
                                   fill_value=log_softmax_values.size(0),
                                   dtype=torch.int32
                                   )

            output_lenghts = data['label_size']
            labels = data['labels']

            loss = nn.CTCLoss(blank=0, zero_infinity=True)(
            log_softmax_values,
            labels,
            input_lenghts,
            output_lenghts)

            final_predictions.append(x)
            final_loss += loss.item()
        return final_predictions, final_loss / len(dataloader)


def transform_timeseries_string(batch):
    new_batch = []
    for string in batch:
        new_string = []
        last_symbol = '*'
        for symbol in string:
            if symbol == '*':
                pred_symbol = ''
            elif symbol == last_symbol:
                pass
            else:
                new_string.append(symbol)
                last_symbol = symbol
        new_batch.append(new_string)
    return new_batch


def sybol_wise_accuracy(true, pred):
    num_true_symbol = 0
    for index in range(min(len(true), len(pred))):
        if pred[index] == true[index]:
            num_true_symbol += 1
    return num_true_symbol / len(true)


def sybol_wise_accuracy_batch(true, pred):
    acc = 0.0
    for batch_ind in range(len(true)):
        acc += sybol_wise_accuracy(true[batch_ind], pred[batch_ind])
    return acc / len(true)


def string_wise_accuracy_batch(pred, true):
    num_correct_strings = 0
    for batch_ind in range(len(true)):
        if ''.join(pred[batch_ind]) == ''.join(true[batch_ind]):
            num_correct_strings += 1
    return num_correct_strings / len(true)

def set_seed(seed):
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    else:
        pass

def decode_preds(preds, encoder):
    preds = preds.permute(1,0,2)
    preds = torch.softmax(preds,2)
    preds = torch.argmax(preds,2)
    preds = preds.detach().cpu().numpy()
    preds_list  = []
    for i in range(preds.shape[0]):
        tmp = []
        for j in preds[i,:]:
            j = j-1
            if j == -1:
                tmp.append("*")
            else:
                tmp.append(encoder.inverse_transform([j])[0])
        element = "".join(tmp)
        preds_list.append(element)
    return preds_list
