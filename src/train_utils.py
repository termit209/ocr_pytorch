from tqdm import tqdm
import torch
from config import *



def train(model, dataloader, optimizer):
    model.train()
    final_loss = 0
    tracker = tqdm(dataloader, total=len(dataloader))
    for data in tracker:
        for key, value in data.items():
            data[key] = value.to(DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        final_loss += loss.item()
    return final_loss / len(dataloader)
    

def evaluate(model, dataloader):
    model.eval()
    final_loss = 0
    final_predictions = []
    tracker = tqdm(dataloader, total=len(dataloader))
    # if you have out of memory error put torch.no_grad()
    with torch.no_grad():
        for data in tracker:
            for key, value in data.items():
                data[key] = value.to(DEVICE)
            batch_predictions, loss = model(**data)
            final_predictions.append(batch_predictions)
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
   