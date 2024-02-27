import time
import functools

import torch
import torch.utils.data

from tqdm import tqdm

import numpy as np

import pandas as pd



def from_logits(f_metric):
    @functools.wraps(f_metric)
    def wrapper(input_logits, target_class, *args, **kwargs):
        input_class = torch.argmax(input_logits, axis=1)
        score = f_metric(input_class, target_class, *args, **kwargs)
        return score
    return wrapper



def run_epoch(model, loss_function, optimizer, data_loader, metrics=None, desc=None, show_progress=True):

    #the computing device is found from the first model parameter. we assume all
    #parameters in the model are stored in the same device, otherwise we will
    #get an error later on
    computing_device = next(model.parameters()).device

    batch_losses = []
    y_batches = []
    y_hat_batches = []
    t_start = time.time()
    for x_batch, y_batch in tqdm( data_loader, desc=desc, leave=False, disable=(not show_progress)):
        
        x_batch = x_batch.to(computing_device)
        y_batch = y_batch.to(computing_device)

        y_hat_batch = model(x_batch)

        batch_loss = loss_function(y_hat_batch, y_batch)

        if model.training:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        batch_losses.append(batch_loss.item())

        y_batches.append(y_batch.detach())
        y_hat_batches.append(y_hat_batch.detach()) 

    t_end = time.time()

    y_epoch = torch.cat(y_batches) #along first dimension
    y_hat_epoch = torch.cat(y_hat_batches) #idem

    results = {}
    results["time"] = t_end - t_start
    results["loss"] = np.mean(batch_losses)

    if metrics is not None:
        for name, func in metrics.items():
            try:
                results[name] = func(y_hat_epoch, y_epoch)
            except:
                results[name] = float("NaN")

    return results