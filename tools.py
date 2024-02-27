import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

import time


def run_epoch( model, f_loss, optimizer, data_loader, score_funcs=None, computing_device=torch.device("cpu"), desc=None, show_progress=True):
    """
    model -- the PyTorch model / "Module" to run for one epoch
    optimizer -- the object that will update the weights of the network
    data_loader -- DataLoader object that returns tuples of (input, label) pairs.
    loss_func -- the loss function that takes in two arguments, the model outputs and the labels, and returns a score
    device -- the compute lodation to perform training
    score_funcs -- a dictionary of scoring functions to use to evalue the performance of the model
    prefix -- a string to pre-fix to any scores placed into the _results_ dictionary.
    desc -- a description to use for the progress bar.
    """
    batch_losses = []
    y_batches = []
    y_hat_batches = []
    start = time.time()
    for x_batch, y_batch in tqdm( data_loader, desc=desc, leave=False, disable=(not show_progress)):
        
        #here, inputs and labels will be in the storage device, which can be the
        #same or different from the compute device
        x_batch = x_batch.to(computing_device)
        y_batch = y_batch.to(computing_device)

        y_hat_batch = model(x_batch)

        batch_loss = f_loss(y_hat_batch, y_batch)

        if model.training:
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        batch_losses.append(batch_loss.item())

        #at this point, both y_batch and y_hat_batch are stored at
        #computing_device, because y_batch was explicitly moved to it, and y_hat
        #has been created at the model's device, which must also be
        #computing_device (otherwise we would have gotten an error). since we
        #are going to use scoring functions defined outside PyTorch, we need
        #these tensors converted back to regular NumPy arrays, and this in turn
        #requires detaching them and moving them to the CPU. 
        
        #these are both lists of batch_len x out_features numpy arrays, one per
        #batch processed so far
        y_batches.append(y_batch.detach().cpu().numpy())
        y_hat_batches.append(y_hat_batch.detach().cpu().numpy()) 

    # end training epoch
    end = time.time()

    y_epoch = np.concatenate(y_batches)
    y_hat_epoch = np.concatenate(y_hat_batches)

    return y_epoch, y_hat_epoch

    #score functions must be consistent with the type of problem. 

    y_pred = np.asarray(y_pred)
    if (
        len(y_pred.shape) == 2 and y_pred.shape[1] > 1
    ):  # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)
    # Else, we assume we are working on a regression problem

    results[prefix + " loss"].append(np.mean(epoch_losses))
    for name, score_func in score_funcs.items():
        try:
            results[prefix + " " + name].append(score_func(y_true, y_pred))
        except:
            results[prefix + " " + name].append(float("NaN"))
    return end - start  # time spent on epoch

# def train_network( model, loss_func, train_loader, val_loader=None, test_loader=None, score_funcs=None, epochs=50, device="cpu", checkpoint_file=None, lr_schedule=None, optimizer=None, disable_tqdm=False,):
# def train_network_new(model, f_loss, optimizer, training_loader, testing_loader=None, validation_loader=None, score_functions = None, epochs = 50, lr_schedule=None, checkpoint_load_path=None, checkpoint_save_path=None):




#determine device by looking at model parameters