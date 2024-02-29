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



def run_epoch(model, loss_function, optimizer, data_loader, desc=None, show_progress=True):

    #the computing device is found from the first model parameter. we assume all
    #parameters in the model are stored in the same device, otherwise we will
    #get an error later on
    computing_device = next(model.parameters()).device

    y_batches = []
    y_hat_batches = []
    torch.cuda.synchronize()
    t_start = time.time()
    for x_batch, y_batch in tqdm( data_loader, desc=desc, leave=False, disable=(not show_progress)):
        
        x_batch = x_batch.to(computing_device)
        y_batch = y_batch.to(computing_device)

        y_hat_batch = model(x_batch)

        if model.training:
            batch_loss = loss_function(y_hat_batch, y_batch)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        y_batches.append(y_batch.detach())
        y_hat_batches.append(y_hat_batch.detach()) 

    torch.cuda.synchronize()
    t_end = time.time()

    y_epoch = torch.cat(y_batches) #along first dimension
    y_hat_epoch = torch.cat(y_hat_batches) #idem

    results = {}
    results["time"] = t_end - t_start
    results["y_hat"] = y_hat_epoch
    results["y"] = y_epoch

    return results


def eval_network(model, loaders, metrics):

    model_training = model.training
    model.eval()

    scores = {}
    for loader_name, loader in loaders.items():
        results = run_epoch( model, None, None, loader, desc=None, show_progress=False)
        for metric_name, metric in metrics.items():
            key = loader_name + " " + metric_name
            try:
                scores[key] = metric(results["y_hat"], results["y"]).item()
            except:
                scores[key] = float("NaN")

    #restore model to its previous mode
    model.train(model_training)

    return scores


def train_network( model, loss_function, optimizer, train_loader, eval_loaders=None, eval_metrics=None,
                  final_epoch=1, save_checkpoint=None, lr_schedule=None, show_progress=True):

    model_training = model.training

    #GET THESE FROM CHECKPOINT
    initial_epoch = 0 #number of completed training epochs before we start
    initial_time = 0

    if eval_loaders is None:
        eval_loaders = {}

    if eval_metrics is None:
        eval_metrics = {"Loss": loss_function}

    step_results = {"Epochs Trained": [initial_epoch], "Accumulated Time": [initial_time]}

    #get current model evaluation before starting training
    scores = eval_network(model, eval_loaders, eval_metrics)
    for name, score in scores.items():
        step_results[name] = [score]

    for epoch in tqdm(range(initial_epoch + 1, final_epoch + 1), desc="Epoch", disable=(not show_progress)):

        model.train()
        train_results = run_epoch( model, loss_function, optimizer, train_loader, desc=None, show_progress=show_progress)
        step_results["Epochs Trained"].append(epoch)

        #add epoch training time to the last accumulated value
        step_results["Accumulated Time"].append(step_results["Accumulated Time"][-1] + train_results["time"])

        scores = eval_network(model, eval_loaders, eval_metrics)
        for name, score in scores.items():
            step_results[name].append(score)


    #restore model to its original mode
    model.train(model_training)

    #save checkpoints on every epoch that is a multiple of checkpoint_interval or the last one
    
    return pd.DataFrame.from_dict(step_results)