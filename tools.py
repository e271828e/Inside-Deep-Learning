import os
import time
import functools

import torch
import torch.utils.data

from tqdm import tqdm

import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



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


def train_network( model, optimizer, loss_function, train_loader, eval_loaders=None, eval_metrics=None,
                  epochs=4, save_path=None, save_id=None, save_interval=2, load_file=None, show_progress=True):

    model_training = model.training

    if load_file is not None:
        checkpoint = torch.load(load_file, map_location=next(model.parameters()).device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        last_epoch = checkpoint["epoch"]
        last_time = checkpoint["time"]
    else:
        last_epoch = 0
        last_time = 0

    if save_id is None:
        save_id = "checkpoint"

    if eval_loaders is None:
        eval_loaders = {}

    if eval_metrics is None:
        eval_metrics = {"Loss": loss_function}

    step_results = {"Last Epoch": [last_epoch], "Training Time": [last_time]}

    #get current model evaluation before starting training
    scores = eval_network(model, eval_loaders, eval_metrics)
    for name, score in scores.items():
        step_results[name] = [score]

    for epoch in tqdm(range(last_epoch + 1, last_epoch + epochs + 1), desc="Epoch", disable=(not show_progress)):

        model.train()
        train_results = run_epoch( model, loss_function, optimizer, train_loader, desc=None, show_progress=show_progress)
        step_results["Last Epoch"].append(epoch)
        step_results["Training Time"].append(train_results["time"])

        scores = eval_network(model, eval_loaders, eval_metrics)
        for name, score in scores.items():
            step_results[name].append(score)


        if save_path is not None and epoch % save_interval == 0:
            file_name = save_id + "_epoch" + str(epoch) + ".pt"
            torch.save( {"epoch": epoch, "time": train_results["time"],
                         "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                         os.path.join(save_path, file_name))

    #restore model to its original mode
    model.train(model_training)

    return pd.DataFrame.from_dict(step_results)


def visualize_2D_classifier(X, y, model, resolution = 100, title=None):
    x_min = np.min(X[:, 0]) - 0.5
    x_max = np.max(X[:, 0]) + 0.5
    y_min = np.min(X[:, 1]) - 0.5
    y_max = np.max(X[:, 1]) + 0.5
    # create a grid of (x, y) points on which to evaluate the model
    xv, yv = np.meshgrid(np.linspace(x_min, x_max, num=resolution),
                         np.linspace(y_min, y_max, num=resolution), indexing="ij")
    # our model expects a tensor of shape [n_batch, 2] (two input features) and
    # outputs a tensor of logits of shape [n_batch, n_classes].we reshape the
    # grid's x and y coordinates into row matrices and stack these horizontally
    # into an array of shape [n_batch, n_features]
    xy_v = np.hstack((xv.reshape(-1, 1), yv.reshape(-1, 1)))
    with torch.no_grad():
        model_device = list(model.parameters())[0].device
        logits = model(torch.tensor(xy_v, dtype=torch.float32).to(model_device))
        # we apply softmax along the second dimension to convert the two values
        # of each of the n_batch outputs into a probability distribution
        y_hat = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    # plot a contour of the first of the two probability values
    cs = plt.contourf(xv, yv, y_hat[:, 0].reshape(resolution, resolution),
                      levels=np.linspace(0, 1, num=resolution), cmap=plt.cm.RdYlBu,)
    ax = plt.gca()
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, style=y, ax=ax)
    if title is not None:
        ax.set_title(title)
