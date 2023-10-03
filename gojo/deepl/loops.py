# Module containing the code typically used to train and evaluate Deep Learning models
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: still under development
#
import torch
import numpy as np
import warnings

from ..core.evaluation import Metric
from ...exception import DataLoaderError
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)


# TODO. Create TorchModelInterface
# TODO. Add model checking
def _processInputParams(model, device: str, training: bool, metrics: list = None) -> tuple:
    """ Description """
    # TODO. Add model checking
    checkMultiInputTypes(
        ('device', device, [str]),
        ('metrics', metrics, [list, type(None)]),
        ('training', training, [bool]),
    )

    # convert model to the input device
    model = model.to(device=device)

    # select training / inference model
    model = model.train() if training else model.eval()

    # check provided metrics for 'loss' or 'loss_std' functions
    if metrics is None:  # if not metrics were provided create the default metric (avg/std loss function)
        metrics = []

    # check input metrics
    unique_metric_names = []
    for i, metric in enumerate(metrics):
        checkInputType('metrics[%d]' % i, metric, [Metric])   # check for gojo.core.evaluation.Metric instances
        unique_metric_names.append(metric.name)

    # check duplicated metric names
    if len(metrics) > 0 and (len(set(unique_metric_names)) != len(unique_metric_names)):
        raise TypeError(
            'Duplicated metric names detected. Input metric names: %r' % unique_metric_names)

    return model, metrics


# TODO. Create TorchModelInterface
# TODO. Add model checking
# TODO. Add dataloader checking
def iterSupervisedEpoch(
        model,
        dataloader,
        optimizer,
        loss_fn,
        device: str,
        training: bool,
        verbose: bool = False,
        metrics: list = None
    ) -> tuple:
    """ Description """

    # process input parameters
    model, metrics = _processInputParams(
        model=model, device=device, training=training, metrics=metrics)

    # iterate over batches
    loss_values = []
    y_preds = []
    y_trues = []
    for batch, dlargs in enumerate(dataloader):
        if len(dlargs) < 2:
            raise DataLoaderError(
                'The minimum number of arguments returned by a dataloader must be 2 where the first element will '
                'correspond to the input data (the Xs) and the second to the target to be approximated (the Ys). '
                'The rest of the returned arguments will be passed in the order returned to the model.')
        X = dlargs[0].to(device=device)
        y = dlargs[1].to(device=device)
        var_args = dlargs[2:]

        # perform model inference (training/testing)
        if training:
            # training loop (calculate gradients and apply backpropagation)
            y_hat = model(X, *var_args)
            # evaluate loss function
            loss = loss_fn(y_hat, y)
            # apply backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            # inference model (no gradients will be computed)
            with torch.no_grad():
                y_hat = model(X, *var_args)
                # evaluate loss function
                loss = loss_fn(y_hat, y)

        # gather model predictions and true labels
        y_pred_np = y_hat.detach().cpu().numpy().astype(float)
        y_true_np = y.detach().cpu().numpy().astype(float)

        # save model predictions and true labels
        y_preds.append(y_pred_np)
        y_trues.append(y_true_np)

        # save loss value
        loss_values.append(loss.detach().cpu().item())

    # calculate metrics (if provided)
    metric_values = {}
    for metric in metrics:
        metric_values[metric.name] = metric(
            np.concatenate(y_trues), np.concatenate(y_preds))

    # calculate loss values
    loss_stats = {
        'loss (mean)': np.mean(loss_values),
        'loss (std)': np.std(loss_values)}

    return loss_stats, metric_values


