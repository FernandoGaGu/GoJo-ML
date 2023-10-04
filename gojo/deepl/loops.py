# Module containing the code typically used to train and evaluate Deep Learning models
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: still under development
#
import torch
import numpy as np
from typing import List, Iterable

from .callback import (
    Callback,
    EarlyStopping
)
from ..core.evaluation import Metric
from ..exception import DataLoaderError
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable,
    checkIterable
)


def _processInputParams(model: torch.nn.Module, device: str, metrics: list = None) -> tuple:
    """ Description """
    # TODO. Add model checking
    checkMultiInputTypes(
        ('model', model, [torch.nn.Module]),
        ('device', device, [str]),
        ('metrics', metrics, [list, type(None)]))

    # convert model to the input device
    model = model.to(device=device)

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


# TODO. Add dataloader checking
def iterSupervisedEpoch(
        model: torch.nn.Module,
        dataloader: Iterable,
        optimizer,
        loss_fn: callable,
        device: str,
        training: bool,
        metrics: list,
        **kwargs
    ) -> tuple:
    """ Description """

    # check input dataloader
    checkIterable('dataloader', dataloader)

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

        # TODO. Loss function calculation can be generalized through a Loss interface.
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
    metric_stats = {}
    for metric in metrics:
        metric_stats[metric.name] = metric(
            np.concatenate(y_trues), np.concatenate(y_preds))

    # calculate loss values
    loss_stats = {
        'loss (mean)': np.mean(loss_values),
        'loss (std)': np.std(loss_values)}

    return loss_stats, metric_stats


def fitNeuralNetwork(
        iter_fn,
        model: torch.nn.Module,
        train_dl: Iterable,
        valid_dl: Iterable,
        n_epochs: int,
        loss_fn: callable,
        optimizer_class,
        optimizer_params: dict = None,
        device: str = None,
        verbose: int = 1,
        metrics: list = None,
        callbacks: List[Callback] = None,
        **kwargs):
    """ Description """
    def _checkValidReturnedIteration(output, func: callable, step: str):
        # check that the returned objects correspond to a two-element tuple
        checkInputType('Output from function "%s" (step "%s")' % (func, step), output, [tuple])

        if len(output) != 2:
            raise IndexError(
                'Returned tuple from "%s" (step "%s") must be a two-element tuple. Number of elements: %d' % (
                    func, step, len(output)))

        for i, e in enumerate(output):
            checkInputType('output[%d]' % i, e, [dict])

    _AVAILABLE_DEVICES = ['cuda', 'mps', 'cpu']
    checkCallable('gojo.deepl.loops.fitNeuralNetwork(loss_fn)', loss_fn)
    checkIterable('gojo.deepl.loops.fitNeuralNetwork(train_dl)', train_dl)
    checkIterable('gojo.deepl.loops.fitNeuralNetwork(valid_dl)', valid_dl)
    checkMultiInputTypes(
        ('n_epochs', n_epochs, [int]),
        ('optimizer_params', optimizer_params, [dict, type(None)]),
        ('device', device, [str, type(None)]),
        ('verbose', verbose, [int]),
        ('metrics', metrics, [list, type(None)]),
        ('callbacks', callbacks, [list, type(None)]))

    # check input iteration function
    if iter_fn not in list(_AVAILABLE_ITERATION_FUNCTIONS.values()):
        raise TypeError(
            'Unrecognized "iter_fn" argument. Available functions are: %r' % getAvailableIterationFunctions())

    # select default device (order: cuda, mps, cpu)
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_built():
            device = 'mps'
        else:
            device = 'cpu'

    # check the selected device
    if device not in _AVAILABLE_DEVICES:
        raise TypeError('Unrecognized device "%s". Available devices are: %r' % (device, _AVAILABLE_DEVICES))

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all

    # process input parameters
    model, metrics = _processInputParams(
        model=model, device=device, metrics=metrics)

    # initialize the optimizer
    optimizer_obj = optimizer_class(model.parameters(), **optimizer_params)

    # perform the training loop
    train_metrics = []
    valid_metrics = []
    train_loss = []
    valid_loss = []
    for epoch in range(n_epochs):
        if verbose >= 1:
            print('\nEpoch (%d) ============================================ ' % (epoch+1))

        # -- training step -> (loss_stats: dict, metric_stats: dict)
        model = model.train()
        train_out = iter_fn(
            model=model,
            dataloader=train_dl,
            optimizer=optimizer_obj,
            loss_fn=loss_fn,
            device=device,
            training=True,
            metrics=metrics,
            **kwargs)

        # check returned function values
        _checkValidReturnedIteration(train_out, iter_fn, 'training')

        # separate loss/metric information
        epoch_train_loss, epoch_train_metrics = train_out

        # save epoch stats
        train_loss.append(epoch_train_loss)
        train_metrics.append(epoch_train_metrics)

        # display training statistics
        if verbose >= 1:
            for info_dict in train_out:
                for name, val in info_dict.items():
                    print('\t (train) %s: %.5f' % (name, val))
            print()

        # -- validation step -> (loss_stats: dict, metric_stats: dict)
        model = model.eval()
        valid_out = iter_fn(
            model=model,
            dataloader=valid_dl,
            optimizer=optimizer_obj,
            loss_fn=loss_fn,
            device=device,
            training=False,
            metrics=metrics,
            **kwargs)

        # check returned function values
        _checkValidReturnedIteration(valid_out, iter_fn, 'validation')

        # separate loss/metric information
        epoch_valid_loss, epoch_valid_metrics = valid_out

        # save epoch stats
        valid_loss.append(epoch_valid_loss)
        valid_metrics.append(epoch_valid_metrics)

        # display validation statistics
        if verbose >= 1:
            for info_dict in valid_out:
                for name, val in info_dict.items():
                    print('\t (valid) %s: %.5f' % (name, val))
            print()

        if callbacks is not None:
            commands_to_exec = [
                callback(
                    model=model,
                    train_metrics=train_metrics,
                    valid_metrics=valid_metrics,
                    train_loss=train_loss,
                    valid_loss=valid_loss)
                for callback in callbacks]

            # Early stopping directive
            if EarlyStopping.DIRECTIVE in commands_to_exec:
                if verbose >= 1:
                    print('!=!=!=!=!=!=!= Executing early stopping')
                break

    # TODO. Organize returned function output
    return dict(
        train_metrics=train_metrics,
        valid_metrics=valid_metrics,
        train_loss=train_loss,
        valid_loss=valid_loss)


def getAvailableIterationFunctions() -> list:
    """ Function that returns a list with all the available iteration functions used as 'iter_fn' argument in
    gojo.deepl.loops.fitNeuralNetwork() callings. """
    return list(_AVAILABLE_ITERATION_FUNCTIONS.keys())


_AVAILABLE_ITERATION_FUNCTIONS = {
    'iterSupervisedEpoch': iterSupervisedEpoch}
