# Module with experimental code to be incorporated into gojo.deepl.loading.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import torch
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from copy import deepcopy

from ..deepl.loops import fitNeuralNetwork
from ..core.base import TorchSKInterface
from ..util.validation import (
    checkInputType,
    checkMultiInputTypes)
from ..util.io import (
    _createObjectRepresentation
)


# TODO. Update documentation
class GNNTorchSKInterfaz(TorchSKInterface):
    """  Wrapper class designed to integrate pytorch models ('torch.nn.Module' instances) based on Graph Neural Networks
    (GNNs) in the :py:mod:`gojo` library functionalities.

    Parameters
    ----------
    model : torch.nn.Module
        Subclass of 'torch.nn.Module'.

    iter_fn : callable
        Function that executes an epoch of the torch.nn.Module typical training pipeline. For more information
        consult :py:mod:`gojo.deepl.loops`.

    loss_function : callable
        Loss function used to train the model.

    n_epochs : int
        Number of epochs used to train the model.

    train_split : float
        Percentage of the training data received in :meth:`train` that will be used to train the model. The rest of
        the data will be used as validation set.

    optimizer_class : type
        Pytorch optimizer used to train the model (see `torch.optim` module.)

    dataset_class : type
        Pytorch class dataset used to train the model (see `torch.utils.data` module or the `gojo` submodule
        :py:mod:`gojo.deepl.loading`).

    dataloader_class : type
        Pytorch dataloader class (`torch.utils.data.DataLoader`).

    optimizer_kw : dict, default=None
        Parameters used to initialize the provided optimizer class.

    train_dataset_kw : dict, default=None
        Parameters used to initialize the provided dataset class for the data used for training.

    valid_dataset_kw : dict, default= None
        Parameters used to initialize the provided dataset class for the data used for validation.

    train_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for training.

    valid_dataloader_kw : dict, default=None
        Parameters used to initialize the provided dataloader class for the data used for validation.

    iter_fn_kw : dict, default=None
        Optional arguments of the parameter `iter_fn`.

    train_split_stratify : bool, default=False
        Parameter indicating whether to perform the train/validation split with class stratification.

    callbacks : List[:class:`gojo.deepl.callback.Callback`], default=None
        Callbacks during model training. For more information see :py:mod:`gojo.deepl.callback`.

    metrics : List[:class:`gojo.core.evaluation.Metric`], default=None
        Metrics used to evaluate the model performance during training. Fore more information see
        :py:mod:`gojo.core.evaluation.Metric`.

    seed : int, default=None
        Random seed used for controlling the randomness.

    device : str, default='cpu'
        Device used for training the model.

    verbose : int, default=1
        Verbosity level. Use -1 to indicate maximum verbosity.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            iter_fn: callable,

            # training parameters
            loss_function,
            n_epochs: int,
            train_split: float,

            # classes
            optimizer_class,
            dataset_class,
            dataloader_class,

            # optional arguments for the input classes
            optimizer_kw: dict = None,
            train_dataset_kw: dict = None,
            valid_dataset_kw: dict = None,
            train_dataloader_kw: dict = None,
            valid_dataloader_kw: dict = None,
            iter_fn_kw: dict = None,

            # other parameters
            train_split_stratify: bool = False,
            callbacks: list = None,
            metrics: list = None,
            seed: int = None,
            device: str = 'cpu',
            verbose: int = 1

    ):
        super().__init__(
            model=model, iter_fn=iter_fn,  loss_function=loss_function, n_epochs=n_epochs, train_split=train_split,
            optimizer_class=optimizer_class, dataset_class=dataset_class, dataloader_class=dataloader_class,
            optimizer_kw=optimizer_kw, train_dataset_kw=train_dataset_kw, valid_dataset_kw=valid_dataset_kw,
            train_dataloader_kw=train_dataloader_kw, valid_dataloader_kw=valid_dataloader_kw, iter_fn_kw=iter_fn_kw,
            train_split_stratify=train_split_stratify, callbacks=callbacks, metrics=metrics, seed=seed, device=device,
            verbose=verbose)

    def __repr__(self):
        return _createObjectRepresentation(
            'GNNTorchSKInterfaz',
            **self.getParameters())

    def train(self, X: np.ndarray, y: np.ndarray or None = None, **kwargs):
        """ Train the model using the input data.

        Parameters
        ----------
        X : np.ndarray
            Predictor variables.

        y : np.ndarray or None, default=None
            Target variable.

        **kwargs
            Optional arguments. Importantly, for models based on GNNs, an adjacency matrix or edge list associated
            with each subject must be provided.
        """

        # reset callbacks inner states
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback.resetState()

        # separate train/validation data
        stratify = None
        if self.train_split_stratify:
            stratify = y
            if y is None:
                warnings.warn(
                    'target indices have been specified to be stratified by class but a value of "y" has not been '
                    'provided as input. Ignoring "train_split_stratify"')
                stratify = None

        # train/validation split based on indices
        indices = np.arange(X.shape[0])
        train_idx, valid_idx = train_test_split(
            indices, train_size=self.train_split, random_state=self.seed, shuffle=True, stratify=stratify)

        # train/validation split of the optional arguments
        train_kwargs = {}
        valid_kwargs = {}
        if len(kwargs) > 0:
            for var_name, var_values in kwargs.items():
                # check optional arguments
                checkInputType('kwargs["%s"]' % var_name, var_values, [list])
                if len(var_values) != len(indices):
                    raise TypeError(
                        'Missmatch in X shape (%d) and **kwargs["%s"] shape (%d).' % (
                            len(indices), var_name, len(var_values)))

                train_kwargs[var_name] = [var_values[idx] for idx in train_idx]
                valid_kwargs[var_name] = [var_values[idx] for idx in valid_idx]

        # create dataloaders
        train_dl = self.dataloader_class(
            self.dataset_class(
                X=X[train_idx],
                y=y[train_idx] if y is not None else y,
                **train_kwargs,
                **self.train_dataset_kw),
            **self.train_dataloader_kw)

        valid_dl = self.dataloader_class(
            self.dataset_class(
                X=X[valid_idx],
                y=y[valid_idx] if y is not None else y,
                **valid_kwargs,
                **self.valid_dataset_kw),
            **self.valid_dataloader_kw)

        # train the model
        history = fitNeuralNetwork(
            iter_fn=self.iter_fn,
            model=self.model,
            train_dl=train_dl,
            valid_dl=valid_dl,
            n_epochs=self.n_epochs,
            loss_fn=self.loss_function,
            optimizer_class=self.optimizer_class,
            optimizer_params=self.optimizer_kw,
            device=self.device,
            verbose=self.verbose,
            metrics=self.metrics,
            callbacks=self.callbacks,
            **self.iter_fn_kw)

        # save model fitting history
        self._fitting_history = history

        self.fitted()

    def performInference(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.

        **kwargs
            Optional arguments. Importantly, for models based on GNNs, an adjacency matrix or edge list associated
            with each subject must be provided. For more information about the optional arguments consult the
            GNNs-associated datasets.

        Returns
        -------
        model_predictions : np.ndarray
            Model predictions associated with the input data.


        Notes
        -----
        This function will internally create an internal dataloader using the template specified for the validation
        data and setting the `shuffle` parameter to False.
        """

        # select the model in inference mode
        self.model = self.model.eval()
        self.model = self.model.to(device=self.device)

        # HACK. Avoid sorting the predictions
        dataloader_op_args = deepcopy(self.valid_dataloader_kw)
        dataloader_op_args['shuffle'] = False

        # create the dataloader
        test_dl = self.dataloader_class(
            self.dataset_class(X=X, y=None, **kwargs, **self.valid_dataset_kw), **dataloader_op_args)

        with torch.no_grad():
            y_pred = []

            # iterate over the input data in batches
            for X_batch, _ in test_dl:
                X_batch = X_batch.to(device=self.device)
                y_hat = self.model(X_batch).detach().cpu().numpy()
                y_pred.append(y_hat)

        y_pred = np.concatenate(y_pred)

        # flatten y_pred when dimensions are (n_samples, 1)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)

        return y_pred
