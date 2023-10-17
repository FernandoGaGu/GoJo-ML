# Module with data loading utilities
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#

import torch
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset

from ..core import base as core_base


class TorchDataset(Dataset):
    """ Basic Dataset class for typical tabular data. This class can be passed to `torch.DataLoaders`
    and subsequently used by the :func:`gojo.deepl.loops.fitNeuralNetwork` function.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Input predictor variables used to fit the models.

    y : np.ndarray or pd.DataFrame or pd.Series, default=None
        Target variables to fit the models (or None).

    Example
    -------
    >>> from gojo import deepl
    >>> from torch.utils.data import DataLoader
    >>>
    >>> # dataset loading ...
    >>> X = np.random.uniform(size=(30, 10))
    >>> y = np.random.uniform(size=30)
    >>>
    >>> # use TorchDataset for an easy use of pytorch DataLoaders
    >>> dataloader = DataLoader(
    >>>     deepl.loading.TorchDataset(X=X, y=y),
    >>>     batch_size=16, shuffle=True)
    >>>
    """
    def __init__(
            self,
            X: np.ndarray or pd.DataFrame,
            y: np.ndarray or pd.DataFrame or pd.Series = None):
        super().__init__()

        # process X-related parameters
        X_dt = core_base.Dataset(X)
        np_X = X_dt.array_data
        self.X = torch.from_numpy(np_X.astype(np.float32))
        self.X_dataset = X_dt

        # initialize y information (default None)
        self.y = None
        self.y_dataset = None

        # y parameter is optional
        if y is not None:
            y_dt = core_base.Dataset(y)
            np_y = y_dt.array_data

            # add extra dimension to y
            if len(np_y.shape) == 1:
                np_y = np_y[:, np.newaxis]

            if np_X.shape[0] != np_y.shape[0]:
                raise TypeError(
                    'Input "X" (shape[0] = %d) and "y" (shape[0] = %d) must contain the same number of entries in the '
                    'first dimension.' % (np_X.shape[0], np_y.shape[0]))

            self.y = torch.from_numpy(np_y.astype(np.float32))
            self.y_dataset = y_dt

    def __getitem__(self, idx: int):
        y = None if self.y is None else self.y[idx]

        return self.X[idx], y

    def __len__(self):
        return self.X.shape[0]

