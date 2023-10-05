# Description

import torch
import numpy as np
import pandas as pd
from typing import Dict
from torch.utils.data import Dataset


from ..core import base as core_base


class TorchDataset(Dataset):
    """ Basic Dataset class for typical tabular-like supervised problems.

    Parameters
    -----------
    """
    def __init__(
            self,
            X: np.ndarray or pd.DataFrame,
            y: np.ndarray or pd.DataFrame or pd.Series):
        super().__init__()

        X_dt = core_base.Dataset(X)
        y_dt = core_base.Dataset(y)

        np_X = X_dt.array_data
        np_y = y_dt.array_data

        # add extra dimension to y
        if len(np_y.shape) == 1:
            np_y = np_y[:, np.newaxis]

        if np_X.shape[0] != np_y.shape[0]:
            raise TypeError(
                'Input "X" (shape[0] = %d) and "y" (shape[0] = %d) must contain the same number of entries in the '
                'first dimension.' % (np_X.shape[0], np_y.shape[0]))

        self.X = torch.from_numpy(np_X.astype(np.float32))
        self.y = torch.from_numpy(np_y.astype(np.float32))
        self.X_dataset = X_dt
        self.y_dataset = y_dt

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]

