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
import warnings
from typing import List
from torch.utils.data import Dataset
import torch_geometric as geom

from ..core import base as core_base
from ..util.validation import checkInputType


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
        if self.y is None:
            return self.X[idx]

        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


class GraphDataset(Dataset):
    """ Class used to generate a dataset adapted to operate with Graph Neural Networks. This class can be passed
    to `torch.utils.data.DataLoader` and subsequently used by the :func:`gojo.deepl.loops.fitNeuralNetwork` function.

    Parameters
    ----------

    X : np.ndarray or pd.DataFrame or List[np.ndarray]
        Input predictor variables used to adjust the models. If a numpy array or a pandas DataFrame is provided,
        entries along dimension 0 will be interpreted as instances, and the 1-axis will be interpreted as the
        number of nodes in the network. In the case where a list of numpy arrays is provided, each element of
        the list will be interpreted as an instance, the 0-axis as the number of nodes, and the remaining
        dimensions as node features.

    y : np.ndarray or pd.DataFrame or pd.Series, default=None
        Target variables to fit the models (or None).

    adj_matrix : np.ndarray or pd.DataFrame or List[np.ndarray, pd.DataFrame], default=None
        Adjacency matrix. If a numpy array or a pandas DataFrame is provided, it must have a shape of
        (`n_nodes`, `n_nodes`). In the case where a list of numpy arrays is provided, each element of the list will be
        interpreted as a graph, and it must have a shape of (`n_nodes`, `n_nodes`).

        One of `adj_matrix` or `edge_index` must be provided.

    edge_index : np.ndarray or pd.DataFrame or List[np.ndarray, pd.DataFrame], default=None
        Edge index. If a numpy array or a pandas DataFrame is provided, it must have a shape of (`2`, `n_nodes`). In
        the case where a list of numpy arrays is provided, each element of the list will be interpreted as a graph,
        and it must have a shape of (`2`, `n_nodes`).

        One of `adj_matrix` or `edge_index` must be provided.

    tabular_x: np.ndarray or pd.DataFrame or List[np.ndarray], default=None
        Tabular characteristics that will be stored in the `tabular_x` attribute of the instances (
        `torch_geometric.data.DataBatch`) returned by this dataset.

        .. important::
            Internally a dimension will be added along axis 1 to prevent `torch_geometric` dataloaders from flattening
            the data to a single dimension.


    Example
    -------
    >>> import numpy as np
    >>> import gojo
    >>>
    >>> n_samples = 10     # number of instances
    >>> n_node_feats = 3   # number of node features
    >>>
    >>> # generate random adjacency matrices, one for each sample
    >>> adj_matrices = []
    >>> for _ in range(n_samples):
    >>>     n_nodes = np.random.randint(5, 30)
    >>>     adj_matrices.append(np.random.randint(0, 2, size=(n_nodes, n_nodes)))
    >>>
    >>> # generate the node features
    >>> # each sample will be (n_nodes, n_node_features)
    >>> node_feats = [
    >>>     np.random.uniform(size=(adj_matrix.shape[0], n_node_feats))
    >>>     for adj_matrix in adj_matrices
    >>> ]
    >>>
    >>> # generate a target feature
    >>> target = np.random.randint(0, 2, size=n_samples)
    >>>
    >>> # create the dataset
    >>> graph_dt = gojo.experimental.deepl_loading.GraphDataset(
    >>>     X=node_feats,
    >>>     y=target,
    >>>     adj_matrix=adj_matrices
    >>> )
    >>>

    """
    def __init__(
        self,
        X: np.ndarray or pd.DataFrame or List[np.ndarray],
        y: np.ndarray or pd.DataFrame or pd.Series = None,
        adj_matrix: np.ndarray or pd.DataFrame or List[np.ndarray, pd.DataFrame] = None,
        edge_index: np.ndarray or pd.DataFrame or List[np.ndarray, pd.DataFrame] = None,
        tabular_x: np.ndarray or pd.DataFrame or List[np.ndarray] = None
    ):
        super(GraphDataset, self).__init__()

        # check input arguments
        if adj_matrix is not None and edge_index is not None:
            warnings.warn(
                'Both "adj_matrix" and "edge_index" have been provided, "edge_index" will be ignored.')

        if adj_matrix is None and edge_index is None:
            raise TypeError('At least one of "adj_matrix" or "edge_index" must be provided.')

        # process the y variable
        y_tensor = torch.from_numpy(np.array([np.nan] * len(X)).astype(np.float32))
        if y is not None:
            # get the y data as a numpy array
            np_y = core_base.Dataset(y).array_data

            # add extra dimension to y (n_samples, n_targets)
            if len(np_y.shape) == 1:
                np_y = np_y[:, np.newaxis]

            # convert the y variable to a torch.Tensor
            y_tensor = torch.from_numpy(np_y.astype(np.float32))

        # process X variable
        x_list_tensor = []
        if isinstance(X, list):

            for i, e in enumerate(X):
                checkInputType('X[i]', e, [np.ndarray])

            # create a list of tensors
            x_list_tensor = [
                torch.from_numpy(X[i].astype(np.float32))
                for i in range(len(X))]

            # if the elements of the tensor are of shape (n_nodes) add n_node_features dimension
            for i in range(len(x_list_tensor)):
                if len(x_list_tensor[i].shape) == 1:
                    x_list_tensor[i] = x_list_tensor[i].unsqueeze(-1)
        else:
            # get the X data as a numpy array
            np_X = core_base.Dataset(X).array_data.astype(np.float32)

            # add extra dimension to X (n_samples, n_nodes, n_node_features)
            if len(np_X.shape) == 2:
                np_X = np_X[:, :, np.newaxis]

            # create a list of tensors
            x_list_tensor = [
                torch.from_numpy(np_X[i, ...]) for i in range(np_X.shape[0])]

        # check y and X shape
        if len(y_tensor) != len(x_list_tensor):
            raise TypeError(
                'Number of samples in "X" (%d) does not match the number of samples in "y" (%d)' % (
                 len(x_list_tensor), len(y_tensor)))

        # process the adjacency matrix / edge index
        edge_index_ = None
        if adj_matrix is not None:
            if isinstance(adj_matrix, list):

                for i, e in enumerate(adj_matrix):
                    checkInputType('adj_matrix[i]', e, [np.ndarray])

                # convert adjacency matrix to edge index
                edge_index_ = [
                    torch.nonzero(torch.from_numpy(adj_matrix[i].astype(int))).t()
                    for i in range(len(adj_matrix))]

            else:
                # create copies of the adjacency matrix as edge index
                adj_matrix_np = core_base.Dataset(adj_matrix).array_data.astype(int)
                edge_index_ = [
                    torch.nonzero(torch.from_numpy(adj_matrix_np)).t()
                    for _ in range(len(x_list_tensor))]

        else:
            if isinstance(edge_index, list):
                # convert edge index to torch.Tensor
                for i, e in enumerate(edge_index):
                    checkInputType('edge_index[i]', e, [np.ndarray])
                    edge_index[i] = torch.from_numpy(edge_index[i].astype(int))

                edge_index_ = edge_index
            else:
                # create copies of the edge index
                np_edge_index = core_base.Dataset(edge_index).array_data.astype(int)
                edge_index_ = [
                    torch.from_numpy(np_edge_index) for _ in range(len(x_list_tensor))]

        # check the shape of the edge index
        assert len(edge_index_) == len(x_list_tensor),\
            'Missmatch in internal "edge_index_" (%d) shape and "x_list_tensor" (%d).' % (
            len(edge_index_), len(x_list_tensor))

        for i in range(len(edge_index_)):
            if len(edge_index_[i].shape) != 2:
                raise TypeError(
                    'edge_index[%d] shape different from 2 (%d)' % (i, len(edge_index_[i].shape)))

            if edge_index_[i].shape[0] != 2:
                raise TypeError(
                    'edge_index[%d].shape[0] different from 2 (%d)' % (i, edge_index_[i].shape[0]))

        # check the consistency in the number of nodes
        for i, (nodes_, sample_) in enumerate(zip(edge_index_, x_list_tensor)):
            if nodes_.max()+1 != sample_.shape[0]:
                raise TypeError(
                    'Different number of nodes in sample %d (edge_index=%d, feature_vector=%d)' % (
                    i, nodes_.max()+1, sample_.shape[0]))

        # check tabular information
        checkInputType('tabular_x', tabular_x, [pd.DataFrame, np.ndarray, List, type(None)])
        if not (tabular_x is None or len(tabular_x) == len(X)):
            raise TypeError(
                '"tabular_x" shape along 0 axis missmatch (expected %d, input %d)' % (len(x_list_tensor), len(X)))
        else:
            if isinstance(tabular_x, list):
                for i, e in enumerate(tabular_x):
                    checkInputType('tabular_x[%d]' % i, e, [np.ndarray])
                tabular_x = np.stack(tabular_x)

            if tabular_x is None:
                tabular_x = torch.from_numpy(np.array([np.nan] * len(X)).astype(np.float32))
            else:
                tabular_x = torch.from_numpy(core_base.Dataset(tabular_x).array_data.astype(np.float32))
                tabular_x = tabular_x.unsqueeze(1)

        # TODO. Implement edge_attr
        # ...

        self.y_tensor = y_tensor
        self.x_list_tensor = x_list_tensor
        self.edge_index = edge_index_
        self.tabular_x = tabular_x

        # create the torch_geometric.data.Data instances
        self.data_list = [
            geom.data.Data(
                x=x, edge_index=ei, y=y, tabular_x=tab_x
            ) for x, ei, y, tab_x in zip(x_list_tensor, edge_index_, y_tensor, tabular_x)
        ]

    def __getitem__(self, idx: int):
        return self.data_list[idx], self.y_tensor[idx]

    def __len__(self):
        return len(self.data_list)
