# Module with functions and utilities used to operate with Graph Neural Networks models through the `torch_geometric`
# framework.
#
# Author: Fernando García Gutiérrez
# Email: ga.gu.fernando.concat@gmail.com
#
import torch
import torch_geometric as geom
from abc import ABCMeta, abstractmethod


class BaseGNN(torch.nn.Module):
    """ Base class used to define the basic models based on Graph Neural Networks (GNN). To extend this class, the
    :meth:`initLayer` method used to initialize each block of the GNN model must be defined.

    Parameters
    ----------
    in_feats : int
        Number of input features (node features).

    hidden_channels : int or list
        Dimension of the hidden representations of the model. If given as an integer, the same dimension will be used
        throughout the layers of the model, and the number of layers must be defined by parameter `n_layers`. If
         provided as a `list`, each value in the list will be interpreted as a number of layers and the `n_layers`
         parameter will be ignored.

    n_layers : int, default=None
        Number of layers of the model. This parameter will only be taken into account if `hidden_channels` is specified
        as an integer.

    aggregation : object, default=None
        `torch_geometric.nn.Aggregation` instance used to add node embeddings.

    **kwargs
        Keyword arguments passed to the :meth:`initLayer` method to initialize the model layers.
    """

    __metaclass__ = ABCMeta

    def __init__(
            self,
            in_feats: int,
            hidden_channels: int or list,
            n_layers: int or None = None,
            aggregation: object or None = None,
            **kwargs
    ):
        super(BaseGNN, self).__init__()

        # process homogeneous hidden_channels across layers
        if isinstance(hidden_channels, int):
            if n_layers is None:
                raise ValueError('If "hidden_channels" is specified as int, "n_layers" must be provided')
            hidden_channels = [hidden_channels] * n_layers

        # add input dimensions
        hidden_channels = [in_feats] + hidden_channels

        # initialize model layers
        model_layers = []
        for i in range(len(hidden_channels) - 1):
            layers = self.initLayer(in_feats=hidden_channels[i], out_feats=hidden_channels[i + 1], **kwargs)
            for layer in layers:
                model_layers.append(layer)

        # create a basic torch_geometric model
        self.model = geom.nn.Sequential(
            'x, edge_index', model_layers
        )

        # save the aggregation schema
        self.aggregation = aggregation

    @abstractmethod
    def initLayer(self, in_feats: int, out_feats: int, **kwargs) -> tuple or list:
        """ Method to be defined to initialize the different blocks of the model layers. The layers must be returned
        in iterable form.

        Parameters
        ----------
        in_feats : int
            Layer input features.

        out_feats : int
            Layer output features.

        **kwargs
            Keyword arguments.
        """
        raise NotImplementedError

    def forward(self, batch, **_):
        out = self.model(x=batch.x, edge_index=batch.edge_index)

        if self.aggregation is not None:
            out = self.aggregation(out, batch.batch)

        return out


class GCN(BaseGNN):
    """ Graph Convolutional Network model.

    Parameters
    ----------
    in_feats : int
        Number of input features (node features).

    hidden_channels : int or list
        Dimension of the hidden representations of the model. If given as an integer, the same dimension will be used
        throughout the layers of the model, and the number of layers must be defined by parameter `n_layers`. If
         provided as a `list`, each value in the list will be interpreted as a number of layers and the `n_layers`
         parameter will be ignored.

    n_layers : int, default=None
        Number of layers of the model. This parameter will only be taken into account if `hidden_channels` is specified
        as an integer.

    aggregation : object, default=None
        `torch_geometric.nn.Aggregation` instance used to add node embeddings.

    act_function : object, default=`torch.nn.ReLU`
        Activation function. By default, `torch.nn.ReLU` will be used.

    add_self_loops : bool
        Indicates whether to add self-loops in `torch_geometric.nn.GCNConv`.

    normalize : bool
        Indicates whether to apply normalization in `torch_geometric.nn.GCNConv`.

    bias : bool
        Indicates whether to used bias in `torch_geometric.nn.GCNConv`.

    eps : float
        Epsilon provided to the `torch_geometric.nn.GraphNorm` constructor.

    p : float
        Dropout probability provided to the `torch.nn.Dropout` constructor.
    """

    # optional arguments allowed for `torch_geometric.nn.GCNConv` layers
    GCN_KWARGS = ['add_self_loops', 'normalize', 'bias']
    # optional arguments allowed for `torch_geometric.nn.GraphNorm` layers
    GRAPH_NORM_KWARGS = ['eps']
    # optional arguments allowed for `torch_geometric.nn.GraphNorm` layers
    DROPOUT_KWARGS = ['p']

    def initLayer(self, in_feats: int, out_feats: int, **kwargs) -> tuple:
        # extract optional arguments for each of the layers
        gcn_kwargs = {k: v for k, v in kwargs.items() if k in self.GCN_KWARGS}
        graph_norm_kwargs = {k: v for k, v in kwargs.items() if k in self.GRAPH_NORM_KWARGS}
        dropout_kwargs = {k: v for k, v in kwargs.items() if k in self.DROPOUT_KWARGS}

        # get the activation function
        act_function = kwargs.get('act_function', torch.nn.ReLU())

        return (
            (geom.nn.GCNConv(in_feats, out_feats, **gcn_kwargs), 'x, edge_index -> x'),
            geom.nn.GraphNorm(in_channels=out_feats, **graph_norm_kwargs),
            act_function,
            torch.nn.Dropout(**dropout_kwargs)
        )

