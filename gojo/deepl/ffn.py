# Module with functions and utilities for generating basic feed forward neural
# networks (FFN)
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: still under development functional
#
import torch
import numpy as np
from collections import OrderedDict


from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable
)


_LINEAR_LAYERS_IDENTIFIER = 'linear'
_ACTIVATION_LAYERS_IDENTIFIER = 'activation'
_DROPOUT_IDENTIFIER = 'dropout'
_BATCHNORM_IDENTIFIER = 'batchnorm'


def generateParametrizedLayers(
        n_layers: int,
        init_layer_dim: int,
        scaffold: str,
        min_width: int,
        max_width: int,
        beta: float or int,
        alpha: float or int) -> list:
    """ Function that allows to generate FFN-layer layout based on a set of hyperparameters.

    Parameters
    ----------
    n_layers : int
        Number of layers.

    init_layer_dim : int
        Dimensions of the first layer.

    scaffold : str
        Model scaffold to arrange the layers. Valid scaffolds are:
            - 'exponential': exponential decay in the number of layers. Controlled by the 'beta' parameter.

                    n^(l) = (1/beta)^(l) * init

            - 'linear': linear decay in the number of layers. Controlled by the 'alpha' parameter.
                    n^(l) = init - alpha * (l)

    min_width : int
        Minimum layer width.

    max_width : int
        Maximum layer width.

    beta : float or int
        Applied for exponential scaffolds.

    alpha : float or int
        Applied for lineal scaffolds.

    Returns
    -------
    dim_per_layer : list
        Dimensions of each layer (total of layers defined by 'n_layers').
    """
    _VALID_SCAFFOLDS = ['linear', 'exponential']

    # check input parameter types
    checkMultiInputTypes(
        ('n_layers', n_layers, [int]),
        ('init_layer_dim', init_layer_dim, [int]),
        ('scaffold', scaffold, [str]),
        ('min_width', min_width, [int]),
        ('max_width', max_width, [int]),
        ('beta', beta, [float, int]),
        ('alpha', alpha, [float, int]))

    # check input parameters that must be greater than 0
    for name, val in [
            ('init_layer_dim', init_layer_dim),
            ('n_layers', n_layers),
            ('min_width', min_width),
            ('max_width', max_width)]:
        if val <= 0:
            raise TypeError('Parameter "%s" cannot be <= 0' % name)

    # check width-related parameters
    if min_width > max_width:
        raise TypeError('Parameters "min_width" cannot be > than "max_width"')

    # check scaffolds
    if scaffold not in _VALID_SCAFFOLDS:
        raise TypeError('Unrecognized scaffold "%s". Valid scaffolds are: %r' % (scaffold, _VALID_SCAFFOLDS))

    # create layer dimensions
    layers = np.array([])
    if scaffold == 'exponential':
        layers = np.ceil(
            np.array([init_layer_dim] * n_layers) * ( (1/beta) ** np.arange(n_layers) ))
    elif scaffold == 'linear':
        layers = np.ceil(
            np.array([init_layer_dim] * n_layers) - ( alpha * np.arange(n_layers) ))
    else:
        assert False, 'Unhandled case in gojo.deepl.ffn.generateParametrizedLayers (scaffold)'

    # adjust min_width and max_width
    layers[layers <= min_width] = min_width
    layers[layers >= max_width] = max_width

    return list(layers.astype(int))


def _createFFN(
        in_feats: int,
        config: list,
        weights_init: list) -> torch.nn.Module:
    """ Function used to generate a torch.nn.Sequential model by using the configuration provided by the
    function 'gojo.deepl.ffn.createSimpleFFNModel', """
    def _createLinearLayer(_in_feats: int, _level_params: tuple, _weights_init):
        # TODO. Add the possibility of optional arguments

        if len(_level_params) < 3:
            raise TypeError(
                'Linear layer creation requires at lest a three element input (input: %r).' % list(_level_params))

        # extract arguments from the input data
        num_out_feats = _level_params[2]

        checkMultiInputTypes(
            ('_level_params[2]', _level_params[2], [int]))

        layer = torch.nn.Linear(in_features=_in_feats, out_features=num_out_feats)

        # change weights initialization (if specified)
        if _weights_init is not None:
            checkCallable('_weights_init', _weights_init)
            _weights_init(layer.weight)

        return layer

    # check parameter types
    checkMultiInputTypes(
        ('in_feats', in_feats, [int]),
        ('config', config, [list]),
        ('weights_init', weights_init, [list]))

    # check parameter values
    if in_feats < 1:
        raise TypeError('Parameter "in_feats" cannot be less than 1.')

    model_dict = OrderedDict()
    n_layer = 0
    for i, level_params in enumerate(config):

        # check minimum required length
        if len(level_params) < 2:
            raise TypeError(
                'Configuration entries must contain at least two elements. Error in index %d (%r)' % (
                    i, list(level_params)))

        # check input types
        checkInputType('config[%d]' % i, level_params, [tuple])
        checkInputType('config[%d][0]' % i, level_params[0], [str])

        # get the level name and level identifier
        level_name = level_params[0]
        level_identifier = level_params[1]

        # check for duplicated names
        if level_name in model_dict.keys():
            raise TypeError('Duplicated entry for level "%s" (index %d)' % (level_name, i))

        # create a linear layer
        if level_identifier == _LINEAR_LAYERS_IDENTIFIER:
            model_dict[level_name] = _createLinearLayer(
                _in_feats=in_feats, _level_params=level_params, _weights_init=weights_init[n_layer])

            # update in_feats and n_layer
            in_feats = level_params[2]
            n_layer += 1

        # add batch normalization
        elif level_identifier == _BATCHNORM_IDENTIFIER:
            model_dict[level_name] = torch.nn.BatchNorm1d(num_features=in_feats)

        # add activation layer
        elif level_identifier == _ACTIVATION_LAYERS_IDENTIFIER:
            if len(level_params) != 3:
                raise TypeError(
                    'Activation function adding requires at lest a three element input '
                    '(input: %r).' % list(level_params))

            # check that the provided type is a callable
            checkCallable('config[%d] - activation function' % i, level_params[2])

            model_dict[level_name] = level_params[2]

        # add dropout layer
        elif level_identifier == _DROPOUT_IDENTIFIER:
            if len(level_params) != 3:
                raise TypeError(
                    'Dropout creation requires at lest a three element input (input: %r).' % list(level_params))

            model_dict[level_name] = torch.nn.Dropout(p=level_params[2])

    return torch.nn.Sequential(model_dict)


def createSimpleFFNModel(
        in_feats: int,
        out_feats: int,
        layer_dims: list,
        layer_activation: list or torch.nn.Module or None or str,
        layer_dropout: list or float = None,
        batchnorm: bool = False,
        weights_init: callable or list = None,
        output_activation: str or torch.nn.Module or None or str = None) -> torch.nn.Module:
    """ Auxiliary function that allows to easily create a simple FFN architecture from the provided input parameters.

    See examples for a quick overview of the posibilities of this function.


    Parameters
    ----------
    in_feats : int
        Number of the features in the input data.

    out_feats : int
        Number of features in the output data.

    layer_dims : list
        Layer widths.

    layer_activation : list or torch.nn.Module or None or str
        Activation funtions. If None is provided a simple affine transformation will take place. If a string
        is provided, the name should match to the name of the torch.nn class (i.e., 'ReLU' for torch.nn.ReLU).

    layer_dropout : list or float, default=None
        Layer dropouts. If an scalar is provided the same dropout rate will be applied for all the layers.

    batchnorm : bool, default=False
        Parameter indicating whether to add batch-normalization layers.

    weights_init : callable or list, default=None
        Function (os list of functions) applied to the generated lienar layers for initializing their weights.

    output_activation : str or torch.nn.Module or None, default=None
        Output activation function (similar to 'layer_activation').

    Returns
    -------
    model : torch.nn.Module
        Generated model.


    Example
    -------
    >>> import torch
    >>> from gojo import deepl
    >>>
    >>>
    >>> model = deepl.ffn.createSimpleFFNModel(
    >>>     in_feats=100,
    >>>     out_feats=1,
    >>>     layer_dims=[100, 60, 20],
    >>>     layer_activation=torch.nn.ReLU(),
    >>>     layer_dropout=0.3,
    >>>     batchnorm=True,
    >>>     output_activation=torch.nn.Sigmoid()
    >>> )
    >>> model
    Out[0]
        Sequential(
              (LinearLayer 0): Linear(in_features=100, out_features=100, bias=True)
              (BatchNormalization 0): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (Activation 0): ReLU()
              (Dropout 0): Dropout(p=0.3, inplace=False)
              (LinearLayer 1): Linear(in_features=100, out_features=60, bias=True)
              (BatchNormalization 1): BatchNorm1d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (Activation 1): ReLU()
              (Dropout 1): Dropout(p=0.3, inplace=False)
              (LinearLayer 2): Linear(in_features=60, out_features=20, bias=True)
              (BatchNormalization 2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (Activation 2): ReLU()
              (Dropout 2): Dropout(p=0.3, inplace=False)
              (LinearLayer 3): Linear(in_features=20, out_features=1, bias=True)
              (Activation 3): Sigmoid()
            )
    >>>
    >>> model = deepl.ffn.createSimpleFFNModel(
    >>>     in_feats=10,
    >>>     out_feats=77,
    >>>     layer_dims=[100, 60, 20],
    >>>     layer_activation=[torch.nn.Tanh(), None, torch.nn.ReLU()],
    >>>     layer_dropout=[0.3, None, 0.1],
    >>>     batchnorm=True,
    >>>     weights_init=[torch.nn.init.kaiming_uniform_] * 3 + [None],
    >>>     output_activation=torch.nn.Sigmoid()
    >>> )
    >>> model

    Out[1]
        Sequential(
              (LinearLayer 0): Linear(in_features=10, out_features=100, bias=True)
              (BatchNormalization 0): BatchNorm1d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (Activation 0): Tanh()
              (Dropout 0): Dropout(p=0.3, inplace=False)
              (LinearLayer 1): Linear(in_features=100, out_features=60, bias=True)
              (BatchNormalization 1): BatchNorm1d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (LinearLayer 2): Linear(in_features=60, out_features=20, bias=True)
              (BatchNormalization 2): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (Activation 2): ReLU()
              (Dropout 2): Dropout(p=0.1, inplace=False)
              (LinearLayer 3): Linear(in_features=20, out_features=99, bias=True)
              (Activation 3): Sigmoid()
            )

    """
    # check input parameter types
    checkMultiInputTypes(
        ('in_feats', in_feats, [int]),
        ('out_feats', out_feats, [int]),
        ('layer_dims', layer_dims, [list]),
        ('layer_activation', layer_activation, [list, torch.nn.Module, type(None), str]),
        ('layer_dropout', layer_dropout, [list, float, type(None)]),
        ('batchnorm', batchnorm, [bool]),
        ('output_activation', output_activation, [str, torch.nn.Module, type(None), str]))

    # check parameter values
    if len(layer_dims) < 1:
        raise TypeError('Parameters "layer_dims" cannot be less than 1.')

    def _param2list(param_name: str, param_val, _layer_dims):
        # process param-level values (convert to a list of the same size as 'layer_dims')
        if not isinstance(param_val, list):
            param_val = [param_val] * len(_layer_dims)

        if len(_layer_dims) != len(param_val):
            raise TypeError(
                'Parameter "%s" (length %d) must be of the same size than "layer_dims" (length %d)"' % (
                    param_name, len(param_val), len(_layer_dims)))

        return param_val

    def _getActivation(_activation):
        # return the activation function to be used
        if isinstance(_activation, str):
            return getattr(torch.nn, _activation)()
        return _activation

    layer_activation = _param2list('layer_activation', layer_activation, layer_dims)
    layer_dropout = _param2list('layer_dropout', layer_dropout, layer_dims)
    # HACK. weights init can be applied to the last layer
    weights_init = _param2list('weights_init', weights_init, layer_dims + [None])

    # create configuration
    config = []
    for i, (dim, activation, do) in enumerate(zip(layer_dims, layer_activation, layer_dropout)):

        # add linear layer
        config.append(('LinearLayer %d' % i, _LINEAR_LAYERS_IDENTIFIER, dim))

        # add batch-normalization
        if batchnorm:
            config.append(('BatchNormalization %d' % i, _BATCHNORM_IDENTIFIER))

        # add the activation function
        if activation is not None:
            config.append(('Activation %d' % i, _ACTIVATION_LAYERS_IDENTIFIER, _getActivation(activation)))

        # add dropout (after activation)
        if do is not None:
            config.append(('Dropout %d' % i, _DROPOUT_IDENTIFIER, do))

    # add output layer
    config.append(('LinearLayer %d' % len(layer_dims), _LINEAR_LAYERS_IDENTIFIER, out_feats))

    if output_activation is not None:
        config.append(
            ('Activation %d' % len(layer_dims), _ACTIVATION_LAYERS_IDENTIFIER, _getActivation(output_activation)))

    return _createFFN(
        in_feats=in_feats,
        config=config,
        weights_init=weights_init)

