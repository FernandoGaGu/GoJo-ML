# Module with functions and utilities for generating basic feed forward neural
# networks (FFN)
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: still under development, not functional
#
import torch
import numpy as np
from collections import OrderedDict


from ..util.validation import (
    checkMultiInputTypes

)

# TODO. Document function
def generateParametrizedLayers(
        n_layers: int,
        init_layer_dim: int,
        scaffold: str,
        min_width: int,
        max_width: int,
        beta: float or int,
        alpha: float or int
    ) -> list:
    """ Function that allows to generate FFN-layer layout based on a set of hyperparameters.


    Parameters
    ----------
    n_layers : int
        Number of layers.

    init_layer_dim : int
        TODO. DESCRIBE

    scaffold : str
        Model scaffold to arrange the layers. Valid scaffolds are:
            - 'exponential': TODO. DESCRIBE

                    n^(l) = (1/beta)^(l) * init

            - 'linear': TODO. DESCRIBE
                    n^(l) = init + alpha * (l)

    min_width : int
        Minimum layer width.

    max_width : int
        Maximum layer width.

    beta : float or int
        Applied for exponential scaffolds.

    alpha : float or int
        Applied for lineal scaffolds.
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
            np.array([init_layer_dim] * n_layers) + ( alpha * np.arange(n_layers) ))
    else:
        assert False, 'Unhandled case in gojo.deepl.ffn.generateParametrizedLayers (scaffold)'

    # adjust min_width and max_width
    layers[layers <= min_width] = min_width
    layers[layers >= max_width] = max_width

    return list(layers)










