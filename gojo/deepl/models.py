# Module with common neural network architectures.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#
import torch

from .ffn import createSimpleFFNModel
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)


class MultiTaskFFN(torch.nn.Module):
    """ Model adapted to perform multi-task classification with a layer specialized in extracting features (accessible
    through :meth:`feature_extractor`) and, as is typical in multi-task model training, layers specialized in
    performing the different tasks. The model will return a tensor with the outputs of each of the layers
    concatenated, where the first `n_clf_task` classification tasks will go first, followed by the `n_reg_task`
    regression tasks.


    Parameters
    -----------

    in_feats : int
        (feature extractor) Number of input features.

    emb_feats : int
        (feature extractor) Number of output features for the feature extractor feed forward network (FFN).

    layer_dims : list
        (feature extractor) Layer dims for the feature extractor feed forward network (FFN).

    n_clf_task : int
        (feature extractor) Number of classification task. If `n_clf_task` = 0, then `n_reg_task` must be greater
        than 1.

    n_reg_task : int
        (feature extractor) Number of regression task. If `n_reg_task` = 0, then `n_clf_task` must be greater than 1.

    multt_layer_dims : list
        (multi-task layers) Architecture used for the task's specialized layers.

    multt_dropout : list or float, default=None
        (multi-task layers) Dropout for the task's specialized layers.

    multt_layer_activation : torch.nn.Module or str or None, default='ELU'
        (multi-task layers) Activation function for the task's specialized layers.

    multt_batchnorm : bool, default=False
        (multi-task layers) Indicates whether to used batch normalization in the task's specialized layers.

    multt_clf_activation : str or torch.nn.Module or list or None, default='Sigmoid'
        (multi-task layers, classification) Output activation function for the classification task. If a list is
        provided this must match the length of the parameter `n_clf_task`.

    multt_reg_activation : str or torch.nn.Module or list or None, default=None
        (multi-task layers, regression) Output activation function for the regression task. If a list is provided this
        must match the length of the parameter `n_reg_task`.

    layer_dropout : list or float or None, default=None
        (feature extractor) Dropout rate for the feature extractor feed forward network (FFN).

    layer_activation : torch.nn.Module or str or None, default='ELU'
        (feature extractor) Activation function for the feature extractor feed forward network (FFN).

    batchnorm : bool, default=False
        (feature extractor param)Indicates whether to used batch normalization in the feature extractor feed forward
        network (FFN).

    Examples
    --------
    >>> from gojo import deepl
    >>>
    >>>
    >>> model = deepl.models.MultiTaskFFN(
    >>>     in_feats=100,
    >>>     emb_feats=20,
    >>>     layer_dims=[250, 50],
    >>>     n_clf_task=2,
    >>>     n_reg_task=3,
    >>>     multt_layer_dims=[20, 10],
    >>>     multt_dropout=0.2,
    >>>     multt_layer_activation='ELU',
    >>>     multt_batchnorm=False,
    >>>     multt_clf_activation='Sigmoid',
    >>>     multt_reg_activation=['TanU', None, None],
    >>>     layer_dropout=0.4,
    >>>     layer_activation='ELU',
    >>>     batchnorm=True
    >>> )
    """
    def __init__(
            self,
            # feature extractor parameters
            in_feats: int,
            emb_feats: int,
            layer_dims: list,
            n_clf_task: int,
            n_reg_task: int,

            # multi-task layers parameters
            multt_layer_dims: list,
            multt_dropout: list or float = None,
            multt_layer_activation: torch.nn.Module or str or None = 'ELU',
            multt_batchnorm: bool = False,
            multt_clf_activation: str or torch.nn.Module or list or None = 'Sigmoid',
            multt_reg_activation: str or torch.nn.Module or list or None = None,

            # other feature extractor parameters
            layer_dropout: list or float or None = None,
            layer_activation: torch.nn.Module or str or None = 'ELU',
            batchnorm: bool = False
    ):
        super(MultiTaskFFN, self).__init__()

        # save input parameters
        self.in_feats = in_feats
        self.emb_feats = emb_feats
        self.n_clf_task = n_clf_task
        self.n_reg_task = n_reg_task
        self.layer_dims = layer_dims
        self.multt_layer_dims = multt_layer_dims
        self.multt_dropout = multt_dropout
        self.multt_layer_activation = multt_layer_activation
        self.multt_batchnorm = multt_batchnorm
        self.multt_clf_activation = multt_clf_activation
        self.multt_reg_activation = multt_reg_activation
        self.layer_dropout = layer_dropout
        self.layer_activation = layer_activation
        self.batchnorm = batchnorm

        # check (and rearrange) input parameters
        self._checkModelParams()

        # create common layers (aka feature extractor)
        self.feature_extractor = createSimpleFFNModel(
            in_feats=self.in_feats,
            out_feats=self.emb_feats,
            layer_dims=self.layer_dims,
            layer_dropout=self.layer_dropout,
            batchnorm=self.batchnorm,
            layer_activation=self.layer_activation,
            output_activation=None)

        # create multitask layers
        # -- classification layers
        clf_layers = []
        for i in range(self.n_clf_task):
            clf_layers.append(
                createSimpleFFNModel(
                    in_feats=self.emb_feats,
                    out_feats=1,
                    layer_dims=self.multt_layer_dims,
                    layer_dropout=self.multt_dropout,
                    batchnorm=self.multt_batchnorm,
                    layer_activation=self.multt_layer_activation,
                    output_activation=self.multt_clf_activation[i]
                ))

        self.clf_layers = torch.nn.ModuleList(clf_layers)

        # -- regression layers
        reg_layers = []
        for i in range(self.n_reg_task):
            reg_layers.append(
                createSimpleFFNModel(
                    in_feats=self.emb_feats,
                    out_feats=1,
                    layer_dims=self.multt_layer_dims,
                    layer_dropout=self.multt_dropout,
                    batchnorm=self.multt_batchnorm,
                    layer_activation=self.multt_layer_activation,
                    output_activation=self.multt_reg_activation[i]
                ))

        self.reg_layers = torch.nn.ModuleList(reg_layers)

    def _checkModelParams(self):
        """ Function used to check the input parameters. """
        # check number of task parameters
        checkMultiInputTypes(
            ('n_clf_task', self.n_clf_task, [int]),
            ('n_reg_task', self.n_reg_task, [int]))

        if self.n_clf_task < 0:
            raise TypeError(
                'The number of classification tasks ("n_clf_task") cannot '
                'be less than 0 (provided %d).' % self.n_clf_task)
        if self.n_reg_task < 0:
            raise TypeError(
                'The number of regression tasks ("n_reg_task") cannot '
                'be less than 0 (provided %d).' % self.n_reg_task)
        if (self.n_clf_task + self.n_reg_task) < 2:
            raise TypeError(
                'n_clf_task + n_reg_task cannot be less than 2 (provided %d).' % (self.n_clf_task + self.n_reg_task))

        # put activation functions into a list
        if not isinstance(self.multt_clf_activation, list):
            self.multt_clf_activation = [self.multt_clf_activation] * self.n_clf_task
        if not isinstance(self.multt_reg_activation, list):
            self.multt_reg_activation = [self.multt_reg_activation] * self.n_reg_task

        # check activation functions shape
        if len(self.multt_clf_activation) != self.n_clf_task:
            raise TypeError('Missmatch between activation functions (%d) and classification tasks (%d)' % (
                len(self.multt_clf_activation), self.n_clf_task))

        if len(self.multt_reg_activation) != self.n_reg_task:
            raise TypeError('Missmatch between activation functions (%d) and regression tasks (%d)' % (
                len(self.multt_reg_activation), self.n_reg_task))

        checkMultiInputTypes(
            ('in_feats', self.in_feats, [int]),
            ('emb_feats', self.emb_feats, [int]),
            ('layer_dims', self.layer_dims, [list]),
            ('multt_layer_dims', self.multt_layer_dims, [list]),
            ('multt_dropout', self.multt_dropout, [list, float, type(None)]),
            ('multt_layer_activation', self.multt_layer_activation, [torch.nn.Module, str, type(None)]),
            ('multt_batchnorm', self.multt_batchnorm, [bool]),
            ('multt_clf_activation', self.multt_clf_activation, [list]),
            ('multt_reg_activation', self.multt_reg_activation, [list]),
            ('layer_dropout', self.layer_dropout, [list, float, type(None)]),
            ('layer_activation', self.layer_activation, [torch.nn.Module, str, type(None)]),
            ('batchnorm', self.batchnorm, [bool])
        )

        # check number of feature parameters
        if self.in_feats <= 0:
            raise TypeError(
                'The number of input features ("in_feats") cannot be less than 1 (provided %d).' % self.in_feats)
        if self.emb_feats <= 0:
            raise TypeError(
                'The number of embedding features ("emb_feats") cannot be less than 1 (provided %d).' % self.emb_feats)

        # check activation functions for classification tasks
        if len(self.multt_clf_activation) > 0:
            for i, e in enumerate(self.multt_clf_activation):
                checkInputType('multt_clf_activation[%d]' % i, e, [type(None), str, torch.nn.Module])

        # check activation functions for regression tasks
        if len(self.multt_reg_activation) > 0:
            for i, e in enumerate(self.multt_reg_activation):
                checkInputType('multt_reg_activation[%d]' % i, e, [type(None), str, torch.nn.Module])

    def forward(self, X: torch.Tensor, **_) -> torch.Tensor:

        # forward pass for the feature extractor
        emb = self.feature_extractor(X)

        # forward pass for the classification tasks
        clf_out = [self.clf_layers[i](emb) for i in range(len(self.clf_layers))]
        # forward pass for the regression tasks
        reg_out = [self.reg_layers[i](emb) for i in range(len(self.reg_layers))]

        # concat classification and regression predictions
        comb_preds = torch.cat(clf_out + reg_out, dim=1)

        return comb_preds

