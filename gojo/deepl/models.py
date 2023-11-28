# Module with common neural network architectures.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#
import torch
import torch_geometric as geom

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


class MultiTaskFFNv2(torch.nn.Module):
    """ (Simplified version of :class:`gojo.deepl.models.MultiTaskFFN`) Model adapted to perform multi-task
    classification with a layer specialized in extracting features (accessible through :meth:`feature_extractor`) and,
    as is typical in multi-task model training, layers specialized in performing the different tasks (accessible
    through :meth:`multitask_projection`). The model will return a tensor with the outputs of each of the layers from
    the input parameter `multitask_projection` concatenated in the same order as declared in the input parameter.

    Parameters
    ----------
     feature_extractor : torch.nn.Module
        Layer that will take the input from the model and generate an embedded representation that will be subsequently
        used by the layers defined in `multitask_projection`.

     multitask_projection : torch.nn.ModuleList
        Layers specialized in different tasks. Their outputs will be concatenated along dimension 1.


    Examples
    --------
    >>> import torch
    >>> from gojo import deepl
    >>>
    >>>
    >>> X = torch.rand(10, 40)    # (batch_size, n_feats)
    >>>
    >>> multitask_model = deepl.models.MultiTaskFFNv2(
    >>>     feature_extractor=torch.nn.Sequential(
    >>>         torch.nn.Linear(40, 20),
    >>>         torch.nn.ReLU()
    >>>     ),
    >>>     multitask_projection=torch.nn.ModuleList([
    >>>         torch.nn.Sequential(
    >>>             torch.nn.Linear(20, 2),
    >>>             torch.nn.Tanh()
    >>>         ),
    >>>         torch.nn.Sequential(
    >>>             torch.nn.Linear(20, 1),
    >>>             torch.nn.Sigmoid()
    >>>         ),
    >>>     ])
    >>> )
    >>>
    >>> with torch.no_grad():
    >>>     mtt_out = multitask_model(X)
    >>>     emb = multitask_model.feature_extractor(X)
    >>>
    >>>
    >>> mtt_out[:, :2].min(), mtt_out[:, :2].max()
        Out[0]: (tensor(-0.2965), tensor(0.1321))
    >>>
    >>> mtt_out[:, 2].min(), mtt_out[:, 2].max()
        Out[1]: (tensor(0.3898), tensor(0.4343))
    >>>
    >>> emb.shape
        Out[2]: torch.Size([10, 20])
    """
    def __init__(
            self,
            feature_extractor: torch.nn.Module,
            multitask_projection: torch.nn.ModuleList
    ):
        super(MultiTaskFFNv2, self).__init__()

        self.feature_extractor = feature_extractor
        self.multitask_projection = multitask_projection

    def _checkModelParams(self):
        """ Function used to check the input parameters. """
        checkMultiInputTypes(
            ('feature_extractor', self.feature_extractor, [torch.nn.Module]),
            ('multitask_projection', self.multitask_projection, [torch.nn.ModuleList]))

    def forward(self, X, **_) -> torch.Tensor:

        # forward pass for the feature extractor
        emb = self.feature_extractor(X)

        # forward pass for the multitask FFNs
        mtt_out = [model(emb) for model in self.multitask_projection]

        # concatenate the output across the batch dimension
        out = torch.cat(mtt_out, dim=1)

        return out


class GNN(torch.nn.Module):
    """ Graph Neural Network wrapper for graph classification. This model allows integrating data in the form of a
    graph through a model defined `gnn_model` parameter, and  tabular data through a model defined in `ffn_model` and
    the resulting information will be fused using the defined `fusion_model`.

    If the parameter `ffn_model` is not provided only the embeddings generated by the model defined in `gnn_model`
    will be passed to the fusion layer (`fusion_model`). If the fusion_model parameter is not given, the embeddings
    resulting from the `gnn_model` model will be returned directly or, if the `ffn_model` parameter is given, the
    embeddings generated by both models concatenated.


    Parameters
    -----------
    gnn_model : torch.nn.Module
        Graph neural network model. See `torch_geometric` for model implementations.

    ffn_model : torch.nn.Module, default=None
        Feed forward network model for generate the output.

    fusion_model : torch.nn.Module, default=None
        Fusion layer for merge GNN and FFN derived information.

    gp_agg : str, default='sum'
        Graph-pooling aggregation.

    """
    def __init__(
            self,
            gnn_model: torch.nn.Module,
            ffn_model: torch.nn.Module = None,
            fusion_model: torch.nn.Module = None,
            use_tabular_x: bool = False,
            gp_agg: str = 'sum'
    ):
        super(GNN, self).__init__()

        self.gnn_model = gnn_model
        self.ffn_model = ffn_model
        self.fusion_model = fusion_model
        self.gp_agg = gp_agg
        self.use_tabular_x = use_tabular_x

    def gnnForward(self, x):
        return self.gnn_model(x=x.x, edge_index=x.edge_index)

    def graphPooling(self, x, batch):
        return geom.utils.scatter(x, batch, reduce=self.gp_agg)

    def ffnModel(self, x):
        return self.ffn_model(x)

    def fusionModel(self, x):
        return self.fusion_model(x)

    def forward(self, batch):
        """

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            `torch_geometric` batch dadta.
        """
        # GNN forward pass
        out = self.gnnForward(batch)

        # graph-level aggregation
        out = self.graphPooling(out, batch.batch)

        # FFN forward pass for the tabular information
        if self.ffn_model is not None:
            ffn_out = self.ffnModel(batch.tabular_x)
        elif getattr(batch, 'tabular_x', None) is not None:
            ffn_out = batch.tabular_x
        else:
            ffn_out = None

        # concatenate FFN/tabular information with the graph embeddings
        if ffn_out is not None:
            out = torch.cat([out, ffn_out], dim=1)

        # FFN forward pass for the fusion model
        if self.fusion_model is not None:
            out = self.fusionModel(out)

        return out


