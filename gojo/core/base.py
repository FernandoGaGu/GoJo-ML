# Module with the wrappers used to make the models compatible with the library (also internal interfaces
# are included here).
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed and functional
#
import torch
import pandas as pd
import numpy as np
import warnings
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split

from ..deepl.loops import fitNeuralNetwork
from ..util.validation import (
    checkInputType,
    checkCallable,
    checkMultiInputTypes,
    checkClass
)
from ..util.io import (
    createObjectRepresentation
)
from ..util.tools import (
    none2dict
)
from ..exception import (
    UnfittedEstimator
)


class Dataset(object):
    """ Class representing a dataset.

    Parameters
    ----------
    data : np.ndarray or pd.DataFrame or pd.Series
        Data to be homogenized as a dataset.
    """
    def __init__(self, data: np.ndarray or pd.DataFrame or pd.Series):
        checkInputType('data', data, [np.ndarray, pd.DataFrame, pd.Series])

        var_names = None
        index_values = None
        array_data = None
        in_type = ''
        if isinstance(data, pd.DataFrame):
            array_data = data.values
            var_names = list(data.columns)
            index_values = np.array(data.index.values)
            in_type = 'pandas.DataFrame'
        elif isinstance(data, pd.Series):
            array_data = data.values
            var_names = [data.name]
            index_values = np.array(data.index.values)
            in_type = 'pandas.Series'
        elif isinstance(data, np.ndarray):
            # numpy arrays will not contain var_names
            array_data = data
            index_values = np.array(np.arange(data.shape[0]))
            in_type = 'numpy.array'

        self._array_data = array_data
        self._var_names = var_names
        self._index_values = index_values
        self._in_type = in_type

    def __repr__(self):
        return createObjectRepresentation(
            'Dataset', shape=self._array_data.shape, in_type=self._in_type)

    def __str__(self):
        return self.__repr__()

    @property
    def array_data(self) -> np.ndarray:
        return self._array_data

    @property
    def var_names(self) -> list:
        return self._var_names

    @property
    def index_values(self) -> np.array:
        return self._index_values


class Model(object):
    """
    Base class (interface) used to define a model that can interact with the 'gojo' library.

    Subclasses must define the following methods:

        - train()
            This method is used to fit a given model to the input data. Once the model has been fitted, inside
            this method, the superclass method 'fitted()' must be called; otherwise, the model will not be recognized
            as fitted to any data, and 'performInference()' will raise a 'gojo.exception.UnfittedEstimator' error.

        - performInference()
            Once the model has been fitted using the 'train()' method (when the 'is_fitted' property is called, the
            returned value should be True), this method allows performing inferences on new data.

        - reset()
            This method should reset the inner estimator, forgetting all the data seen.

        - getParameters()
            This method must return a dictionary containing the parameters used by the model. The parameters
            returned by this method will be used to store metadata about the model.

        - updateParameters()
            This method must update the inner parameters of the model.

    This abstract class provides the following properties:

        - parameters -> dict
            Returns the hyperparameters of the model.

        - is_fitted -> bool
            Indicates whether a given model has been fitted (i.e., if the 'train()' method was called).

    And the following methods:

        - fitted()
            This method should be called inside the 'train()' method to indicate that the model was
            fitted to the input data and can now perform inferences using the 'performInference()' subroutine.

        - resetFit()
            This method is used to reset learned model weights.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def performInference(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray or None, **kwargs):
        """ Method used to fit a model to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the model.

        y : np.ndarray or None, optional
            Data labels (optional).

        **kwargs
            Additional training parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs):
        """ Method used to reset the fitted model. """
        raise NotImplementedError

    @abstractmethod
    def getParameters(self) -> dict:
        """ Method that must return the model parameters.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def updateParameters(self, **kwargs):
        """ Method used to update model parameters. """
        raise NotImplementedError

    @property
    def parameters(self) -> dict:
        """ Return the model parameters defined in the 'getParameters()' method.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        params = self.getParameters()

        checkInputType('model.getParameters()', params, [dict])   # check the returned type

        return params

    @property
    def is_fitted(self) -> bool:
        """ Indicates whether the model has been trained by calling the 'train()' method.

        Returns
        -------
        model_fitted : bool
            Returns True if the model was fitted.
        """
        return self._is_fitted

    def update(self, **kwargs):
        """ Method used to update model parameters. """
        self.updateParameters(**kwargs)
        self.resetFit()

    def fitted(self):
        """ Method called to indicate that a given model have been fitted. """
        self._is_fitted = True

    def resetFit(self):
        """ Method used to reset a fitted model. """
        self.reset()
        self._is_fitted = False


class SklearnModelWrapper(Model):
    """ Wrapper used for easy integration of models following the sklearn interface into the 'gojo' library
    and functionality.

    Parameters
    ---------
    model_class : type
        Model following the 'sklearn.base.BaseEstimator' interface. The class provided need not be a subclass
        of the sklearn interface but should provide the basic 'fit()' and 'predict()' (or 'predict_proba()') methods.

    predict_proba : bool, default=False
         Parameter that indicates whether to call the predict_proba() method when making predictions. If this
         parameter is False (default behavior) the 'predict()' method will be called. If the parameter is set to
         True and the model provided does not have the predict_proba method implemented, the 'predict()' method
         will be called and a warning will inform that an attempt has been made to call the predict_proba() method.

    **kwargs
        Additional model hyparameters. This parameters will be passed to the 'model_class' constructor.

    Example
    -------
    >>> from gojo import core
    >>> from sklearn.naive_bayes import GaussianNB
    >>>
    >>> # create model
    >>> model = core.SklearnModelWrapper(
    >>>     GaussianNB, predict_proba=True, priors=[0.25, 0.75])
    >>>
    >>> # train model
    >>> model.train(X, y)    # X and y are numpy.arrays
    >>>
    >>> # perform inference
    >>> y_hat = model.performInference(X_new)    # X_new is a numpy.array
    >>>
    >>> # reset model fitting
    >>> model.resetFit()
    >>> model.is_fitted    # must return False
    """
    def __init__(self, model_class, predict_proba: bool = False, **kwargs):
        super(SklearnModelWrapper, self).__init__()

        checkMultiInputTypes(
            ('model_class', model_class, [ABCMeta]),
            ('predict_proba', predict_proba, [bool]))

        self._model_class = model_class
        self._in_params = kwargs
        self.predict_proba = predict_proba
        self._model_obj = model_class(**kwargs)

    def __repr__(self):
        return createObjectRepresentation(
            'SklearnModelWrapper',
            base_model=str(self._model_class).replace('<class ', '').replace('>', ''),
            model_params=self._in_params,
            predict_proba=self.predict_proba
        )

    def __str__(self):
        return self.__repr__()

    def getParameters(self) -> dict:
        return self._in_params

    def updateParameters(self, **kwargs):
        """ Method used to update the inner model parameters.

        IMPORTANT NOTE: Model parameters should be updated by calling the update() method
        from the model superclass.
        """
        for name, value in kwargs.items():
            self._in_params[name] = value

    def train(self, X: np.ndarray, y: np.ndarray or None, **kwargs):
        """ Method used to fit a model to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the model.

        y : np.ndarray or None, optional
            Data labels (optional).
        """
        self._model_obj = self._model_obj.fit(X, y)

        self.fitted()

    def reset(self):
        self._model_obj = self._model_class(**self._in_params)

    def performInference(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the model predictions.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform inference.

        Returns
        -------
        model_predictions : np.ndarray
            Model predictions associated with the input data.
        """
        if not self.is_fitted:
            raise UnfittedEstimator()

        if self.predict_proba:
            if not getattr(self._model_obj, 'predict_proba'):
                warnings.warn('Input model hasn\'t the predict_proba method implemented')
                predictions = self._model_obj.predict(X)
            else:
                predictions = self._model_obj.predict_proba(X)
        else:
            predictions = self._model_obj.predict(X)

        return predictions


class TorchSKInterface(Model):
    """ Description """
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
        super(TorchSKInterface, self).__init__()

        self.model = model
        self.iter_fn = iter_fn
        self.loss_function = loss_function

        # input classes
        self.optimizer_class = optimizer_class
        self.dataset_class = dataset_class
        self.dataloader_class = dataloader_class

        # input classes initialization parameters
        self.optimizer_kw = none2dict(optimizer_kw)
        self.train_dataset_kw = none2dict(train_dataset_kw)
        self.valid_dataset_kw = none2dict(valid_dataset_kw)
        self.train_dataloader_kw = none2dict(train_dataloader_kw)
        self.valid_dataloader_kw = none2dict(valid_dataloader_kw)
        self.iter_fn_kw = none2dict(iter_fn_kw)

        # other parameters
        self.n_epochs = n_epochs
        self.train_split = train_split
        self.train_split_stratify = train_split_stratify
        self.callbacks = callbacks
        self.metrics = metrics
        self.seed = seed
        self.device = device
        self.verbose = verbose

        # save a copy of the input model for resetting the inner state
        self._in_model = deepcopy(model)
        self._fitting_history = None

        # check model parameters
        self._checkModelParams()

    def _checkModelParams(self):
        # check input parameters
        checkCallable('iter_fn', self.iter_fn)
        checkCallable('loss_function', self.loss_function)
        checkClass('optimizer_class', self.optimizer_class)
        checkClass('dataset_class', self.dataset_class)
        checkClass('dataloader_class', self.dataloader_class)
        checkMultiInputTypes(
            ('model', self.model, [torch.nn.Module]),
            ('n_epochs', self.n_epochs, [int]),
            ('train_split', self.train_split, [float]),
            ('optimizer_kw', self.optimizer_kw, [dict, type(None)]),
            ('train_dataset_kw', self.train_dataset_kw, [dict, type(None)]),
            ('valid_dataset_kw', self.valid_dataset_kw, [dict, type(None)]),
            ('train_dataloader_kw', self.train_dataloader_kw, [dict, type(None)]),
            ('valid_dataloader_kw', self.valid_dataloader_kw, [dict, type(None)]),
            ('iter_fn_kw', self.iter_fn_kw, [dict, type(None)]),
            ('train_split_stratify', self.train_split_stratify, [bool]),
            ('callbacks', self.callbacks, [list, type(None)]),
            ('metrics', self.metrics, [list, type(None)]),
            ('seed', self.seed, [int, type(None)]),
            ('device', self.device, [str]),
            ('verbose', self.verbose, [int]))

    def __repr__(self):
        return createObjectRepresentation(
            'TorchSKInterface',
            **self.getParameters())

    def __str__(self):
        return self.__repr__()

    @property
    def fitting_history(self) -> tuple:
        return self._fitting_history

    def getParameters(self) -> dict:
        params = dict(
            model=self.model,
            iter_fn=self.iter_fn,
            loss_function=self.loss_function,
            n_epochs=self.n_epochs,
            train_split=self.train_split,
            train_split_stratify=self.train_split_stratify,
            optimizer_class=self.optimizer_class,
            dataset_class=self.dataset_class,
            dataloader_class=self.dataloader_class,
            optimizer_kw=self.optimizer_kw,
            train_dataset_kw=self.train_dataset_kw,
            valid_dataset_kw=self.valid_dataset_kw,
            train_dataloader_kw=self.train_dataloader_kw,
            valid_dataloader_kw=self.valid_dataloader_kw,
            iter_fn_kw=self.iter_fn_kw,
            callbacks=self.callbacks,
            metrics=self.metrics,
            seed=self.seed,
            device=self.device,
            verbose=self.verbose)

        return params

    def updateParameters(self, **kwargs):

        raise NotImplementedError('This class not support parameter updates. See alternative classes such as: '
                                  '"gojo.core.ParametrizedTorchSKInterface"')

    def train(self, X: np.ndarray, y: np.ndarray or None, **kwargs):

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

        # create dataloaders
        train_dl = self.dataloader_class(
            self.dataset_class(
                X=X[train_idx],
                y=y[train_idx] if y is not None else y,
                **self.train_dataset_kw),
            **self.train_dataloader_kw)

        valid_dl = self.dataloader_class(
            self.dataset_class(
                X=X[valid_idx],
                y=y[valid_idx] if y is not None else y,
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

    def reset(self):
        self.model = deepcopy(self._in_model)
        self._fitting_history = None

    def performInference(self, X: np.ndarray, batch_size: int = None, **kwargs) -> np.ndarray:

        checkMultiInputTypes(
            ('batch_size', batch_size, (int, type(None))))

        # select the model in inference mode
        self.model = self.model.eval()
        self.model = self.model.to(device=self.device)

        if batch_size is None:
            batch_size = X.shape[0]
        else:
            if batch_size < 0:
                warnings.warn('Batch size cannot be less than 0. Selecting batch size to 1.')
                batch_size = 1
            if batch_size > X.shape[0]:   # maximum batch size will be cast to the input data shape
                batch_size = X.shape[0]

        with torch.no_grad():
            y_pred = []

            # iterate over the input data in batches
            curr_batch_split = 0
            while curr_batch_split < X.shape[0]:
                # select batch predictions
                inX = X[curr_batch_split:(curr_batch_split + batch_size), ...]
                # convert predictions to torch Tensor
                inX = torch.from_numpy(inX).type(torch.float).to(device=self.device)
                # make predictions
                y_hat = self.model(inX).detach().cpu().numpy()
                y_pred.append(y_hat)

                curr_batch_split += batch_size

        y_pred = np.concatenate(y_pred)

        # flatten y_pred with dimensions are (n_samples, 1)
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 1:
            y_pred = y_pred.reshape(-1)

        return y_pred


class ParametrizedTorchSKInterface(TorchSKInterface):
    """ Description """
    _NOT_DEFINED_PARAMETER = '__NOT_DEFINED'

    def __init__(
            self,
            generating_fn: callable,
            gf_params: dict,
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
        # check input parameters
        checkCallable('generating_func', generating_fn)
        checkInputType('gf_params', gf_params, [dict, type(None)])
        gf_params = {} if gf_params is None else gf_params

        # generate the model
        model = generating_fn(**gf_params)

        super(ParametrizedTorchSKInterface, self).__init__(
            model=model,
            iter_fn=iter_fn,
            loss_function=loss_function,
            n_epochs=n_epochs,
            train_split=train_split,
            optimizer_class=optimizer_class,
            dataset_class=dataset_class,
            dataloader_class=dataloader_class,
            optimizer_kw=optimizer_kw,
            train_dataset_kw=train_dataset_kw,
            valid_dataset_kw=valid_dataset_kw,
            train_dataloader_kw=train_dataloader_kw,
            valid_dataloader_kw=valid_dataloader_kw,
            iter_fn_kw=iter_fn_kw,
            train_split_stratify=train_split_stratify,
            callbacks=callbacks,
            metrics=metrics,
            seed=seed,
            device=device,
            verbose=verbose)

        # save parameter-specific values
        self.gf_params = gf_params
        self._generating_fn = generating_fn

    def __repr__(self):
        return createObjectRepresentation(
            'ParametrizedTorchSKInterface',
            **self.getParameters())

    def __str__(self):
        return self.__repr__()

    def updateParameters(self, **kwargs):

        for name, value in kwargs.items():

            if name == 'model':
                warnings.warn(
                    'The internal model is being modified. This will have no effect because the model will'
                    ' be subsequently regenerated.')

            if name.startswith('_'):
                raise KeyError('Parameter "%s" is not accessible.' % name)

            # update dict-level parameters
            if '__' in name:
                name_levels = name.split('__')
                if len(name_levels) != 2:
                    raise TypeError(
                        'If you want to modify the values of an internal dictionary, you must specify the name of '
                        'the dictionary by separating the key with "__" so that two parameters are involved. In this '
                        'case a different number of parameters has been detected: %r' % name_levels)

                dict_name, key = name_levels

                # check that the input parameter was defined
                if getattr(self, dict_name, self._NOT_DEFINED_PARAMETER) is self._NOT_DEFINED_PARAMETER:
                    raise TypeError('Parameter "%s" not found. Review gojo.core.ParametrizedTorchSKInterface '
                                    'documentation.' % dict_name)

                # check that the specified parameter is a dictionary
                checkInputType(dict_name, getattr(self, dict_name), [dict])

                # check if the dictionary key exists
                if key not in getattr(self, dict_name).keys():
                    raise KeyError('Key "%s" not found in parameter "%s".' % (key, dict_name))

                # update parameter
                getattr(self, dict_name)[key] = value

            else:
                # update para-level parameters
                # check that the input parameter was defined
                if getattr(self, name, self._NOT_DEFINED_PARAMETER) is self._NOT_DEFINED_PARAMETER:
                    raise TypeError(
                        'Parameter "%s" not found. Review gojo.core.ParametrizedTorchSKInterface documentation.' % name)

                # set the new value
                setattr(self, name, value)

        # regenerate the model
        self._in_model = self._generating_fn(**self.gf_params)

    def getParameters(self) -> dict:
        params = super().getParameters()
        params['generating_fn'] = self._generating_fn
        params['gf_params'] = self.gf_params
        
        return params








