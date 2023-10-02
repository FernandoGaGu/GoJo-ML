# Module with the wrappers used to make the models compatible with the library (also internal interfaces
# are included here).
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed and functional
#
import pandas as pd
import numpy as np
import warnings
from abc import ABCMeta, abstractmethod

from ..util.validation import (
    checkInputType,
    checkMultiInputTypes
)
from ..util.io import (
    createObjectRepresentation
)
from ..exception import (
    UnfittedEstimator
)


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


