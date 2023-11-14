# Module containing the code related to the application of transformations to the data prior to the
# training of the models within the functions of 'gojo.core.loops'.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#
import numpy as np
from abc import ABCMeta, abstractmethod
from copy import deepcopy


from ..util.validation import (
    checkInputType,
    checkClass
)
from ..util.io import (
    _createObjectRepresentation
)
from ..exception import UnfittedTransform


class Transform(object):
    """ Base interface for applying transformations to the input data in the :py:mod:`gojo.core.loops` subroutines.
    Internally, the training data will be passed to the :meth:`fit` method for adjusting the transformation to the
    training dataset statistics, and subsequently, the transformation will be applied to the training and test data
    by means of the :meth:`transform`.

    Subclasses must define the following methods:

        - fit()
            Method used to fit a transform to a given input data.

        - transform()
            Method used to perform the transformations to the input data.

        - reset()
            Method used to reset the fitted transform

        - copy()
            Method used to make a copy of the transform.

        - getParameters()
            Method that must return the transform parameters.

        - updateParameters()
            Method used to update the transform parameters.

    This abstract class provides the following properties:

        - is_fitted
            Indicates whether the transformation has been fitted by calling the :meth:`fit` method.

    And the following methods:

        - update()
            Method used to update the transform parameters.

        - fitted()
            Method called (usually internally) to indicate that a given transformation have been fitted.

        - resetFit()
            Method used to reset a fitted transformation (usually called internally).

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray or None, **kwargs):
        """ Method used to fit a transform to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the transformation.

        y : np.ndarray or None, optional
            Data labels (optional).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to perform the transformations to the input data.

        Parameters
        ----------
        X : np.ndarray
            Input data used to perform the transformations.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs):
        """ Method used to reset the fitted transform. """
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        """ Method used to make a copy of the transform. """
        raise NotImplementedError

    @abstractmethod
    def getParameters(self) -> dict:
        """ Method that must return the transform parameters.

        Returns
        -------
        model_parameters : dict
            Model parameters.
        """
        raise NotImplementedError

    @abstractmethod
    def updateParameters(self, **kwargs):
        """ Method used to update the transform parameters. """
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """ Indicates whether the transformation has been fitted by calling the :meth:`fit` method.

        Returns
        -------
        model_fitted : bool
            Returns True if the model was fitted.
        """
        return self._is_fitted

    def update(self, **kwargs):
        """ Method used to update the transform parameters. """
        self.updateParameters(**kwargs)
        self.resetFit()

    def fitted(self):
        """ Method called to indicate that a given transformation have been fitted. """
        self._is_fitted = True

    def resetFit(self):
        """ Method used to reset a fitted transformation. """
        self.reset()
        self._is_fitted = False


class SKLearnTransformWrapper(Transform):
    """ Wrapper used to easily incorporate the transformations implemented in the `sklearn` library.

    Parameters
    ----------
    transform_class : Type
        `sklearn` transform. The instances of this class must have the `fit` and `transform` methods defined according
        to the `sklearn` implementation.

    **kwargs
        Optional arguments used to initialize instances of the provided class.


    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>>
    >>> # GOJO libraries
    >>> import gojo
    >>> from gojo import core
    >>>
    >>> # previous model transforms
    >>> transforms = [
    >>>     core.SKLearnTransformWrapper(StandardScaler),
    >>>     core.SKLearnTransformWrapper(PCA, n_components=5)
    >>> ]
    >>>
    >>> # default model
    >>> model = core.SklearnModelWrapper(
    >>>     SVC, kernel='poly', degree=1, coef0=0.0,
    >>>     cache_size=1000, class_weight=None
    >>> )
    >>>
    >>> cv_report = core.evalCrossVal(
    >>>     X=X, y=y,
    >>>     model=model,
    >>>     cv=gojo.util.getCrossValObj(cv=5),
    >>>     transforms=transforms)
    >>>
    """
    def __init__(self, transform_class, **kwargs):
        super(SKLearnTransformWrapper, self).__init__()

        checkClass('transform_class', transform_class)

        self._transform_class = transform_class
        self._in_params = kwargs
        self._transform_obj = transform_class(**kwargs)

    def __repr__(self):
        return _createObjectRepresentation(
            'SKLearnTransformWrapper',
            base_transform=str(self._transform_class).replace('<class ', '').replace('>', ''),
            transform_params=self._in_params,
        )

    def __str__(self):
        return self.__repr__()

    def getParameters(self) -> dict:
        return self._in_params

    def updateParameters(self, **kwargs):
        """ Method used to update the inner transform parameters.

        IMPORTANT NOTE: Transform parameters should be updated by calling the :meth:`update` method from the
        superclass :class:`gojo.core.transform.Transform`.
        """
        for name, value in kwargs.items():
            self._in_params[name] = value

    def fit(self, X: np.ndarray, y: np.ndarray or None = None, **kwargs):
        """ Method used to fit a transform to a given input data.

        Parameters
        ----------
        X : np.ndarray
            Input data to fit the model.

        y : np.ndarray or None, default=None
            Data labels (optional).
        """
        self._transform_obj = self._transform_obj.fit(X, y)

        self.fitted()

    def reset(self):
        """ Reset the model fit. """
        self._transform_obj = self._transform_class(**self._in_params)

    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """ Method used to apply the transformations.

        Parameters
        ----------
        X : np.ndarray
            Input data to be transformed.

        Returns
        -------
        X_trans : np.ndarray
            Transformer data.
        """
        if not self.is_fitted:
            raise UnfittedTransform()

        return self._transform_obj.transform(X)

    def copy(self):
        """ Make a deepcopy of the instance. """
        return deepcopy(self)
