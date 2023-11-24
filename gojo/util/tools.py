# Module with various functionalities used by the library.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: uncompleted and not functional, still in development
#
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler)


from .validation import (
    checkMultiInputTypes,
    checkInputType)


class SimpleSplitter(object):
    """ Wrapper of the sklearn `sklearn.model_selection.train_test_split` function used to perform a simple partitioning
    of the data into a training and a test set (optionally with stratification).


    Parameters
    ----------
    test_size : float
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test
        split. If int, represents the absolute number of test samples.

    stratify : bool, default=False
        If not False, data is split in a stratified fashion, using this as the class labels.

    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting. If shuffle=False then stratify must be None.


    Examples
    --------
    >>> import numpy as np
    >>> from gojo import util
    >>>
    >>> np.random.seed(1997)
    >>>
    >>> n_samples = 20
    >>> n_feats = 10
    >>> X = np.random.uniform(size=(n_samples, n_feats))
    >>> y = np.random.randint(0, 2, size=n_samples)
    >>>
    >>> splitter = util.SimpleSplitter(
    >>>     test_size=0.2,
    >>>     stratify=True,
    >>>     random_state=1997
    >>> )
    >>>
    >>> for train_idx, test_idx in splitter.split(X, y):
    >>>     print(len(train_idx), y[train_idx].mean())
    >>>     print(len(test_idx), y[test_idx].mean())

    """

    def __init__(
            self,
            test_size: float,
            stratify: bool = False,
            random_state: int = None,
            shuffle: bool = True):
        self.test_size = test_size
        self.stratify = stratify
        self.random_state = random_state
        self.shuffle = shuffle

    def split(
            self,
            X: np.ndarray or pd.DataFrame,
            y=None) -> np.ndarray:
        """ Generates indices to split data into training and test set. """
        indices = np.arange(len(X))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y if self.stratify else None,
            random_state=self.random_state,
            shuffle=self.shuffle)

        yield train_idx, test_idx


def getCrossValObj(cv: int = None, repeats: int = 1, stratified: bool = False, loocv: bool = False,
                   random_state: int = None) -> RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut:
    """ Function used to obtain the sklearn class used to perform an evaluation of the models according
    to the cross-validation or leave-one-out cross-validation (LOOCV) schemes.

    Parameters
    ----------
    cv : int, default=None
        (cross-validation) This parameter is used to specify the number of folds. Ignored when loocv
        is set to True.

    repeats : int, default=1
        (cross-validation) This parameter is used to specify the number of repetitions of an N-repeats
        cross-validation. Ignored when loocv is set to True.

    stratified : bool, default=False
        (cross-validation) This parameter is specified whether to perform the cross-validation with class
        stratification. Ignored when loocv is set to True.

    loocv : bool, default=False
        (Leave-one-out cross validation) Indicates whether to perform a LOOCV. If this parameter is set to
        True the rest of the parameters will be ignored.

    random_state : int, default=None
        (cross-validation) Random state for study replication.

    Returns
    -------
    cv_obj : RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut
        Cross-validation instance from the `sklearn <https://scikit-learn.org/stable/modules/cross_validation.html>`_
        library.
    """

    checkMultiInputTypes(
        ('cv', cv, [int, type(None)]),
        ('repeats', repeats, [int, type(None)]),
        ('stratified', stratified, [bool]),
        ('loocv', loocv, [bool]),
        ('random_state', random_state, [int, type(None)]))

    if loocv:
        return LeaveOneOut()
    else:
        if cv is None:
            raise TypeError(
                'Parameter "cv" in "gojo.util.tools.getCrossValObj()" must be selected to a integer if '
                'loocv is set to False.')

        if stratified:
            return RepeatedStratifiedKFold(n_repeats=repeats, n_splits=cv, random_state=random_state)
        else:
            return RepeatedKFold(n_repeats=repeats, n_splits=cv, random_state=random_state)


def _applyScaling(data: np.ndarray or pd.DataFrame, scaler, **scaler_args) -> pd.DataFrame or np.ndarray:
    """ General subroutine for feature scaling. """
    col_names, index_names = None, None
    if isinstance(data, pd.DataFrame):
        col_names = data.columns
        index_names = data.index

    if isinstance(data, pd.Series):
        raise TypeError('pandas.Series not yet supported.')

    scaled_data = scaler(**scaler_args).fit_transform(data)

    if isinstance(data, pd.DataFrame):
        scaled_data = pd.DataFrame(scaled_data, columns=col_names, index=index_names)

    return scaled_data


def minMaxScaling(data: pd.DataFrame or np.ndarray, feature_range: tuple = (0, 1)) -> pd.DataFrame or np.ndarray:
    """ Apply a min-max scaling to the provided data range.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to be scaled.

    feature_range : tuple, default=(0, 1)
        Feature range to scale the input data

    Returns
    -------
    scaled_data : pd.DataFrame or np.ndarray
        Data scaled to the provided range.
    """
    return _applyScaling(data, MinMaxScaler, feature_range=feature_range)


def zscoresScaling(data: pd.DataFrame or np.ndarray) -> pd.DataFrame or np.ndarray:
    """ Apply a z-scores scaling to the provided data range.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to be scaled.

    Returns
    -------
    scaled_data : pd.DataFrame or np.ndarray
        Z-scores
    """
    return _applyScaling(data, StandardScaler)


def _none2dict(v):
    return {} if v is None else v


def getNumModelParams(model: torch.nn.Module) -> int:
    """ Function that returns the number of trainable parameters from a :class:`torch.nn.Module` instance. """
    checkInputType('model', model, [torch.nn.Module])

    return sum(param.numel() for param in model.parameters())
