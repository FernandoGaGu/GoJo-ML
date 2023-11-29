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
from typing import Tuple
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler)

from .io import _createObjectRepresentation
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

    def __repr__(self):
        return _createObjectRepresentation(
            'SimpleSplitter',
            test_size=self.test_size,
            stratify=self.stratify,
            random_state=self.random_state,
            shuffle=self.shuffle
        )

    def __str__(self):
        return self.__repr__()

    def split(
            self,
            X: np.ndarray or pd.DataFrame,
            y: np.ndarray or pd.Series = None) -> Tuple[np.ndarray, np.ndarray]:
        """ Generates indices to split data into training and test set.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.

        y : np.ndarray or pd.Series, default=None
            If `stratify` was specified as `True` this variable will be used for performing a stratified split.
        """
        indices = np.arange(len(X))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y if self.stratify else None,
            random_state=self.random_state,
            shuffle=self.shuffle)

        yield train_idx, test_idx


class InstanceLevelKFoldSplitter(object):
    """ Splitter that allows to make splits at instance level ignoring the observations associated to the instance.

    .. important::
        The observations of the input data of the :meth:`split` method will be associated with the identifiers provided
        in `instance_id`.


    Parameters
    ----------
    n_splits : int
        Number of folds. Must be at least 2.

    instance_id : np.ndarray
        Array identifying the instances to perform the splits.

    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.

    shuffle : bool, default=True
        Indicates whether to shuffle the data before performing the split.

    random_state : int, default=None
        Controls the randomness of each repeated cross-validation instance.
    """

    def __init__(
            self,
            n_splits: int,
            instance_id: np.ndarray,
            n_repeats: int = 1,
            shuffle: bool = True,
            random_state: int = None):

        checkMultiInputTypes(
            ('n_splits', n_splits, [int]),
            ('instance_id', instance_id, [np.ndarray]),
            ('n_repeats', n_repeats, [int]),
            ('shuffle', shuffle, [bool]),
            ('random_state', random_state, [int, type(None)]),
        )
        # check input types
        if n_splits <= 1:
            raise TypeError('"n_splits" must be > 1')
        if n_repeats <= 0:
            raise TypeError('"n_repeats" must be > 0')
        if len(instance_id) <= 2:
            raise TypeError('"instance_id" cannot be <= 2')

        self.n_splits = n_splits
        self.instance_id = instance_id
        self.n_repeats = n_repeats
        self.shuffle = shuffle
        self.random_state = random_state
        self._indices = np.arange(len(instance_id))

        # get the unique ids, and create an id-position(s) hash
        self._unique_instance_id = np.unique(instance_id)
        self._instance_id_hash = {
            _id: np.where(instance_id == _id)[0]
            for _id in self._unique_instance_id
        }

        # generate partitions
        self._train_indices, self._test_indices = self._generateSplits()

        # iterator-level states
        self._current_iteration = 0

    def __repr__(self):
        return _createObjectRepresentation(
            'InstanceLevelKFoldSplitter',
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            shuffle=self.shuffle,
            random_state=self.random_state,
            observations=len(self.instance_id),
            unique_instances=len(self._unique_instance_id),
        )

    def __str__(self):
        return self.__repr__()

    def _generateSplits(self):
        """ Internal method needed to generate the internal splits. """

        # calculate the size of the split
        split_size = int(np.ceil(len(self._unique_instance_id) / self.n_splits))

        # select the random state for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        split_indices = []
        for n_repeat in range(self.n_repeats):
            repeat_split_indices = []

            # transform unique ids to indices
            indices = np.arange(len(self._unique_instance_id))

            # random permutation of the indices
            if self.shuffle:
                indices = np.random.permutation(indices)

            # add split indices
            for n_split in range(self.n_splits):
                repeat_split_indices.append(indices[n_split * split_size:n_split * split_size + split_size])

            # inner checking
            assert len(np.unique(np.concatenate(repeat_split_indices))) == len(
                self._unique_instance_id), 'Inner checking fails (0)'

            # save all folds
            split_indices = split_indices + repeat_split_indices

        # expand the indices to the positions
        unfolded_split_indices = []
        for indices in split_indices:
            unfolded_split_indices.append(
                np.concatenate([
                    self._instance_id_hash[_id]
                    for _id in self._unique_instance_id[indices]])
            )

        # create final train/test folds
        train_indices = []
        test_indices = []
        for rep in range(self.n_repeats):
            for split_i in range(self.n_splits):
                train_indices_ = []
                for split_j in range(self.n_splits):
                    curr_idx = self.n_splits * rep + split_j
                    if split_i == split_j:
                        # select test data
                        test_indices.append(unfolded_split_indices[curr_idx])
                    else:
                        # select train data
                        train_indices_.append(unfolded_split_indices[curr_idx])
                train_indices.append(np.concatenate(train_indices_))

        return train_indices, test_indices

    def split(
            self,
            X: np.ndarray or pd.DataFrame,
            y=None) -> Tuple[np.ndarray, np.ndarray]:
        """ Generate the splits. This function will return a tuple where the first element will correspond to
        the training indices and the second element to the test indices.

        .. important::
            `X` must match with `instance_id`.

        Parameters
        ----------
        X : np.ndarray or pd.DataFrame
            Input data.

        y : object, default=None
            Ignored parameter. Implemented for `sklearn` compatibility.
        """
        indices = np.arange(len(X))

        if len(self.instance_id) != len(indices):
            raise TypeError(
                'Input parameter "instance_id" must be of the same size as the input data.'
                'Provided number of samples "%d", expected "%d"' % (len(indices), len(self.instance_id))
            )

        while self._current_iteration < len(self._train_indices):
            train_indices = self._train_indices[self._current_iteration]
            test_indices = self._test_indices[self._current_iteration]
            self._current_iteration += 1
            yield train_indices, test_indices

        self._current_iteration = 0


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
