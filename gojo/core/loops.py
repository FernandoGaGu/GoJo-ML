# Module containing the code typically used to train and evaluate Machine Learning models (e.g.,
# cross-validation subroutines).
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed and functional
# TODO. Add documentation for the function 'evalCrossValNestedHPO'
#
import pandas as pd
import numpy as np
import warnings
import joblib
import multiprocessing as mp
import optuna
from functools import partial
from typing import List
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut)

from .base import (
    Model,
    Dataset
)
from .report import (
    CVReport
)
from .transform import (
    Transform
)
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable
)
from ..exception import (
    UnfittedTransform
)


def _getModelPredictions(model: Model, X: np.ndarray) -> np.ndarray:
    """ Subroutine that return the model predictions. Model prediction order resolution:
    The predictions will be returned as numpy.arrays.
    """
    checkMultiInputTypes(
        ('X', X, [np.ndarray]),
        ('model', model, [Model]))

    predictions = model.performInference(X)

    checkInputType('model.performInference() -> out', predictions, [np.ndarray])

    return predictions


def _fitModelAndPredict(model: Model, X_train: np.ndarray, X_test: np.ndarray,
                        y_train: np.ndarray = None) -> np.ndarray:
    """ Subroutine used to fit a model and make the predictions. """
    checkMultiInputTypes(
        ('X_train', X_train, [np.ndarray]),
        ('X_test', X_test, [np.ndarray]),
        ('y_train', y_train, [np.ndarray, type(None)]),
        ('model', model, [Model]))

    if model.is_fitted:
        warnings.warn(
            'Providing a fitted model to "gojo.core.loops._fitModelAndPredict()". The model provided will be '
            'automatically reset using "model.resetFit()" and re-fitted.')
        model.resetFit()

    model.train(X_train, y_train)
    predictions = _getModelPredictions(model=model, X=X_test)

    return predictions


def _applyTransforms(transforms: List[Transform], X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    """ Subroutine that applies the provided transforms.  """
    checkMultiInputTypes(
        ('transforms', transforms, [list]),
        ('X', X, [np.ndarray]),
        ('y', y, [np.ndarray, type(None)]))

    if len(transforms) == 0:
        raise TypeError('Parameter "transformations" is an empty list.')

    for i, transform in enumerate(transforms):
        checkInputType('transformations[%d]' % i, transform, [Transform])
        # check for unfitted transforms
        if not transform.is_fitted:
            raise UnfittedTransform()

        X = transform.transform(X=X, y=y)

    return X


def _fitAndApplyTransforms(transforms: List[Transform], X_train: np.ndarray, X_test: np.ndarray,
                           y_train: np.ndarray = None, y_test: np.ndarray = None) -> tuple:
    """ Subroutine used to fit transforms and make predictions.

    NOTE: This functions performs inplace modification of the input transforms.
    """
    checkMultiInputTypes(
        ('transforms', transforms, [list]),
        ('X_train', X_train, [np.ndarray]),
        ('X_test', X_test, [np.ndarray]),
        ('y_train', y_train, [np.ndarray, type(None)]),
        ('y_test', y_test, [np.ndarray, type(None)]))

    if len(transforms) == 0:
        raise TypeError('Parameter "transformations" is an empty list.')

    for i, transform in enumerate(transforms):
        checkInputType('transformations[%d]' % i, transform, [Transform])

        # check for fitted transforms
        if transform.is_fitted:
            warnings.warn(
                'Providing a fitted transform to "gojo.core.loops._fitTransformsAndApply()". The transform provided '
                'will be automatically reset using "transform.resetFit()" and re-fitted.')
            transform.resetFit()

        # fit the transformations based on the training data, and apply the transformation
        # to the training/test data
        transform.fit(X=X_train, y=y_train)
        X_train = transform.transform(X=X_train, y=y_train)
        X_test = transform.transform(X=X_test, y=y_test)

    return X_train, X_test


def _evalCrossValFold(
        _n_fold: int,
        _model: Model,
        _X_train: np.ndarray,
        _y_train: np.ndarray or None,
        _X_test: np.ndarray,
        _y_test: np.ndarray,
        _train_idx: np.ndarray,
        _test_idx: np.ndarray,
        _predict_train: bool,
        _return_model: bool,
        _reset_model_fit: bool,
        _transforms: list or None,
        _return_transforms: bool,
        _reset_transforms: bool) -> tuple:
    """ Subroutine used internally to train and perform the predictions of a model in relation to a fold. This
    subroutine has been segmented to allow parallelization of training.

    Parameters
    ----------
    _model : gojo.core.base.Model
        Model to be trained and used to make the predictions on '_X_test'.

    _X_train : np.ndarray
        Data used for model training.

    _y_train : np.ndarray or None
        Labels used for model training.

    _X_test : np.ndarray
        Data used for the model to make inferences on new data.

    _predict_train : bool
        Parameter indicating whether to return the predictions on the data used
        to train the models.

    _return_model : bool
        Parameter indicating whether to return a deepcopy of the trained model.

    _reset_model_fit : bool
        Parameter indicating if the model should be reset by calling to the 'resetFit()'
        method.

    _transforms : list or None
        Transformations that will be applied to the data before training the model. These
        transformations will be adjusted based on the training data and will be used to transform
        the training and test data. They will be applied sequentially.

    _return_transforms : bool
        Parameter indicating whether to return the transforms.

    _reset_transforms : bool
        Parameter indicating if the transforms should be reset by calling to the
        'resetFit()' method.

    Returns
    -------
    (_n_fold, y_pred_test, y_pred_train, y_true_test, y_true_train, test_idx, train_idx, trained_model,
    _transforms) : tuple
        Elements specified according to the input parameters of the method. The tuple will contain sub-tuples
        of two elements, where the first element will identify the information and the second will correspond
        to the information.


    IMPORTANT NOTE: If the input parameter '_reset_model_fit' is set to False the input model will remain trained (
    inplace modifications will take place). Applicable also for transforms ('_transforms' and '_reset_transforms'
    parameter).
    """

    # fit transformations to the training data and apply to the test data
    if _transforms is not None:
        _X_train, _X_test = _fitAndApplyTransforms(
            transforms=_transforms, X_train=_X_train, X_test=_X_test, y_train=_y_train, y_test=_y_test)

    # train the model and make the predictions on the test data
    y_pred_test = _fitModelAndPredict(
        model=_model, X_train=_X_train, X_test=_X_test, y_train=_y_train)

    # make predictions on the training data and save training data information
    y_pred_train = None
    y_true_train = None
    train_idx = None
    if _predict_train:
        y_pred_train = _getModelPredictions(model=_model, X=_X_train)
        y_true_train = _y_train
        train_idx = _train_idx

    trained_model = None
    if _return_model:
        trained_model = _model.copy()

    transforms = None
    if _return_transforms and _transforms is not None:
        transforms = [_trans.copy() for _trans in _transforms]

    # reset transforms
    if _reset_transforms and _transforms is not None:
        for _transform in _transforms:
            _transform.resetFit()

    # reset trained model
    if _reset_model_fit:
        _model.resetFit()

    return (
        ('n_fold', _n_fold),
        ('pred_test', y_pred_test),
        ('pred_train', y_pred_train),
        ('true_test', _y_test),
        ('true_train', y_true_train),
        ('test_idx', _test_idx),
        ('train_idx', train_idx),
        ('trained_model', trained_model),
        ('transforms', transforms)
    )


def _createCVReport(cv_results: list, X_dataset, y_dataset) -> CVReport:
    # HACk. a little hard-coded...
    cv_report = CVReport(
        raw_results=cv_results,
        X_dataset=X_dataset,
        y_dataset=y_dataset,
        n_fold_key='n_fold',
        pred_test_key='pred_test',
        true_test_key='true_test',
        pred_train_key='pred_train',
        true_train_key='true_train',
        test_idx_key='test_idx',
        train_idx_key='train_idx',
        trained_model_key='trained_model',
        fitted_transforms_key='transforms'
    )

    return cv_report


# TODO. Document function
def evalCrossVal(
        X: np.ndarray or pd.DataFrame,
        y: np.ndarray or pd.DataFrame or pd.Series,
        model: Model,
        cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut,
        transforms: List[Transform] or None = None,
        verbose: int = -1,
        n_jobs: int = 1,
        save_train_preds: bool = False,
        save_transforms: bool = False,
        save_models: bool = False,

    ):
    """ Description """
    checkMultiInputTypes(
        ('X', X, [np.ndarray, pd.DataFrame]),
        ('y', y, [np.ndarray, pd.DataFrame, pd.Series]),
        ('model', model, [Model]),
        ('cv', cv, [RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut]),
        ('transforms', transforms, [list, type(None)]),
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('save_models', save_models, [bool]),
        ('save_transforms', save_transforms, [bool]),
        ('save_train_preds', save_train_preds, [bool]),
    )

    # create the model datasets
    X_dt = Dataset(X)
    y_dt = Dataset(y)

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all
    show_pbar = False

    # levels > 0 should display a tqdm loading bar
    if verbose > 0:
        show_pbar = True

    # train the model and make the predictions according to the cross-validation
    # schema provided
    if n_jobs == 1:
        cv_results = [
            _evalCrossValFold(
                _n_fold=i,
                _model=model,
                _X_train=X_dt.array_data[train_idx],
                _y_train=y_dt.array_data[train_idx],
                _X_test=X_dt.array_data[test_idx],
                _y_test=y_dt.array_data[test_idx],
                _train_idx=train_idx,
                _test_idx=test_idx,
                _predict_train=save_train_preds,
                _return_model=save_models,
                _reset_model_fit=True,     # inplace modifications take place inside this function
                _transforms=transforms,
                _return_transforms=save_transforms,
                _reset_transforms=True     # inplace modifications take place inside this function
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Performing cross-validation...', disable=not show_pbar)
        ]
    else:
        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        if n_jobs <= 0:
            raise TypeError(
                'Parameter "n_jobs" cannot be less than 0 (only -1 is allowed indicating use all cpu cores).')

        cv_results = joblib.Parallel(n_jobs=n_jobs, backend='loky')(
            joblib.delayed(_evalCrossValFold)(
                _n_fold=i,
                _model=model,
                _X_train=X_dt.array_data[train_idx],
                _y_train=y_dt.array_data[train_idx],
                _X_test=X_dt.array_data[test_idx],
                _y_test=y_dt.array_data[test_idx],
                _train_idx=train_idx,
                _test_idx=test_idx,
                _predict_train=save_train_preds,
                _return_model=save_models,
                # inplace modifications will not take place inside this function so save the computation setting this
                # to False, but... better prevent
                _reset_model_fit=True,     # inplace modifications take place inside this function
                _transforms=transforms,
                _return_transforms=save_transforms,
                _reset_transforms=True     # inplace modifications take place inside this function
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Performing cross-validation...', disable=not show_pbar)
        )

    # the model should not remain fitted after the execution of the previous subroutines
    if model.is_fitted:
        warnings.warn(
            'Detected a fitted model after cross-validation procedure in "gojo.core.loops.evalCrossVal(...)"')

    cv_report = _createCVReport(
        cv_results=cv_results,
        X_dataset=X_dt,
        y_dataset=y_dt,
    )

    return cv_report


# TODO. Add documentation
def evalCrossValNestedHPO(
        X: np.ndarray or pd.DataFrame,
        y: np.ndarray or pd.DataFrame or pd.Series,
        model: Model,
        search_space: dict,
        outer_cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut,
        inner_cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut,
        hpo_sampler: optuna.samplers.BaseSampler,
        hpo_n_trials: int,
        minimization: bool,
        metrics: list,
        objective_metric: str = None,
        agg_function: callable = None,
        transforms: List[Transform] or None = None,
        verbose: int = -1,
        n_jobs: int = 1,
        save_train_preds: bool = False,
        save_transforms: bool = False,
        save_models: bool = False):
    """

    Example of 'search_space'
    -----------------------
    >>> search_space = {
    >>>     'max_depth': ('suggest_int', (2, 10)),           # sample from a categorical distribution
    >>>     'max_samples': ('suggest_float', (0.5, 1.0) ),   # ... from a uniform distribution
    >>> }

    """
    def _trialHPO(
            _trial,
            _X: np.ndarray,
            _y: np.ndarray,
            _model: Model,
            _search_space: dict,
            _cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut,
            _metrics: list,
            _minimize: bool,
            _objective_metric: str = None,
            _customAggFunction: callable = None) -> float:
        """ Subroutine used to run a HPO trial. """

        if _objective_metric is None and _customAggFunction is None:
            raise TypeError(
                'gojo.core.loops.evalCrossValNestedHPO._trialHPO(). Either "_objective_metric" or "_customAggFunction" '
                'should be defined')

        # sample parameters from the trial distribution
        _optim_params = {
            name: getattr(_trial, values[0])(name, *values[1])    # example trial.suggest_int('param_name', (0, 10))
            for name, values in _search_space.items()
        }

        _model = model.copy()        # avoid inplace modifications
        _model.update(**_optim_params)   # update model parameters

        # perform the nested cross-validation
        _cv_report = evalCrossVal(
            X=_X,
            y=_y,
            model=_model,
            cv=_cv,
            transforms=None,
            verbose=0,
            n_jobs=1,                # avoid using nested parallel executions
            save_train_preds=_customAggFunction is not None,  # save only if a costume aggregation function was provided
            save_models=False,        # does not save models
            save_transforms=False
        )

        # compute performance metrics
        _scores = _cv_report.getScores(metrics=_metrics, supress_warnings=True)

        if _customAggFunction is not None:
            # use a custom aggregation function to aggregate the fold results, the input for this
            # function will correspond to the scores returned by the gojo.core.report.CVReport.getScores
            # function
            _objective_score = _customAggFunction(_scores)
        else:
            # by default consider the average value of the specified function over the test set
            assert 'test' in _scores.keys(), 'Internal error in gojo.core.loops.evalCrossValNestedHPO._trialHPO. ' \
                                             'Missing "test" key in CVReport.getScores keys.'
            # select the test scores
            _test_scores = _scores['test']

            # check that the specified metric exists
            if _objective_metric not in _test_scores.columns:
                raise TypeError('Missing metric "%s". Available metrics are: %r' % (
                    _objective_metric, _test_scores.columns.tolist()))

            _objective_score = _test_scores[_objective_metric].mean()

        # by default optuna perform a minimization
        if not _minimize:
            _objective_score = -1 * _objective_score

        if not isinstance(_objective_score, (int, float)):
            raise TypeError(
                'Returned score used to optimize model hyperparameters should be a scalar. '
                'Returned type: {}'.format(type(_objective_score)))

        return float(_objective_score)

    # check provided input types
    checkMultiInputTypes(
        ('X', X, [np.ndarray, pd.DataFrame]),
        ('y', y, [np.ndarray, pd.DataFrame, pd.Series]),
        ('model', model, [Model]),
        ('search_space', search_space, [dict]),
        ('outer_cv', outer_cv, [RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut]),
        ('inner_cv', inner_cv, [RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut]),
        ('hpo_sampler', hpo_sampler, [optuna.samplers.BaseSampler]),
        ('metrics', metrics, [list]),
        ('objective_metric', objective_metric, [str, type(None)]),
        ('hpo_n_trials', hpo_n_trials, [int]),
        ('minimization', minimization, [bool]),
        ('transforms', transforms, [list, type(None)]),
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('save_models', save_models, [bool]),
        ('save_transforms', save_transforms, [bool]),
        ('save_train_preds', save_train_preds, [bool]),
    )

    # check consistency of the search space dictionary
    for i, (param_name, hpo_values) in enumerate(search_space.items()):
        checkMultiInputTypes(
            ('search_space (item %d)' % i, param_name, [str]),
            ('search_space["%s"]' % param_name, hpo_values, [tuple, list]),
            ('search_space["%s"][0]' % param_name, hpo_values[0], [str]),
            ('search_space["%s"][1]' % param_name, hpo_values[1], [tuple, list]))

    # check the provided aggregation function
    if agg_function is not None:
        checkCallable('agg_function', agg_function)

    # check number of jobs
    if n_jobs == -1:
        n_jobs = mp.cpu_count()

    if n_jobs <= 0:
        raise TypeError(
            'Parameter "n_jobs" cannot be less than 0 (only -1 is allowed indicating use all cpu cores).')

    # create the model datasets
    X_dt = Dataset(X)
    y_dt = Dataset(y)

    # verbose parameters
    verbose = np.inf if verbose < 0 else verbose   # negative values indicate activate all

    # levels > 0 should display the number of the current fold
    show_fold_number = False
    show_best_combinations = False
    show_hpo_best_values = False

    # verbosity 1 to show pbar
    show_pbar = False
    if verbose == 1:
        show_pbar = True

    # verbosity greater than 1
    if verbose  > 1:
        show_fold_number = True
        show_best_combinations = True
        show_hpo_best_values = True

    # verbosity grater than 2 to show optuna logs
    if verbose < 2:
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # supress optuna warnings below verbosity level <= 1

    # train the model optimizing their hyperparameters
    hpo_trials_history = {}
    hpo_trials_best_params = {}
    fold_stats = []   # used to init the gojo.core.report.CVReport instance
    for i, (train_idx, test_idx) in tqdm(
            enumerate(outer_cv.split(X_dt.array_data, y_dt.array_data)),
            desc='Performing cross-validation...',
            disable=not show_pbar):

        if show_fold_number:    # verbose information
            print('\nFold %d =============================================\n' % (i+1))

        # extract train/test data
        X_train = X_dt.array_data[train_idx]
        y_train = y_dt.array_data[train_idx]
        X_test = X_dt.array_data[test_idx]
        y_test = y_dt.array_data[test_idx]

        transforms_ = None
        if transforms is not None:
            # TODO. Another option is to apply the transforms inside the HPO, but
            # TODO. it can become a very computationally-intensive alternative...
            # apply transformations based on the training data (DESIGN DECISION)
            if save_transforms:
                # fit a copy of the input transformations
                transforms_ = [trans.copy() for trans in transforms]
            else:
                # reset fit and allow inplace modifications of the input transforms
                for transform in transforms:
                    transform.resetFit()
                transforms_ = transforms

            # fit and apply the input transformations based on the training data
            X_train, X_test = _fitAndApplyTransforms(
                transforms=transforms_, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        # create a partial initialization of the function to optimize
        partial_trialHPO = partial(
            _trialHPO,
            _X=X_train,
            _y=y_train,
            _model=model,
            _search_space=search_space,
            _cv=inner_cv,
            _metrics=metrics,
            _minimize=minimization,
            _objective_metric=objective_metric,
            _customAggFunction=agg_function
        )

        # create the optuna study instance
        # deepcopy the provided sampler to avoid inplace modifications
        study = optuna.create_study(sampler=deepcopy(hpo_sampler))
        study.optimize(partial_trialHPO, n_trials=hpo_n_trials, n_jobs=n_jobs)

        # save HPO results
        hpo_trials_history[i] = study.trials_dataframe()
        hpo_trials_best_params[i] = study.best_params

        # display verbosity information
        if show_hpo_best_values:
            study_df = study.trials_dataframe()
            print('Best trial: %d' % study_df.iloc[np.argmin(study_df['value'].values)].loc['number'])
            print('Best value: %.5f' % study_df.iloc[np.argmin(study_df['value'].values)].loc['value'])
            print()

        if show_best_combinations:
            print('Optimized model hyperparameters: {}\n'.format(study.best_params))

        # update input model hyperparameters
        optim_model = model.copy()
        optim_model.update(**study.best_params)

        # train the model and make the predictions on the test data
        fold_results = _evalCrossValFold(
            _n_fold=i,
            _model=optim_model,
            _X_train=X_train,
            _y_train=y_train,
            _X_test=X_test,
            _y_test=y_test,
            _train_idx=train_idx,
            _test_idx=test_idx,
            _predict_train=save_train_preds,
            _return_model=save_models,
            _reset_model_fit=True,
            _transforms=None,   # transforms were applied at the beginning of the loop
            _return_transforms=False,
            _reset_transforms=False
        )

        # add transforms to the returned fold results
        if save_transforms:
            fold_results = list(fold_results)   # convert to list for inplace modifications
            for idx, (name, _) in enumerate(fold_results):
                # replace 'transforms' key
                if name == 'transforms':
                    fold_results[idx] = (name, transforms_)
            fold_results = tuple(fold_results)

        fold_stats.append(fold_results)

    # the model should not remain fitted after the execution of the previous subroutines
    if model.is_fitted:
        warnings.warn(
            'Detected a fitted model after cross-validation procedure in "gojo.core.loops.evalCrossVal(...)"')

    cv_report = _createCVReport(
        cv_results=fold_stats,
        X_dataset=X_dt,
        y_dataset=y_dt)

    # add HPO metadata
    cv_report.addMetadata(
        hpo_history=hpo_trials_history,
        hpo_best_params=hpo_trials_best_params,
    )

    return cv_report




