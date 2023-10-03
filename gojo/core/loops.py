# Module containing the code typically used to train and evaluate Machine Learning models (e.g.,
# cross-validation subroutines).
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: uncompleted and not functional, still in development
#
import pandas as pd
import numpy as np
import warnings
import joblib
import multiprocessing as mp
import optuna
from functools import partial
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
from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    checkCallable
)


def _getModelPredictions(model: Model, X: np.ndarray,) -> np.ndarray:
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
        _reset_model_fit: bool) -> tuple:
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

    Returns
    -------
    (_n_fold, y_pred_test, y_pred_train, y_true_test, y_true_train, test_idx, train_idx, trained_model) : tuple
        Elements specified according to the input parameters of the method. The tuple will contain sub-tuples
        of two elements, where the first element will identify the information and the second will correspond
        to the information.


    IMPORTANT NOTE: If the input parameter '_reset_model_fit' is set to False the input model will remain trained (
    inplace modifications will take place).
    """
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
        trained_model = deepcopy(_model)

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
        ('trained_model', trained_model)
    )


def evalCrossVal(
        X: np.ndarray or pd.DataFrame,
        y: np.ndarray or pd.DataFrame or pd.Series,
        model: Model,
        cv: RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut,
        verbose: int = -1,
        n_jobs: int = 1,
        save_train_preds: bool = False,
        save_models: bool = False,

    ):
    """ Description """
    checkMultiInputTypes(
        ('X', X, [np.ndarray, pd.DataFrame]),
        ('y', y, [np.ndarray, pd.DataFrame, pd.Series]),
        ('model', model, [Model]),
        ('cv', cv, [RepeatedKFold, RepeatedStratifiedKFold, LeaveOneOut]),
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('save_models', save_models, [bool]),
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
                _reset_model_fit=True     # inplace modifications take place inside this function
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Making predictions...', disable=not show_pbar)
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
                _reset_model_fit=True
            ) for i, (train_idx, test_idx) in tqdm(
                enumerate(cv.split(X_dt.array_data, y_dt.array_data)),
                desc='Making predictions...', disable=not show_pbar)
        )

    # the model should not remain fitted after the execution of the previous subroutines
    if model.is_fitted:
        warnings.warn(
            'Detected a fitted model after cross-validation procedure in "gojo.core.loops.evalCrossVal(...)"')

    # HACk. a little hard-coded...
    cv_report = CVReport(
        raw_results=cv_results,
        X_dataset=X_dt,
        y_dataset=y_dt,
        n_fold_key='n_fold',
        pred_test_key='pred_test',
        true_test_key='true_test',
        pred_train_key='pred_train',
        true_train_key='true_train',
        test_idx_key='test_idx',
        train_idx_key='train_idx',
        trained_model_key='trained_model'
    )

    return cv_report


# TODO. Review implementation
# TODO. Add more input parameters
# TODO. Save HPO statistics
# TODO. Add parameter checking
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
        verbose: int = -1,
        n_jobs: int = 1,
        save_train_preds: bool = False,
        save_models: bool = False):
    """

    Example of 'search_space'
    -----------------------
    >>> search_space = {
    >>>     'max_depth': ( (2, 10), 'suggest_int' ),   # sample from a categorical distribution
    >>>     'max_samples': ( (0.5, 1.0), 'suggest_float' ),   # ... from a uniform distribution
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
            name: getattr(_trial, values[1])(name, *values[0])    # example trial.suggest_int('param_name', (0, 10))
            for name, values in _search_space.items()
        }

        _model = deepcopy(_model)        # avoid inplace modifications
        _model.update(**_optim_params)   # update model parameters

        # perform the nested cross-validation
        _cv_report = evalCrossVal(
            X=_X,
            y=_y,
            model=_model,
            cv=_cv,
            verbose=0,
            n_jobs=1,                # avoid using nested parallel executions
            save_train_preds=_customAggFunction is not None,  # save only if a costume aggregation function was provided
            save_models=False        # does not save models
        )

        # compute performance metrics
        _scores = _cv_report.getScores(metrics=_metrics)

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
        ('verbose', verbose, [int]),
        ('n_jobs', n_jobs, [int]),
        ('save_models', save_models, [bool]),
        ('save_train_preds', save_train_preds, [bool]),
    )

    # check consistency of the search space dictionary
    for i, (param_name, hpo_values) in enumerate(search_space.items()):
        checkMultiInputTypes(
            ('search_space (item %d)' % i, param_name, [str]),
            ('search_space["%s"]' % param_name, hpo_values, [tuple, list]),
            ('search_space["%s"][0]' % param_name, hpo_values[0], [tuple, list]),
            ('search_space["%s"][1]' % param_name, hpo_values[1], [str]))

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
    show_fold_number = False

    # levels > 0 should display the number of the current fold
    if verbose > 0:
        show_fold_number = True

    # train the model optimizing their hyperparameters
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X_dt.array_data, y_dt.array_data)):

        if show_fold_number:    # verbose information
            print('\n============================================= Fold %d\n' % i)

        X_train = X_dt.array_data[train_idx]
        y_train = y_dt.array_data[train_idx]

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

        # save HPO results (Still in development)
        return study

