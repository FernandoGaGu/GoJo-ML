# Script used to test the code from gojo.core.loops
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import sys
import numpy as np
import pandas as pd

sys.path.append('..')

import gojo
from gojo import core


def test_evalCrossVal():
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB
    from sklearn import metrics

    # load test dataset (Wine)
    wine_dt = datasets.load_wine()

    # create the target variable. Classification problem 0 vs rest
    # to see the target names you can use wine_dt['target_names']
    y = (wine_dt['target'] == 2).astype(int)
    X = wine_dt['data']

    cv_report = core.evalCrossVal(
        X=X,
        y=y,
        model=core.SklearnModelWrapper(
            GaussianNB, predict_proba=False, priors=[0.25, 0.75]),
        cv=gojo.util.getCrossValObj(cv=5, loocv=True, stratified=True, random_state=1997),
        verbose=True,
        save_train_preds=True,
        save_models=False,
        n_jobs=1

    )
    scores = cv_report.getScores(
        core.getDefaultMetrics('binary_classification'),
        loocv=True)

    print(scores['test'])


if __name__ == '__main__':
    test_evalCrossVal()
