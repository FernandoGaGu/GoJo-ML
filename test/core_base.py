# Script used to test the code from gojo.core.base
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


def check_dt():
    def _printDataset(_dt):
        print('.array_data', _dt.array_data)
        print('.var_names', _dt.var_names)
        print('.index_values', _dt.index_values)

    gojo.core.evalCrossVal(
        np.ndarray([1]),
        np.ndarray([1]),
        gojo.core.Model(),
        True,
        None
    )

    df = pd.DataFrame([{'foo': 1, 'foo2': 2}, {'foo': 45, 'foo2': 10}])

    dt = gojo.core.Dataset(df)
    print(dt)

    arr = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [4, 5, 6]
    ])
    dt2 = gojo.core.Dataset(arr)
    print(dt2)

    dt3 = gojo.core.Dataset(df['foo'])
    print(dt3)
    _printDataset(dt3)


def check_model():
    from sklearn import datasets
    from sklearn.naive_bayes import GaussianNB

    model = core.SklearnModelWrapper(
        GaussianNB, predict_proba=True, priors=[0.25, 0.75])

    print(model)


if __name__ == '__main__':
    #check_dt()
    check_model()
