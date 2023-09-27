# Script used to test the code from deid.core.evaluation
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import sys
import numpy as np

sys.path.append('..')

import deid


def check_metric():
    """ Check deid.core.evaluation.Metric class. """
    # --- create metric for binary problem
    metric = deid.core.Metric(
        name='Test_metric',
        function=lambda _y_true, _y_pred, scale: scale * np.mean(_y_true == _y_pred),   # test function
        bin_threshold=0.5,
        scale=2
    )

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0.8])
    out = metric(y_true, y_pred)    # accuracy 0.75 * 2

    assert out == 1.5, 'Invalid test (1)'

    # --- create metric for multiclass problem (multiclass categorical)
    y_true = np.array([0, 2, 0, 1])
    y_pred = np.array([0, 2, 0, 0])
    metric = deid.core.Metric(
        name='Test_metric_multiclass_cat',
        function=lambda _y_true, _y_pred, scale: scale * np.mean(_y_true == _y_pred),   # test function
        multiclass=True,
        scale=2,
        number_of_classes=3,
        use_multiclass_sparse=False
    )
    out = metric(y_true, y_pred)    # accuracy 0.75 * 2

    assert out == 1.5, 'Invalid test (2)'

    # --- create metric for multiclass problem (multiclass sparse)
    y_true = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

    y_pred = np.array([[0, 0, 1],
                       [0, 1, 0],
                       [0, 1, 0],
                       [1, 0, 0]])

    metric = deid.core.Metric(
        name='Test_metric_multiclass_spa',
        function=lambda _y_true, _y_pred, scale: scale * np.mean(_y_true == _y_pred),   # test function
        multiclass=True,
        scale=1,
        number_of_classes=3,
        use_multiclass_sparse=False
    )
    out = metric(y_true, y_pred)    # accuracy 0.75 * 2

    assert out == 0.75, 'Invalid test (3)'


def check_metrics_aux_functions():
    from sklearn import metrics as sk_metrics

    metrics = deid.core.getDefaultMetrics('binary_classification', ['f1_score', 'auc', 'foo'])
    metric = deid.core.Metric(
        'accuracy',
        deid.core.flatFunctionInput(sk_metrics.accuracy_score),
        bin_threshold=0.5)
    y_pred = np.array([[0.2, 0.9, 0.7, 0.3], [0.2, 0.9, 0.7, 0.3]])
    y_true = np.array([[0, 1, 1, 0], [1, 0, 0, 1]])

    out = metric(y_true=y_true, y_pred=y_pred)

    out = deid.core.getScores(y_true, y_pred, [metric])
    print(out)



if __name__ == '__main__':
    check_metric()
    check_metrics_aux_functions()

