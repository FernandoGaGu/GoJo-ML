# Script used to test the code from gojo.core.evaluation
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import sys
import numpy as np
import pandas as pd

sys.path.append('..')

import gojo


def check_scaling():

    x1 = np.random.normal(1, 0.5, size=100)
    x2 = np.random.normal(2, 1.5, size=100)
    x = pd.DataFrame(np.stack([x1, x2]).T)

    """
    min_max_np = gojo.util.minMaxScaling(x, (-1, 1))
    print(np.min(min_max_np, axis=0), np.max(min_max_np, axis=0),
          np.mean(min_max_np, axis=0), np.std(min_max_np, axis=0))
    """

    zscores_np = gojo.util.minMaxScaling(x, (-1, 1))
    print(zscores_np.min())
    print(zscores_np.max())
    print(zscores_np.mean())
    print(zscores_np.std())


if __name__ == '__main__':
    check_scaling()

