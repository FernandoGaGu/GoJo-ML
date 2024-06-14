Advanced use
============

This notebook is intended for those users who want to make advanced use
of the module by defining their own advanced functionalities and benefit
from the subroutines already implemented in the **gojo** library.

.. code:: python

    import numpy as np
    from sklearn import datasets

.. code:: python

    # For the tests we will use the test dataset used in Example 1....
    # load test dataset (Wine)
    wine_dt = datasets.load_wine()
    
    # create the target variable. Classification problem 0 vs rest
    # to see the target names you can use wine_dt['target_names']
    y = (wine_dt['target'] == 1).astype(int)  
    X = wine_dt['data']
    
    X.shape, y.shape




.. parsed-literal::

    ((178, 13), (178,))



Definition of your own transformations (gojo.interfaces.Transform)
------------------------------------------------------------------

To define your own transformations you can make use of the
**gojo.interfaces.Transform** class. Let’s see how to define our own
transformations using an example.

In the example we will implement a very naive strategy of feature
selection based on trying different combinations of variables (number of
variables in each combination defined by **n_vars**, and number of
interations specified by **n_iters**) and selecting the combination that
works best. To evaluate the quality of the selected variables we will
use the GaussianNB_\_ model of sklearn.

   For more information use: **help(interfaces.Transform)**

.. code:: python

    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    
    from gojo import interfaces
    from gojo import core
    from gojo import util


.. parsed-literal::

    C:\Users\fgarcia\anaconda3\envs\mlv0\lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


.. code:: python

    class RandomPermutationSelection(interfaces.Transform):
        def __init__(self, n_vars: int, n_iters: int, random_state: int = None):
            super().__init__()    # IMPORTANT. Don't forget to call the superclass constructor
            
            self.n_vars = n_vars
            self.n_iters = n_iters
            self.random_state = random_state
            self.selected_features = None
    
        def fit(self, X: np.ndarray, y: np.ndarray, **_):
    
            # fix the random seed
            np.random.seed(self.random_state)
    
            # create a selection array
            findex = np.arange(X.shape[1])
    
            # iterate over random feature sets
            best_fset = None
            best_score = -np.inf
            for _ in range(self.n_iters):
                binary_mask = np.zeros(shape=X.shape[1])
    
                # random shuffle of findex
                np.random.shuffle(findex)
    
                # get selected features
                binary_mask[findex[:self.n_vars]] = 1
                sel_features = np.where(binary_mask == 1)[0]
    
                # test model performance
                cv_score = cross_val_score(
                    GaussianNB(), 
                    X=X[:, sel_features], 
                    y=y,
                    scoring='f1')
                avg_cv_score = np.mean(cv_score)
                
                # save features
                if avg_cv_score > best_score:
                    best_score = avg_cv_score
                    best_fset = sel_features
                
            self.selected_features = best_fset
    
        def transform(self, X: np.ndarray, **_):
            assert self.selected_features is not None, 'Unfitted transform'
            return X[:, self.selected_features]
    
        def reset(self):
            self.selected_features = None

.. code:: python

    fselector = RandomPermutationSelection(
        n_vars=5, n_iters=500)
    
    fselector.fit(X, y)

.. code:: python

    fselector.selected_features




.. parsed-literal::

    array([ 0,  4,  9, 10, 12], dtype=int64)



.. code:: python

    fselector_copy = fselector.copy()   # test the copy method
    fselector.reset()                   # reset the transform
    fselector.selected_features, fselector_copy.selected_features




.. parsed-literal::

    (None, array([ 0,  4,  9, 10, 12], dtype=int64))



Now that we have implemented our custom transformation, we are going to
introduce it into a cross validation loop by saving the transformations
so that we can explore the selected characteristics of each fold. Here
we are going to use the same model and approach used in the notebook
**Example 1. Model evaluation by cross validation.ipynb**

.. code:: python

    # model definition
    model = interfaces.SklearnModelWrapper(
        model_class=SVC,
        kernel='poly', degree=1, coef0=0.0,
        cache_size=1000, class_weight=None
    )
    
    # cross-validation definition
    cv_obj = util.splitter.getCrossValObj(cv=5, repeats=1, stratified=True)
    
    
    # z-score scaling 
    zscores_scaler = interfaces.SKLearnTransformWrapper(transform_class=StandardScaler)
    
    # put all transformation in a list (they will be applied sequentially)
    transformations = [zscores_scaler, fselector]

.. code:: python

    cv_report = core.evalCrossVal(
        X=X,
        y=y,
        model=model,
        cv=cv_obj,
        save_train_preds=True,
        save_models=True,
        save_transforms=True,    
        transforms=transformations,
        n_jobs=5
    )



.. parsed-literal::

    Performing cross-validation...: 5it [00:00, 363.73it/s]


.. code:: python

    performance = cv_report.getScores(
        core.getDefaultMetrics('binary_classification')
    )
    performance['test']




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>accuracy</th>
          <th>balanced_accuracy</th>
          <th>precision</th>
          <th>recall</th>
          <th>sensitivity</th>
          <th>specificity</th>
          <th>negative_predictive_value</th>
          <th>f1_score</th>
          <th>auc</th>
          <th>n_fold</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.916667</td>
          <td>0.905844</td>
          <td>0.923077</td>
          <td>0.857143</td>
          <td>0.857143</td>
          <td>0.954545</td>
          <td>0.913043</td>
          <td>0.888889</td>
          <td>0.905844</td>
          <td>0</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.916667</td>
          <td>0.892857</td>
          <td>1.000000</td>
          <td>0.785714</td>
          <td>0.785714</td>
          <td>1.000000</td>
          <td>0.880000</td>
          <td>0.880000</td>
          <td>0.892857</td>
          <td>1</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.916667</td>
          <td>0.909524</td>
          <td>0.928571</td>
          <td>0.866667</td>
          <td>0.866667</td>
          <td>0.952381</td>
          <td>0.909091</td>
          <td>0.896552</td>
          <td>0.909524</td>
          <td>2</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.914286</td>
          <td>0.904762</td>
          <td>0.923077</td>
          <td>0.857143</td>
          <td>0.857143</td>
          <td>0.952381</td>
          <td>0.909091</td>
          <td>0.888889</td>
          <td>0.904762</td>
          <td>3</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.942857</td>
          <td>0.952381</td>
          <td>0.875000</td>
          <td>1.000000</td>
          <td>1.000000</td>
          <td>0.904762</td>
          <td>1.000000</td>
          <td>0.933333</td>
          <td>0.952381</td>
          <td>4</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    performance['test'].mean().loc['f1_score']




.. parsed-literal::

    0.8975325670498083



.. code:: python

    fitted_transforms = cv_report.getFittedTransforms()
    fitted_transforms




.. parsed-literal::

    {0: [SKLearnTransformWrapper(
          base_transform='sklearn.preprocessing._data.StandardScaler',
          transform_params={}
      ),
      <__main__.RandomPermutationSelection at 0x27156732f50>],
     1: [SKLearnTransformWrapper(
          base_transform='sklearn.preprocessing._data.StandardScaler',
          transform_params={}
      ),
      <__main__.RandomPermutationSelection at 0x27156731780>],
     2: [SKLearnTransformWrapper(
          base_transform='sklearn.preprocessing._data.StandardScaler',
          transform_params={}
      ),
      <__main__.RandomPermutationSelection at 0x27156732110>],
     3: [SKLearnTransformWrapper(
          base_transform='sklearn.preprocessing._data.StandardScaler',
          transform_params={}
      ),
      <__main__.RandomPermutationSelection at 0x27156732260>],
     4: [SKLearnTransformWrapper(
          base_transform='sklearn.preprocessing._data.StandardScaler',
          transform_params={}
      ),
      <__main__.RandomPermutationSelection at 0x27156732020>]}



Lets explore the selected features in each fold

.. code:: python

    for n_fold, transform in fitted_transforms.items():
        print('Selected features in fold %d: %r' % (n_fold, list(transform[1].selected_features)))


.. parsed-literal::

    Selected features in fold 0: [0, 2, 4, 9, 10]
    Selected features in fold 1: [0, 2, 7, 9, 10]
    Selected features in fold 2: [4, 6, 9, 10, 12]
    Selected features in fold 3: [0, 2, 4, 8, 9]
    Selected features in fold 4: [0, 2, 5, 9, 10]


We have seen that this feature selection, although naive, tends to
select always the same features.
