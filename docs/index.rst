.. gojo - Documentation documentation master file, created by
   sphinx-quickstart on Sun Oct  8 13:50:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GoJo - Make Machine/Deep Learning pipelines simple
================================================

   Python library with a variety of pipeline implementations typically used to develop and evaluate Deep Learning and Machine Learning models.

**IMPORTANT NOTE**: Library still under development and subjected to changes (this is an alpha version). Check out the github_ page.

This Python library offers a comprehensive suite of pipeline implementations, specifically designed to streamline the entire process of developing and evaluating Deep Learning and Machine Learning models. These pipelines serve as an invaluable resource for researchers, data scientists, and engineers by providing a structured and efficient framework for tasks such as data preprocessing, feature engineering, model training, and performance evaluation. Whether you're working on computer vision, natural language processing, recommendation systems, or any other machine learning domain, this library equips you with the tools and utilities necessary to expedite your project development. With a rich set of modules and customizable components, it caters to a wide range of use cases and simplifies the complex journey of model creation and fine-tuning, ultimately contributing to more robust and accurate predictive models.


.. _github: https://github.com/FernandoGaGu/GoJo-ML


Installation
-------------

First clone the current repository:

.. code:: bash

   git clone git@github.com:FernandoGaGu/Gojo-ML.git


and after cloning the repository run the setup.py script 

.. code:: bash

   cd Gojo-ML
   python setup.py install



Usage
-------------

Below is a basic example of use, for more examples see the notebooks located in the notebook folder. At the moment there are few examples and they are poorly documented, later, as the library stabilizes, more examples will be added.

.. code:: python

   # In this example, an SVM model (implemented in sklearn) is evaluated via cross-validation
   import pandas as pd
   from sklearn import datasets
   from sklearn.svm import SVC
   from gojo import core
   from gojo import util

   # load test dataset (Wine)
   wine_dt = datasets.load_wine()

   # create the target variable. Classification problem 1 vs rest
   y = (wine_dt['target'] == 1).astype(int)  
   X = wine_dt['data']

   # create a simple model using the sklearn interfaz
   model = core.SklearnModelWrapper(
       SVC, kernel='poly', degree=3, coef0=50,
       cache_size=1000, class_weight=None)

   # evaluate the model using cross-validation
   cv_report = core.evalCrossVal(
       X=X, y=y,
       model=model,
       cv=util.getCrossValObj(cv=5, stratified=True, random_state=1997),
       n_jobs=-1)

   # compute classification metrics
   scores = cv_report.getScores(
       core.getDefaultMetrics('binary_classification', bin_threshold=0.5))

   # average scores across folds over the test dataset, and... THIS IS ALL!!
   test_scores = pd.DataFrame(scores['test'].mean(axis=0)).round(decimals=3)



Warranties
-------------

The code used, although it has been reviewed and tested with different test problems where it has been shown to lead to correct solutions, is still under development and it is possible to experience undetected bugs. If any errors appear, please report them to us via issues_.


.. _issues: https://github.com/FernandoGaGu/GoJo-ML/issues


Modules
==================

.. toctree::
   :maxdepth: 4

   gojo.core
   gojo.deepl
   gojo.plotting
   gojo.util
   gojo.experimental


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
