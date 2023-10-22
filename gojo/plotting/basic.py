# Module with plotting functions.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#
import pandas as pd
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)


def linePlot(
        *dfs,
        x: str,
        y: str,
        err: str = None,
        err_alpha: float = 0.3,
        labels: list = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (6, 3.5),
        style: str = 'ggplot',
        dpi: int = 100,
        colors: list = None,
        title: str = '',
        title_size: int or float = 15,
        title_pad: int = 15,
        legend_pos: str = 'upper right',
        legend_size: int or float = 12,
        xlabel_size: float or int = 13,
        ylabel_size: float or int = 13,
        grid_alpha: float = 0.5,
        lw: float or int or list = None,
        ls: str or list = None,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Line plot function.

    Parameters
    ----------
    *dfs : pd.DataFrame
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        Y-axis variable. Must be present in the input dataframes.

    err : str
        Variable indicating the errors associated with the lines. Must be present in the input
        dataframes.

    err_alpha : float, default=0.3
        Opacity used to plot the errors.

    labels : list, default=None
        Labels used for identifying the input dataframes.

    ax : matplotlib.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(6, 3.5)
        Figure size.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    dpi : int, default=100
        Figure dpi.

    colors : list, default=None
        Colors used for identifying the dataframe information.

    title : str, default=''
        Plot title.

    title_size : int or float, default=0.5
        Title font size.

    title_pad : int, default=15
        Title pad.

    legend_pos : str, default='upper right'
        Legend position.

    legend_size : int, default=12
        Legend size.

    xlabel_size : float or int, default=13
        X-axis label size.

    ylabel_size : float ot int, default=13
        Y-axis label size.

    grid_alpha : float, default=0.5
        Grid opacity.

    lw : float or int or list, default=None
        Line(s) width(s).

    ls : str or list, default=None
        Line(s) styles(s).

    save : str, default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict, default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool, default=True
        Parameter indicating whether to save the generated plot.

    Examples
    --------
    >>> from gojo import plotting
    >>>
    >>> # train_info, test_info are pandas dataframes returned by gojo.deepl.fitNeuralNetwork
    >>> plotting.linePlot(
    >>>     train_info, valid_info,
    >>>     x='epoch', y='loss (mean)', err='loss (std)',
    >>>     labels=['Train', 'Validation'],
    >>>     title='Model convergence',
    >>>     ls=['solid', 'dashed'],
    >>>     style='default', legend_pos='center right')
    >>>
    """

    checkMultiInputTypes(
        ('x', x, [str]),
        ('y', y, [str]),
        ('err', err, [str, type(None)]),
        ('err_alpha', err_alpha, [float]),
        ('labels', labels, [list, type(None)]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('figsize', figsize, [tuple]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('colors', colors, [list, type(None)]),
        ('grid_alpha', grid_alpha, [float]),
        ('lw', lw, [list, float, int, type(None)]),
        ('ls', ls, [list, str, type(None)]),
        ('dpi', dpi, [int]),
        ('style', style, [str]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # check input data types
    for i, df in enumerate(dfs):
        checkInputType('df (%d)' % i, df, [pd.DataFrame])
        if x not in df.columns:
            raise TypeError('Missing "x" variable "%s". Available variables are: %r' % (x, list(df.columns)))

        if y not in df.columns:
            raise TypeError('Missing "y" variable "%s". Available variables are: %r' % (y, list(df.columns)))

        if not (err is None or err in df.columns):
            raise TypeError('Missing "err" variable "%s". Available variables are: %r' % (err, list(df.columns)))

    if labels is None:
        labels = ['(%d)' % (i + 1) for i in range(len(dfs))]

    if lw is None:
        lw = [None] * len(dfs)
    elif isinstance(lw, (float, int)):
        lw = [lw] * len(dfs)

    if len(dfs) != len(lw):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "lw" (%d)' % (len(dfs), len(lw)))

    if ls is None:
        ls = ['solid'] * len(dfs)
    elif isinstance(ls, str):
        ls = [ls] * len(dfs)

    if len(dfs) != len(ls):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "ls" (%d)' % (len(dfs), len(ls)))

    if len(dfs) != len(labels):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "legend_labels" (%d)' % (len(dfs), len(labels)))

    if not (colors is None or len(dfs) == len(colors)):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "colors" (%d)' % (len(dfs), len(colors)))

    # plot information
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        for i, (label, df) in enumerate(zip(labels, dfs)):
            # select color (if specified)
            color = None if colors is None else colors[i]

            # plot line
            ax.plot(
                df[x].values, df[y].values, label=label, lw=lw[i], ls=ls[i], color=color)

            # plot error
            if err is not None:
                ax.fill_between(
                    df[x].values,
                    df[y].values + df[err].values,
                    df[y].values - df[err].values,
                    color=color, alpha=err_alpha)

        # figure layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=grid_alpha)
        ax.legend(loc=legend_pos, prop=dict(size=legend_size))
        ax.set_xlabel(x, size=xlabel_size)
        ax.set_ylabel(y, size=ylabel_size)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()


def confusionMatrix(
        df: pd.DataFrame,
        y_pred: str,
        y_true: str,
        average_folds: str = None,
        y_pred_threshold: float or None = 0.5,
        normalize: bool = True,
        labels: list = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (5, 4),
        dpi: int = 100,
        cmap: str = 'Blues',
        alpha: float = 0.7,
        cm_font_size: int = 14,
        xaxis_label: str = None,
        yaxis_label: str = None,
        axis_label_size: int = 15,
        axis_label_pad: int = 15,
        axis_tick_size: int = 12,
        title: str = '',
        title_size: int = 15,
        title_pad: int = 15,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Function used to represent a confusion matrix from a pandas DataFrame with predictions and true values (e.g.,
    returned by methods :meth:`gojo.core.report.CVReport.getTestPredictions` and
    :meth:`gojo.core.report.CVReport.getTrainPredictions`).

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame with the model predictions.

        Example
        -------
        >>> df
            Out[0]
                                    pred_labels  true_labels
            n_fold indices
            0      2                0.0          0.0
                   6                0.0          0.0
                   11               0.0          0.0
                   12               0.0          0.0
                   13               0.0          0.0
            ...                     ...          ...
            4      987              0.0          0.0
                   992              0.0          0.0
                   1011             0.0          0.0
                   1016             0.0          0.0
                   1018             0.0          0.0

            [1020 rows x 2 columns]
        >>>

    y_pred : str
        Variable indicating which values are predicted by the model.

    y_true : str
        Variable indicating which values are the ground truth.

    average_folds : str, default=None
        Variable that stratifies the predictions (e.g.n at the folds level) to represent the mean and standard deviation
        values of the confusion matrix.

    y_pred_threshold : float or None, default=0.5
        Threshold to be used to binarize model predictions.

    normalize : bool, default=True
        Parameter indicating whether to express the normalized confusion matrix (as a percentage).

    labels : list, default=None
        Labels used to identify the classes. By default, they will be C0, C1, ..., CX.

    ax : matplotlib.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(5, 4)
            Figure size.

    dpi : int, default=100
            Figure dpi.

    cmap : str, default='Blues'
        Colormap.

    alpha : float, default=0.7
        Plot opacity.

    cm_font_size : int, default=14
        Confusion matriz font size.

    xaxis_label : str, default=None
        X-axis label.

    yaxis_label : str, default=None
        Y-axis label.

    axis_label_size : int, default=15
        XY-axis label size.

    axis_label_pad : int, default=15
        XY-axis pad.

    axis_tick_size : int, default=12
        XY-ticks size.

    title : str, default=''
        Title.

    title_size : int, default=15
        Title size.

    title_pad : int, default=15
        Title pad.

    save : str, default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict, default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool, default=True
        Parameter indicating whether to save the generated plot.

    Examples
    --------
    >>> from gojo import core
    >>> from gojo import plotting
    >>>
    >>> # ... data loading and model definition
    >>>
    >>> # perform the cross validation
    >>> cv_report = core.evalCrossVal(
    >>>     X=X,
    >>>     y=y,
    >>>     model=model,
    >>>     cv=util.getCrossValObj(cv=5)
    >>> )
    >>>
    >>> # get the model predictions on the test data
    >>> predictions = cv_report.getTestPredictions()
    >>>
    >>> # plot the confusion matrix
    >>> plotting.confusionMatrix(
    >>>     df=predictions,
    >>>     y_pred='pred_labels',
    >>>     y_true='true_labels',
    >>>     average_folds='n_fold',
    >>>     normalize=True,
    >>>     labels=['Class 1', 'Class 2'],
    >>>     title='Confusion matrix',
    >>> )
    >>>
    """
    checkMultiInputTypes(
        ('df', df, [pd.DataFrame]),
        ('y_pred', y_pred, [str]),
        ('y_true', y_true, [str]),
        ('average_folds', average_folds, [str, type(None)]),
        ('y_pred_threshold', y_pred_threshold, [float, type(None)]),
        ('normalize', normalize, [bool]),
        ('labels', labels, [list, type(None)]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('dpi', dpi, [int]),
        ('cmap', cmap, [str]),
        ('alpha', alpha, [float]),
        ('cm_font_size', cm_font_size, [int]),
        ('xaxis_label', xaxis_label, [str, type(None)]),
        ('yaxis_label', yaxis_label, [str, type(None)]),
        ('axis_label_size', axis_label_size, [int]),
        ('axis_label_pad', axis_label_pad, [int]),
        ('axis_tick_size', axis_tick_size, [int]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # avoid inplace modifications when resetting index and applying threshold
    df = df.copy().reset_index()

    if y_pred not in df.columns:
        raise TypeError('Missing "%s" column in dataframe. Available columns are: %r' % (y_pred, list(df.columns)))

    if y_true not in df.columns:
        raise TypeError('Missing "%s" column in dataframe. Available columns are: %r' % (y_true, list(df.columns)))

    # select default parameters
    if xaxis_label is None:
        xaxis_label = 'Predicted label'

    if yaxis_label is None:
        yaxis_label = 'True label'

    font_layout = {
        'family': 'sans-serif',
        'weight': 'normal',
        'size': cm_font_size}

    # binarize input predictions
    if y_pred_threshold is not None:
        df[y_pred] = (df[y_pred] > y_pred_threshold).astype(int)

    # calculate confusion matrices
    cms = []
    if average_folds is not None:
        # calculate a confusion matrix per fold
        if average_folds not in df.columns:
            raise TypeError(
                'Missing "%s" column in dataframe. Available columns are: %r' % (average_folds, list(df.columns)))

        for _, sub_df in df.groupby(average_folds):
            cms.append(confusion_matrix(
                y_true=sub_df[y_true].values,
                y_pred=sub_df[y_pred].values))
    else:
        # calculate a global confusion matrix
        cms.append(confusion_matrix(
            y_true=df[y_true].values,
            y_pred=df[y_pred].values))

    # stack confusion matrices
    cms = np.stack(cms)

    # normalize to percentage
    if normalize:
        cms = (cms / cms.sum(axis=1)[:, np.newaxis] * 100)

    # format confusion matrix representation
    if average_folds is None:
        assert cms.shape[0] == 1, 'Internal error (0)'
        cms = cms[0]
        avg_cms = cms
        cms_repr = np.empty(shape=cms.shape, dtype=object)
        for i in range(cms.shape[0]):
            for j in range(cms.shape[1]):
                cms_repr[i, j] = ('%.2f' % cms[i, j]) if normalize else ('%d' % cms[i, j])
    else:
        avg_cms = cms.mean(axis=0)
        std_cms = cms.std(axis=0)
        cms_repr = np.empty(shape=cms.shape[1:], dtype=object)
        for i in range(cms_repr.shape[0]):
            for j in range(cms_repr.shape[1]):
                str_val = ('%.2f' % avg_cms[i, j]) + r'$\pm$' + ('%.2f' % std_cms[i, j])
                cms_repr[i, j] = str_val

    # default ticks labels
    if labels is None:
        labels = ['C%d' % (i + 1) for i in range(cms_repr.shape[0])]

    if len(labels) != cms_repr.shape[0]:
        raise TypeError(
            'Number of classes (%d), number of labels provided in the param "labels" (%d)' % (
                cms_repr.shape[0], len(labels)))

    # create figure layout
    with plt.style.context('bmh'):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        # represent the average value
        ax.matshow(avg_cms, cmap=cmap, alpha=alpha)

        # annotate confusion matrix
        for i in range(cms_repr.shape[0]):
            for j in range(cms_repr.shape[1]):
                ax.text(
                    x=j, y=i,
                    s=cms_repr[i, j],
                    va='center', ha='center',
                    fontdict=font_layout)

        # formal layout
        ax.grid(False)
        ax.set_title(title, size=title_size, pad=title_pad)
        ax.set_xlabel(xaxis_label, fontsize=axis_label_size, labelpad=axis_label_pad)
        ax.set_ylabel(yaxis_label, fontsize=axis_label_size, labelpad=axis_label_pad)

        # remove axis-ticks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.set_yticklabels([''] + labels, fontsize=axis_tick_size)
            ax.set_xticklabels([''] + labels, fontsize=axis_tick_size)

        # change the position of X-axis ticks to the bottom
        ax.xaxis.set_ticks_position('bottom')

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()


def barPlot(
        *dfs,
        x: str,
        y: str,
        labels: list = None,
        colors: list or str = None,
        ax: mpl.axes.Axes = None,
        figsize: tuple = (6, 3.5),
        style: str = 'ggplot',
        dpi: int = 100,
        err_capsize: float or int = 0.15,
        err_lw: float or int = 1.5,
        grid_alpha: float = 0.15,
        xlabel_size: int or float = 13,
        ylabel_size: int or float = 13,
        title: str = '',
        title_size: int = 15,
        title_pad: int = 15,
        legend_pos: str = 'upper right',
        legend_bbox_to_anchor: tuple = None,
        legend_size: int = 12,
        yvmin: float = None,
        yvmax: float = None,
        xvmin: float = None,
        xvmax: float = None,
        hide_xlabel: bool = False,
        hide_ylabel: bool = False,
        xaxis_tick_size: int or float = 12,
        yaxis_tick_size: int or float = 12,
        xaxis_rotation: float or int = 0.0,
        yaxis_rotation: float or int = 0.0,
        save: str = None,
        save_kw: dict = None,
        show: bool = True):
    """ Bar plot function

    Parameters
    ----------
    *dfs
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        Y-axis variable. Must be present in the input dataframes.

    labels : list, default=None
        Labels used for identifying the input dataframes.

    colors : list or str, default=None
        Colors used for identifying the dataframe information.

    ax : mpl.axes.Axes, default=None
        Axes used to represent the figure.

    figsize : tuple, default=(6, 3.5)
        Figure size.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

    dpi : int, default=100
        Figure dpi.

    err_capsize : float, default=0.15
        Error capsize.

    err_lw : float, default=1.5
        Error linewidth.

    grid_alpha : float, default=0.15
        Gird lines opacity.

    xlabel_size : int, default=13
        Size of the x-label.

    ylabel_size : int, default=13
        Size of the y-label.

    title : str, default=''
        Plot title.

    title_size : int, default=15
        Title font size.

    title_pad : int, default=15
        Title pad.

    legend_pos : str, default='upper right'
        Legend position.

    legend_bbox_to_anchor : tuple, default=None
        Used for modifying the legend position relative to the position defined in `legend_pos`.

    legend_size : int, default=12
        Legend size.

    yvmin : float, default=None
        Minimum value in the y axis.

    yvmax : float, default=None
        Maximum value in the y-axis.

    xvmin : float, default=None
        Minimum value in the x-axis.

    xvmax : float, default=None
        Maximum value in the x-axis.

    hide_xlabel : bool, default=False
        Parameter indicating whether to hide the x-axis label.

    hide_ylabel : bool, default=False
        Parameter indicating whether to hide the y-axis label.

    xaxis_tick_size : int, default=12
        Controls the x-axis tick size.

    yaxis_tick_size : int, default=12
        Controls the y-axis tick size.

    xaxis_rotation : float or int, default=0.0
        Y-axis tick rotation.

    yaxis_rotation : float or int, default=0.0
        Y-axis tick rotation.

    save : str, default=None
        Parameter indicating whether to save the generated plot. If None (default) the plot will not be
        saved.

    save_kw : dict, default=None
        Optional parameters for saving the plot. This parameter will not have effect if the
        save parameter was set as None.

    show : bool, default=True
        Parameter indicating whether to save the generated plot.


    Examples
    --------
    >>> from gojo import core
    >>> from gojo import plotting
    >>>
    >>> # i.e., compute model performance metrics
    >>> scores_1 = report1.getScores(
    >>>     core.getDefaultMetrics(
    >>>     binary_classification', bin_threshold=0.5))['test']
    >>>
    >>> scores_2 = report1.getScores(
    >>>     core.getDefaultMetrics(
    >>>     binary_classification', bin_threshold=0.5))['test']
    >>>
    >>> # adapt for barplot representation
    >>> scores_1 = scores_1.melt()
    >>> scores_2 = scores_2.melt()
    >>>
    >>>
    >>> plotting.barPlot(
    >>>     scores_1, scores_2,
    >>>     x='variable', y='value',
    >>>     labels=['Model 1', 'Model 2'],
    >>>     title='Cross-validation results'
    >>> )
    """
    checkMultiInputTypes(
        ('x', x, [str]),
        ('y', y, [str]),
        ('labels', labels, [list, type(None)]),
        ('colors', colors, [list, str, type(None)]),
        ('ax', ax, [mpl.axes.Axes, type(None)]),
        ('figsize', figsize, [tuple]),
        ('style', style, [str]),
        ('dpi', dpi, [int]),
        ('err_capsize', err_capsize, [float, int]),
        ('err_lw', err_lw, [float, int]),
        ('grid_alpha', grid_alpha, [float]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('title_pad', title_pad, [int, float]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('legend_bbox_to_anchor', legend_bbox_to_anchor, [tuple, type(None)]),
        ('yvmin', yvmin, [float, type(None)]),
        ('yvmax', yvmax, [float, type(None)]),
        ('xvmin', xvmin, [float, type(None)]),
        ('xvmax', xvmax, [float, type(None)]),
        ('hide_xlabel', hide_xlabel, [bool]),
        ('hide_ylabel', hide_ylabel, [bool]),
        ('xaxis_tick_size', ylabel_size, [int, float]),
        ('yaxis_tick_size', ylabel_size, [int, float]),
        ('xaxis_rotation', xaxis_rotation, [int, float]),
        ('yaxis_rotation', yaxis_rotation, [int, float]),
        ('save', save, [str, type(None)]),
        ('save_kw', save_kw, [dict, type(None)]),
        ('show', show, [bool]))

    # check input data types
    for i, df in enumerate(dfs):
        checkInputType('df (%d)' % i, df, [pd.DataFrame])
        if x not in df.columns:
            raise TypeError('Missing "x" variable "%s". Available variables are: %r' % (x, list(df.columns)))

        if y not in df.columns:
            raise TypeError('Missing "y" variable "%s". Available variables are: %r' % (y, list(df.columns)))

    if labels is None:
        labels = ['(%d)' % (i + 1) for i in range(len(dfs))]

    # check legend_labels consistency
    if len(labels) != len(dfs):
        raise TypeError(
            'Missing labels in "labels". Number of labels provided ' \
            '(%d) not match the number of input DataFrames (%d)' % (len(labels), len(dfs)))

    if not (colors is None or isinstance(colors, str) or len(dfs) == len(colors)):
        raise TypeError(
            'Mismatch shape between input dataframes (%d) and "colors" (%d)' % (len(dfs), len(colors)))

    # merge dataframes information
    dfs_ = []
    for label, df in zip(labels, dfs):
        df = df.copy()
        df['_label'] = label
        dfs_.append(df)

    merge_df = pd.concat(dfs_, axis=0)

    # plot information
    with plt.style.context(style):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            fig.set_dpi(dpi)

        sns.barplot(
            data=merge_df, x=x, y=y, hue='_label',
            err_kws={'linewidth': err_lw},
            capsize=err_capsize,
            ax=ax,
            palette=colors)

        # set axis limits
        ax.set_ylim(bottom=yvmin, top=yvmax)
        ax.set_xlim(left=xvmin, right=xvmax)

        # figure layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=grid_alpha)
        ax.legend(
            loc=legend_pos,
            bbox_to_anchor=legend_bbox_to_anchor,
            prop=dict(size=legend_size))
        ax.set_xlabel('' if hide_xlabel else x, size=xlabel_size)
        ax.set_ylabel('' if hide_ylabel else y, size=ylabel_size)
        ax.set_title(title, size=title_size)
        plt.xticks(fontsize=xaxis_tick_size, rotation=xaxis_rotation)
        plt.yticks(fontsize=yaxis_tick_size, rotation=yaxis_rotation)
        ax.set_title(title, size=title_size, pad=title_pad)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()

