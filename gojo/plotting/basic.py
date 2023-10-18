# Module with plotting functions.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, functional, and documented.
#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)


def linePlot(
        *dfs, x: str, y: str, err: str = None, err_alpha: float = 0.3, legend_labels: list = None,
        legend_pos: str = 'upper right', legend_size: int or float = 12, xlabel_size: float or int = 13,
        ylabel_size: float or int = 13, title: str = '', title_size: int or float = 15,
        figsize: tuple = (6, 3.5), colors: list = None, grid_alpha: float = 0.5, lw: float or int or list = None,
        ls: str or list = None, dpi: int = 100, style: str = 'ggplot', save: str = None, save_kw: dict = None,
        show: bool = True):
    """ Line plot function.

    Parameters
    ----------
    *dfs : pd.DataFrame
        Input dataframes with the data to be represented.

    x : str
        X-axis variable. Must be present in the input dataframes.

    y : str
        y-axis variable. Must be present in the input dataframes.

    err : str
        Variable indicating the errors associated with the lines. Must be present in the input
        dataframes.

    err_alpha : float, default=0.3
        Opacity used to plot the errors.

    legend_labels : list, default=None
        Labels used for identifying the input dataframes.

    legend_pos : str, default='upper right'
        Legend position.

    legend_size : int, default=12
        Legend size.

    xlabel_size : float or int, default=13
        X-axis label size.

    ylabel_size : float ot int, default=13
        Y-axis label size.

    title : str, default=''
        Plot title.

    title_size : int or float, default=0.5
        Title font size.

    figsize : tuple, default=(6, 3.5)
        Figure size.

    colors : list, default=None
        Colors used for identifying the dataframe information.

    grid_alpha : float, default=0.5
        Grid opacity.

    lw : float or int or list, default=None
        Line(s) width(s).

    ls : str or list, default=None
        Line(s) styles(s).

    dpi : int, default=100
        Figure dpi.

    style : str, default='ggplot'
        Plot styling. (see 'matplotlib.pyplot.styles')

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
    >>>     legend_labels=['Train', 'Validation'],
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
        ('legend_labels', legend_labels, [list, type(None)]),
        ('legend_pos', legend_pos, [str]),
        ('legend_size', legend_size, [int, float]),
        ('xlabel_size', xlabel_size, [int, float]),
        ('ylabel_size', ylabel_size, [int, float]),
        ('title', title, [str]),
        ('title_size', title_size, [int, float]),
        ('figsize', figsize, [tuple]),
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

    if legend_labels is None:
        legend_labels = ['(%d)' % (i + 1) for i in range(len(dfs))]

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

    if len(dfs) != len(legend_labels):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "legend_labels" (%d)' % (len(dfs), len(legend_labels)))

    if not (colors is None or len(dfs) == len(colors)):
        raise TypeError(
            'Missmatch shape between input dataframes (%d) and "colors" (%d)' % (len(dfs), len(colors)))

    # plot information
    with plt.style.context(style):
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(dpi)

        for i, (label, df) in enumerate(zip(legend_labels, dfs)):
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
        ax.set_title(title, size=title_size)

        # save figure if specified
        if save:
            save_kw = {} if save_kw is None else save_kw
            plt.savefig(save, **save_kw)

        if show:
            plt.show()

