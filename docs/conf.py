# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'gojo - Documentation'
copyright = '2024, Fernando García Gutiérrez'
author = 'Fernando García Gutiérrez'
release = '0.1.5.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.intersphinx',  # added this but does not help
]
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'optuna': ('https://optuna.readthedocs.io/en/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}

autodoc_mock_imports = [
    'numpy',
    'sklearn',
    'scipy',
    'joblib',
    'loguru',
    'pandas',
    'optuna',
    'tqdm',
    'torch',
    'torch_geometric',
    'typing',
    'matplotlib',
    'seaborn',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'style_nav_header_background': '#F24C4C',
    'display_version': True,
    'prev_next_buttons_location': 'both',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
