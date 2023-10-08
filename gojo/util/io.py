# Module with functions related with input/output operations.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import os
import json
import joblib
import pickle
import gzip
from datetime import datetime

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    fileExists,
    pathExists
)

# TODO. Implement a custom memory-efficient backend
# available backends used for saving and loading Python objects
_AVAILABLE_SERIALIZATION_BACKENDS = ['joblib', 'pickle', 'joblib_gzip', 'pickle_gzip']
_DEFAULT_BACKEND = 'joblib_gzip'


def saveJson(data: dict, file: str):
    """ Saves the input dictionary into a json file. IMPORTANT NOTE: numpy types must be previously converted to
    Python types. """
    checkMultiInputTypes(
        ('data', data, [dict]),
        ('file', file, [str]))
    fileExists(file, False)   # avoid overwrite existing files

    with open(file, 'w') as f:
        json.dump(data, f)


def loadJson(file: str) -> dict:
    """ Load a json file. """
    checkInputType('file', file, [str])
    fileExists(file, True)   # the file must previously exist

    with open(file) as f:
        content = json.load(f)

    return content


def serialize(obj, path: str, time_prefix: bool = False, overwrite: bool = False,
              backend: str = _DEFAULT_BACKEND) -> str:
    """ Function used to serialize Python objects.

    Parameters
    ----------
    obj : object
        Object to be saved.

    path : str
        File used to save the provided object.

    time_prefix : bool, default=False
        Parameter indicating whether to add a time prefix to the exported file (YYYYMMDD-HHMMSS).

    overwrite : bool, default=False
        Parameter indicating whether to overwrite a possible existing file.

    backend : str, default='joblib'
        Backend used for serialize the object.

    Returns
    -------
    path : str
        Serialized object.
    """
    checkMultiInputTypes(
        ('path', path, [str]),
        ('time_prefix', time_prefix, [bool]),
        ('overwrite', overwrite, [bool]))

    # separate path and file name
    path_to_obj, filename = os.path.split(path)
    path_to_obj = os.path.abspath(path_to_obj)

    # add time prefix
    if time_prefix:
        filename = '%s-%s' % (datetime.now().strftime('%y%m%d-%H%M%S'), filename)

    # create the path to the output file
    file_fp = os.path.join(path_to_obj, filename)

    # the input path must previously exist
    pathExists(path_to_obj, must_exists=True)

    if not overwrite:
        fileExists(file_fp, must_exists=False)

    # export the object
    return _serialize(obj, file_fp, backend)


def _serialize(obj, path: str, backend: str) -> str:
    """ Subroutine used to serialize objects. """
    # check input types
    checkMultiInputTypes(
        ('backend', backend, [str]),
        ('path', path, [str]))

    # check backends
    if backend not in _AVAILABLE_SERIALIZATION_BACKENDS:
        raise TypeError('Unrecognized backend "%s". Available backends are: %r' % (
            backend, _AVAILABLE_SERIALIZATION_BACKENDS))

    if backend == 'joblib':
        out = _serializeJoblib(obj, path)
    elif backend == 'joblib_gzip':
        out = _gzip(_serializeJoblib(obj, path))
    elif backend == 'pickle':
        out = _serializePickle(obj, path)
    elif backend == 'pickle_gzip':
        out = _gzip(_serializePickle(obj, path))
    else:
        assert False, 'Unhandled case'

    return out


def _serializeJoblib(obj, path) -> str:
    """ Joblib serialization backend. """
    joblib.dump(obj, path)

    return path


def _serializePickle(obj, path) -> str:
    """ Pickle serialization backend. """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

    return path


def _gzip(in_path: str) -> str:
    """ Apply a gzip compression. """
    with open(in_path, 'rb') as f:
        with gzip.open(in_path + '.gz', 'wb') as fgz:
            fgz.writelines(f)

    os.remove(in_path)   # remove uncompressed file

    return in_path + '.gz'


def load(file: str, backend: str = _DEFAULT_BACKEND) -> object:
    """ Function used to load serialized Python objects (see 'gojo.util.io.serialize').

    Parameters
    ----------
    file : str
        Object to be loaded.

    backend : str, default='joblib'
        Backend used for serialize the object.

    Returns
    -------
    obj : object
        Loaded object.
    """
    checkMultiInputTypes(
        ('file', file, [str]),
        ('backend', backend, [str]))

    # check backends
    if backend not in _AVAILABLE_SERIALIZATION_BACKENDS:
        raise TypeError('Unrecognized backend "%s". Available backends are: %r' % (
            backend, _AVAILABLE_SERIALIZATION_BACKENDS))

    # check that the input file exists
    fileExists(file, must_exists=True)

    # load the object
    if backend == 'joblib':
        obj = _loadJoblib(file)
    elif backend == 'joblib_gzip':
        obj = _loadJoblibGzip(file)
    elif backend == 'pickle':
        obj = _loadPickle(file)
    elif backend == 'pickle_gzip':
        obj = _loadPickleGzip(file)
    else:
        assert False, 'Unhandled case'

    return obj


def _loadJoblib(path: str) -> object:
    """ Load a joblib serialized object. """
    return joblib.load(path)


def _loadJoblibGzip(path: str) -> object:
    """ Load a joblib + gzip serialized object. """
    with gzip.open(path, 'rb') as fgz:
        obj = joblib.load(fgz)

    return obj


def _loadPickle(path: str) -> object:
    """ Load a pickle serialized object. """
    with open(path, 'rb') as f:
        obj = pickle.load(f)

    return obj


def _loadPickleGzip(path: str) -> object:
    """ Load a joblib + gzip serialized object. """
    with gzip.open(path, 'rb') as fgz:
        obj = pickle.load(fgz)

    return obj


def pprint(*args, verbose: bool = True):
    if verbose:
        print(*args)


def createObjectRepresentation(class_name: str, **parameters) -> str:
    """ Function used to create object representation for the __repr__() method. """
    checkInputType('class_name', class_name, [str])

    representation = '{}('.format(class_name)
    if not isinstance(parameters, dict):
        representation += ')'
    else:
        for k, v in parameters.items():
            representation += '\n    {}={},'.format(k, v)
        representation = representation.rstrip(',')
        representation += '\n)'

    return representation


