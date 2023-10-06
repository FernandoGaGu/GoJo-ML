# Module with tools related to data validation.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import os
import inspect
from collections.abc import Iterable


def checkInputType(input_var_name: str, input_var: object, valid_types: list or tuple):
    """ Function that checks that the type of the input variable "input_var" is within the valid types "valid_types"."""
    if not isinstance(input_var_name, str):
        raise TypeError(f'input_var_name must be a string. Provided {str(input_var_name)}.')
    if not isinstance(valid_types, (tuple, list)):
        raise TypeError(f'valid_types must be a list or a tuple. Provided {str(valid_types)}.')

    if not isinstance(input_var, tuple(valid_types)):
        raise TypeError(
            f'Input variable {input_var_name} of type {str(type(input_var))} not in available types: '
            f'{",".join([str(v) for v in valid_types])}.')


def checkMultiInputTypes(*args):
    """ Wrapper of function checkInputType to check multiple variables at the same time. """
    for element in args:
        if not ((isinstance(element, tuple) or isinstance(element, list)) and len(element) == 3):
            raise TypeError('The arguments of this function must consist of tuples or lists of three arguments '
                            'following the signature of the gojo.util.checkInputType() function.')

        checkInputType(*element)


def fileExists(file: str, must_exists: bool):
    """ Function that checks if a given file exists or not exists"""
    checkMultiInputTypes(
        ('file', file, [str]),
        ('must_exists', must_exists, [bool]))

    if must_exists:
        if not os.path.exists(file):
            raise FileNotFoundError('File "{}" not found.'.format(os.path.abspath(file)))
    else:
        if os.path.exists(file):
            raise FileExistsError('File "{}" already exists.'.format(os.path.abspath(file)))


def checkCallable(input_obj_name: str, obj: callable):
    """ Function used to check if a given object is callable. """
    if not callable(obj):
        raise NotImplementedError('"{}" is not callable.'.format(input_obj_name))


def checkIterable(input_obj_name: str, obj):
    """ Function used to check if a given object is an iterable. """
    if not isinstance(obj, Iterable):
        raise NotImplementedError('"{}" is not an iterable.'.format(input_obj_name))


def checkClass(input_obj_name: str, obj):
    """ Function used to check if a given object is a class. """
    if not inspect.isclass(obj):
        raise TypeError('"{}" is not a class.'.format(input_obj_name))

