import json

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType,
    fileExists
)


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
            representation += '\n\t{}={},'.format(k, v)
        representation = representation.rstrip(',')
        representation += '\n)'

    return representation


