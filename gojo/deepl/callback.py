# Module containing the callbacks used by the training loops of deep learning models.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: still under development
#
from abc import ABCMeta, abstractmethod

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)

class Callback(object):
    """ Description """
    __metaclass__ = ABCMeta

    def __init__(self, name: str, verbose: bool):
        checkMultiInputTypes(
            ('name', name, [str]),
            ('verbose', verbose, [bool]))

        self.verbose = verbose
        self._name = name

    def __repr__(self):
        return self._name

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs) -> str:
        command = self.evaluate(*args, **kwargs)

        checkInputType('gojo.deepl.callback.Callbak.__call__()', command, [str, type(None)])

        return command

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def resetState(self):
        raise NotImplementedError

