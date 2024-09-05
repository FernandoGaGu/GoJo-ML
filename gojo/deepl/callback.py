# Module containing the callbacks used by the training loops of deep learning models.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: completed, and testing to be done, and documented.
#
import os
import numpy as np
import pandas as pd
import warnings
import torch
from abc import ABCMeta, abstractmethod
from pathlib import Path

from ..util.validation import (
    checkMultiInputTypes,
    checkInputType
)
from ..util.io import saveTorchModel


class Callback(object):
    """ Base class (interface) used to define the callbacks to be executed in each iteration of the training
    loop of the neural networks defined in :func:`gojo.deepl.loop.fitNeuralNetwork`.
    These callbacks provide directives to modify the training of the models. A classic example would be the
    early stopping callback (defined in :class:`gojo.deepl.callback.EarlyStopping`).

    Subclasses must define the following methods:

        - evaluate()
            This method will make available to the callback the following arguments used (and updated) in the current
            iteration of the :func:'gojo.deepl.loop.fitNeuralNetwork' training loop:

                model : :class:`gojo.core.base.TorchSKInterface` or :class:`gojo.core.base.ParametrizedTorchSKInterface`
                    Model to be trained.
                train_metrics : list
                    Train computed metrics until the last epoch.
                valid_metrics : list
                    Validation computed metrics until the last epoch.
                train_loss : list
                    Train computed loss until the last epoch.
                valid_loss : list
                    Validation computed loss until the last epoch.

            This method has to return a directive (as a string) that will be interpreted by the
            :func:`gojo.deepl.loop.fitNeuralNetwork` inner loop.

        - resetState()
            This method should reset the inner state of the callback.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name: str):
        checkMultiInputTypes(
            ('name', name, [str]))

        self._name = name

    def __repr__(self):
        return self._name

    def __str__(self):
        return self.__repr__()

    def __call__(self, *args, **kwargs) -> str:
        command = self.evaluate(*args, **kwargs)

        checkInputType('gojo.deepl.callback.Callback.__call__()', command, [str, type(None)])

        return command

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> str:
        raise NotImplementedError

    @abstractmethod
    def resetState(self):
        raise NotImplementedError


class EarlyStopping(Callback):
    """ Callback used to perform an early stopping of the :func:`gojo.deepl.loop.fitNeuralNetwork` training loop.

    Parameters
    ----------
    it_without_improve : int
        Number of iterations that must be completed without the model showing a decrease in the loss value
        over the validation set (average of the last epochs or count of the last epochs, as defined by
        parameter `track`) to perform an early stopping ending the loop execution.

    track : str, default='mean'
        Method used to compare the latest value of the loss on the validation set with respect to the
        historical value. Methods currently available:

            - 'mean': compare the current value with respect to the average of the `it_without_improve` epochs.
            - 'count': compare the current value with respect to `it_without_improve` epochs.

    """

    VALID_TRACKING_OPTS = ['mean', 'count']
    _LOSS_IDENTIFICATION_KEY = 'loss (mean)'   # HACK. Hard-coding, key used to identify the average loss values
    DIRECTIVE = 'stop'

    def __init__(self, it_without_improve: int, track: str = 'mean'):
        super().__init__(name='EarlyStopping')

        assert track in EarlyStopping.VALID_TRACKING_OPTS

        self.it_without_improve = it_without_improve
        self.track = track

        self._saved_valid_loss = []

    def _getLastLossValue(self, stats: list) -> float:
        """ Function used to get and check the current loss values. """
        curr_loss = stats[-1].get(self._LOSS_IDENTIFICATION_KEY, np.nan)

        # check for NaNs in the current loss
        if pd.isna(curr_loss):
            warnings.warn('Current average loss value is NaN.')

        return curr_loss

    def evaluate(self, valid_loss: list, **_) -> str:
        """ Early stopping inner logic. """
        # check input type
        checkInputType('gojo.deepl.callback.EarlyStopping.evaluate(valid_loss)', valid_loss, [list])

        command = None

        if len(self._saved_valid_loss) < self.it_without_improve:
            # not enough iterations performed
            self._saved_valid_loss.append(self._getLastLossValue(valid_loss))

        else:
            # there is enough iterations performed to check loss improvements
            curr_loss = self._getLastLossValue(valid_loss)
            if self.track == 'count':
                if np.all(curr_loss > np.array(self._saved_valid_loss)[-1 * self.it_without_improve:]):
                    command = self.DIRECTIVE
            elif self.track == 'mean':
                if curr_loss > np.mean(self._saved_valid_loss[-1 * self.it_without_improve:]):
                    command = self.DIRECTIVE
            else:
                raise NotImplementedError()

            self._saved_valid_loss.append(curr_loss)

        return command

    def resetState(self):
        """ Reset callback """
        self._saved_valid_loss = []


class SaveCheckPoint(Callback):
    """ Callback used to save the model parameters during training.

    Parameters
    ----------
    output_dir : str
        Output directory used to store model parameters. If it does not exist, it will
        be created automatically.

    key : str
        Key used to identify the model.

    each_epoch : int
        Specify the number of epochs to save for each model.

    verbose : bool, default=True
        Parameter that indicates whether to display messages on the screen when
        executing the early stop.
    """
    DIRECTIVE = None

    def __init__(
        self,
        output_dir: str,
        key: str,
        each_epoch: int,
        verbose: bool = True
    ):
        super().__init__(name='SaveCheckPoint')

        self.output_dir = output_dir
        self.key = key
        self.each_epoch = each_epoch
        self.verbose = verbose

    def evaluate(self, n_epoch: int, model: torch.nn.Module, **_):

        if n_epoch > 0:
            # create the output directory if it does not exist
            if not os.path.exists(self.output_dir):
                Path(self.output_dir).mkdir(parents=True)

            # save model
            if n_epoch % self.each_epoch == 0:
                out_file = saveTorchModel(
                    base_path=self.output_dir,
                    key='%s_checkpoint_%d' % (self.key, int(n_epoch)),
                    model=model
                )

                if self.verbose:
                    self.message(out_file)

        return self.DIRECTIVE

    @staticmethod
    def message(out_file: str):
        print('\nSaved model %s\n' % out_file)

    def resetState(self):
        """ Reset callback """
        pass
