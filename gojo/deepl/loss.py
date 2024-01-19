# Module containing the definitions of some loss functions typically used in Deep Learning models.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: under development
#
import torch
from typing import Tuple
from copy import deepcopy


def weightedBCEwithNaNs(y_hat: torch.Tensor, y_true: torch.Tensor, weight: float) -> torch.Tensor:
    """ Similar to :func:`gojo.deepl.loss.weightedBCE` but allowing the incorporation of missing values in `y_true`.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    weight : float
        Weight applied to the positive class.
    """
    na_mask = torch.isnan(y_true)  # create the nan mask
    y_true_ = deepcopy(y_true)  # avoid inplace modifications of the input values
    y_true_[na_mask] = 0.0  # fill missing values
    y_true_bin_mask = (1 - na_mask.int())  # create a binary mask for masking the unreduced loss (NaNs represented as 0)

    # calculate the unreduced loss
    unreduced_loss = torch.nn.functional.binary_cross_entropy(
        input=y_hat,
        target=y_true_,
        reduction='none'
    )

    # ignore entries with missing values
    unreduced_loss_masked = unreduced_loss * y_true_bin_mask

    # apply weights to the positive class
    w_unreduced_loss_masked = unreduced_loss_masked * (y_true_ * weight + (1 - y_true_))

    # reduce the weighted loss
    w_loss_masked = (w_unreduced_loss_masked.sum(dim=0) / (
                y_true_bin_mask.sum(dim=0) + 1e-06)).mean()  # add constant to avoid zero division

    return w_loss_masked


def weightedBCE(y_hat: torch.Tensor, y_true: torch.Tensor, weight: float) -> torch.Tensor:
    """ Calculate the binary cross-entropy by weighting the positive class.

    Parameters
    ----------
    y_hat : torch.Tensor
        Model predictions.

    y_true : torch.Tensor
        Ground true values.

    weight : float
        Weight applied to the positive class.

    """

    # calculate the unreduced loss
    unreduced_loss = torch.nn.functional.binary_cross_entropy(
        input=y_hat, target=y_true, reduction='none')

    # apply weights to the positive class
    w_unreduced_loss = unreduced_loss * (y_true * weight + (1 - y_true))

    # reduce the weighted loss
    w_loss = torch.mean(w_unreduced_loss)

    return w_loss


class BCELoss(torch.nn.Module):
    """ Weighted binary cross-entropy.

    Parameters
    ----------
    weight : float, default = 1.0
        Weight applied to the positive class.

    allow_nans : bool, default = False
        Boolean parameter indicating whether the true values contain missing values. If the value is indicated as
        `False` this class will use :func:`gojo.deepl.loss.weightedBCE` as internal function, if the value is indicated
        as `True`, the class will use :func:`gojo.deepl.loss.weightedBCEwithNaNs` as internal function.
    """
    def __init__(self, weight: float or int = 1.0, allow_nans: bool = False):
        super(BCELoss, self).__init__()

        self.weight = weight
        self.allow_nans = allow_nans

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.allow_nans:
            w_loss = weightedBCEwithNaNs(y_hat=y_hat, y_true=y_true, weight=self.weight)
        else:
            w_loss = weightedBCE(y_hat=y_hat, y_true=y_true, weight=self.weight)

        return w_loss


class ELBO(torch.nn.Module):
    """ Evidence lower bound (ELBO) loss function as described in "Auto-Encoding Variational Bayes" from Kigma and
    Welling (2014).

    Parameters
    ----------
    kld_weight : float, default=1.0
        Weight applied to the Kullback-Leibler divergence term.

    """
    def __init__(self, kld_weight: float = 1.0):
        super(ELBO, self).__init__()

        self.kld_weight = kld_weight

    def forward(
            self,
            x_hat: torch.Tensor,
            x_true: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """ Forward pass.

        Parameters
        ----------
        x_hat : torch.Tensor
            Reconstructed model input.

        x_true : torch.Tensor
            True model input.

        mu : torch.Tensor
            Mean projection vector.

        logvar : torch.Tensor
            Log-var projection vector.

        Returns
        -------
        output : Tuple[torch.Tensor, dict]
            This function will return a two element tuple where the first element will correspond to the loss while
            the second element will be a dictionary containing other loss function related parameters.

        """
        rec_loss = torch.nn.functional.mse_loss(x_hat, x_true)

        # compute the kullback leibler divergende (https://statproofbook.github.io/P/norm-kl.html)
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        return rec_loss + self.kld_weight * kld, {
            'reconstruction_loss': rec_loss.detach().cpu().item(),
            'KLD': kld.detach().cpu().item()}




