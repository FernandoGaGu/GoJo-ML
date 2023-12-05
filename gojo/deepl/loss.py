# Module containing the definitions of some loss functions typically used in Deep Learning models.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: under development
#
import torch
from typing import Tuple


class WBCELoss(torch.nn.Module):
    """ Weighted binary cross-entropy.

    Parameters
    ----------
    weight : float, default = 1.0
        Weight applied to the positive class.

    """
    def __init__(self, weight: float or int = 1.0):
        super(WBCELoss, self).__init__()

        self.weight = weight

    def forward(self, y_hat: torch.Tensor, y_true: torch.Tensor):
        clf_loss = torch.nn.functional.binary_cross_entropy(
            input=y_hat, target=y_true, reduction='none')

        clf_loss = clf_loss + clf_loss * y_true * self.weight

        return torch.mean(clf_loss)


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




