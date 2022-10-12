# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
import gpytorch
from typing import Any
from .regressor import Regressor
from .representation import Representation

# =============================================================================
# BASE CLASSES
# =============================================================================
class SupervisedModel(torch.nn.Module, abc.ABC):
    """A supervised model.

    Parameters
    ----------
    representation : Representation
        Module to project small molecule graph to latent embeddings.

    regressor : Regressor
        Module to convert latent embeddings to likelihood parameters.

    likelihood : Likelihood
        Module to convert likelihood parameters and data to probabilities.

    Methods
    -------
    condition

    loss

    """

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
    ) -> None:
        super(SupervisedModel, self).__init__()

        assert representation.out_features == regressor.in_features
        self.representation = representation
        self.regressor = regressor

    def forward(self, x):
        """ Make predictive posterior. """
        representation = self.representation(x)
        print(representation.shape)
        posterior = self.regressor(representation)
        return posterior

    def loss(self, x, y):
        """Default loss function. """
        representation = self.representation(x)
        loss = self.regressor.loss(representation, y)
        return loss
