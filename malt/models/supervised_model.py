# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
import gpytorch
from typing import Any
from .regressor import Regressor
from .representation import Representation
from .likelihood import (
    Likelihood,
    SimpleLikelihood,
    HeteroschedasticGaussianLikelihood,
)


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
        likelihood: Likelihood,
    ) -> None:
        super(SupervisedModel, self).__init__()

        assert representation.out_features == regressor.in_features

        self.representation = representation
        self.regressor = regressor
        self.likelihood = likelihood

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """ Make predictive posterior. """
        raise NotImplementedError

    def loss(self, x, y):
        """Default loss function. """
        predictive_distribution = self.forward(x)
        return -predictive_distribution.log_prob(y[..., None]).mean()


class SimpleSupervisedModel(SupervisedModel):
    """ A supervised model that only takes graph. """

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
        likelihood: SimpleLikelihood,
    ):
        assert regressor.out_features == likelihood.in_features

        super(SimpleSupervisedModel, self).__init__(
            representation=representation,
            regressor=regressor,
            likelihood=likelihood,
        )


    def forward(self, g):
        # graph -> latent representation
        h = self.representation(g)

        # latent_representation -> parameters
        theta = self.regressor(h)

        # parameters -> distribution
        distribution = self.likelihood.condition(theta)

        return distribution

    condition = forward


class GaussianProcessSupervisedModel(SupervisedModel, gpytorch.models.GP):
    """ A supervised model that only takes graph. """

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
        likelihood: Any = HeteroschedasticGaussianLikelihood(),
    ):

        assert representation.out_features == regressor.in_features

        super(GaussianProcessSupervisedModel, self).__init__(
            representation=representation,
            regressor=regressor,
            likelihood=likelihood,
        )

    def forward(self, g):
        # graph -> latent representation
        h = self.representation(g)

        # latent representation -> distribution
        y_pred = self.regressor(h)

        return y_pred

    condition = forward

    def loss(self, x, y):
        predictive_distribution = self.forward(x)
        return -gpytorch.mlls.ExactMarginalLogLikelihood(
            self.regressor.likelihood, self,
        )(
            predictive_distribution, y
        )
