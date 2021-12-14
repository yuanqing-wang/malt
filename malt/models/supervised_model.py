# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
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
        assert regressor.out_features == likelihood.in_features

        self.representation = representation
        self.regressor = regressor
        self.likelihood = likelihood

    @abc.abstractmethod
    def condition(self, *args, **kwargs):
        """ Make predictive posterior. """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        """ Compute loss. """
        raise NotImplementedError


class SimpleSupervisedModel(SupervisedModel):
    """ A supervised model that only takes graph. """

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
        likelihood: SimpleLikelihood,
    ):
        super(SimpleSupervisedModel, self).__init__(
            representation=representation,
            regressor=regressor,
            likelihood=likelihood,
        )

    def condition(self, g):
        # graph -> latent representation
        h = self.representation(g)

        # latent_representation -> parameters
        theta = self.regressor(h)

        # parameters -> distribution
        distribution = self.likelihood.condition(theta)

        return distribution

    def loss(self, g, y):
        # get predictive posterior distribution
        distribution = self.condition(g)

        return -distribution.log_prob(y).mean()


class GaussianProcessSupervisedModel(SupervisedModel):
    """ A supervised model that only takes graph. """

    x_tr = None
    y_tr = None

    def __init__(
        self,
        representation: Representation,
        regressor: Regressor,
        likelihood: SimpleLikelihood,
    ):
        assert isinstance(
            likelihood,
            HeteroschedasticGaussianLikelihood,
        )
        super(GaussianProcessSupervisedModel, self).__init__(
            representation=representation,
            regressor=regressor,
            likelihood=likelihood,
        )

    def _blind_condition(self, g):
        return torch.distributions.Normal(
            torch.zeros(g.batch_size, 1),
            torch.ones(g.batch_size, 1),
        )

    def condition(self, g):
        if self.x_tr is None or self.y_tr is None:
            return self._blind_condition(g)

        # graph -> latent representation
        h = self.representation(g)
        return self.regressor.condition(h, x_tr=self.x_tr, y_tr=self.y_tr)

    def loss(self, g, y):
        h = self.representation(g)
        self.register_buffer("x_tr", h)
        self.register_buffer("y_tr", y)
        return self.regressor.loss(h, y)
