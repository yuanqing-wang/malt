# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
from .regressor import Regressor
from .representation import Representation
from .likelihood import Likelihood, SimpleLikelihood

# =============================================================================
# BASE CLASSES
# =============================================================================
class SupervisedModel(torch.nn.Module, abc.ABC):
    """ A supervised model. """
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
        distribution = self.likelihood(theta)

    def loss(self, g, y):
        # get predictive posterior distribution
        distribution = self.condition(g)

        return -distribution.log_prob(y).mean()
