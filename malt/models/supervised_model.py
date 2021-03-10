# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class SupervisedModel(torch.nn.Module, abc.ABC):
    """ A supervised model. """
    def __init__(
            self,
            representation: torch.nn.Module,
            regressor: torch.nn.Module,
        ) -> None:
        super(SupervisedModel, self).__init__()
        self.representation = representation
        self.regressor = regressor

    @abc.abstractmethod
    def condition(self, *args, **kwargs):
        """ Make predictive posterior. """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, *args, **kwargs):
        """ Compute loss. """
        raise NotImplementedError
