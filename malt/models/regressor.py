# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class Regressor(torch.nn.Module, abc):
    """ Base class for a regressor.

    """
    def __init__(self, *args, **kwargs):
        super(Regressor, self).__init__()

    @abc.abstractmethod
    def condition(self, *args, **kwargs):
        raise NotImplementedError

    @
