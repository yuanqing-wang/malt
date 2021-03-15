# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
from . import utility_functions
from .data.dataset import Dataset

# =============================================================================
# BASE CLASSES
# =============================================================================
class Policy(torch.nn.Module, abc.ABC):
    """ Base class for policy. """
    def __init__(self):
        super(Policy, self).__init__()

    @abc.abstractmethod
    def forward(
            self, distribution: torch.distributions.Distribution
        ) -> torch.Tensor:
        """ Provide the indices to acquire.

        Parameters
        ----------
        points : Dataset
            A list of points to predict.

        Returns
        -------
        torch.Tensor
            Ranked indices.

        """
        raise NotImplementedError

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Greedy(Policy):
    """ Greedy policy. """
    def __init__(
            self,
            utility_function=utility_functions.expected_improvement,
            acquisition_size: int=1,
        ):
        super(Greedy, self).__init__()
        self.utility_function = utility_function
        self.acquisition_size = acquisition_size

    def forward(
            self, distribution: torch.distributions.Distribution
        ) -> torch.Tensor:
        score = self.utility_function(distribution)
        _, idxs = torch.topk(
            score, self.acquisition_size,
        )
        return idxs
