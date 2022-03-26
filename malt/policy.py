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
        """Provide the indices to acquire.

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
        acquisition_size: int = 1,
    ):
        super(Greedy, self).__init__()
        self.utility_function = utility_function
        self.acquisition_size = acquisition_size

    def forward(
        self, distribution: torch.distributions.Distribution
    ) -> torch.Tensor:
        score = self.utility_function(distribution)
        _, idxs = torch.topk(
            score,
            self.acquisition_size,
            dim=0,
        )
        return idxs


class Random(Policy):
    """ Greedy policy. """

    def __init__(
        self,
        acquisition_size: int = 1,
    ):
        super(Random, self).__init__()
        self.acquisition_size = acquisition_size

    def forward(
        self, distribution: torch.distributions.Distribution
    ) -> torch.Tensor:
        
        idxs = torch.randint(
            high = distribution.batch_shape[0],
            size = (self.acquisition_size,)
        )

        return idxs



class ThompsonSampling(Policy):
    """ Thompson sampling policy. """

    def __init__(
        self,
        acquisition_size: int = 1,
    ):
        super(ThompsonSampling, self).__init__()
        self.acquisition_size = acquisition_size

    def forward(
        self, distribution: torch.distributions.Distribution
    ) -> torch.Tensor:

        # sample f_X
        thetas = distribution.sample(
            (self.acquisition_size, )
        )

        # find unique argmax f_X
        idxs_dups = torch.argmax(thetas, axis=1).ravel()
        idxs_ts = torch.unique(idxs_dups)
        num_idxs = len(idxs_ts)

        # if we didn't fill the round, select rest randomly
        if num_idxs < self.acquisition_size:

            # find unselected indices; mask indices
            mask = torch.zeros(len(data)).bool()
            mask[idxs_ts] = 1
            range_masked = torch.arange(len(data))[~mask]

            # shuffle unselected indices and select
            idx = torch.randperm(range_masked.nelement())
            range_masked = range_masked.view(-1)[idx].view(range_masked.size())
            idx_rand = range_masked[:(self.acquisition_size - num_idxs)]
            
            # append random indices
            idxs = torch.cat([idxs_ts, idx_rand])

        else:
            idxs = idxs_ts

        return idxs