# =============================================================================
# IMPORTS
# =============================================================================
import torch
from typing import Union

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def random(distribution: torch.distributions.Distribution):
    return torch.randn(distribution.batch_shape)


def uncertainty(distribution: torch.distributions.Distribution):
    return distribution.variance


def expected_improvement(
    distribution: torch.distributions.Distribution,
    y_best: Union[torch.Tensor, float] = 0.0,
    n_samples: int = 64,
):
    improvement = torch.nn.functional.relu(
        distribution.sample(torch.Size([n_samples])) - y_best
    )

    return improvement.mean(axis=0)


def probability_of_improvement(
    distribution: torch.distributions.Distribution,
    y_best: Union[torch.Tensor, float] = 0.0,
):
    return 1.0 - distribution.cdf(y_best)


def upper_confidence_boundary(
    distribution: torch.distributions.Distribution,
    percentage: Union[torch.Tensor, float] = 0.95,
):
    percentage = torch.tensor(percentage)
    return distribution.icdf(1 - (1 - percentage) / 2)
