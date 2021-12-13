# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class Likelihood(torch.nn.Module, abc.ABC):
    """ Base class for likelihood. """

    def __init__(self, in_features: int) -> None:
        super(Likelihood, self).__init__()
        self.in_features = in_features

    @abc.abstractmethod
    def condition(self, *args, **kwargs) -> torch.distributions.Distribution:
        """ Make predictive posterior distribution from parameters. """
        raise NotImplementedError

    @abc.abstractmethod
    def loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class SimpleLikelihood(Likelihood):
    """ Likelihood with only parameter input. """

    def __init__(self, in_features: int) -> None:
        super(SimpleLikelihood, self).__init__(in_features=in_features)

    def loss(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        distribution = self.condition(theta)
        return -distribution.log_prob(y).mean()


# =============================================================================
# MODULE CLASSES
# =============================================================================
class HomoschedasticGaussianLikelihood(SimpleLikelihood):
    """ A Gaussian likelihood with homoschedastic noise model. """

    def __init__(self, log_sigma=0.0) -> None:
        super(HomoschedasticGaussianLikelihood, self).__init__(
            in_features=1,
        )
        self.register_buffer("log_sigma", torch.tensor(log_sigma))

    def condition(
        self,
        theta: torch.Tensor,
    ) -> torch.distributions.Distribution:
        # decompose theta
        assert theta.dim() == 2
        assert theta.shape[1] == 1
        mu = theta
        return torch.distributions.Normal(
            loc=mu,
            scale=self.log_sigma.exp(),
        )


class HeteroschedasticGaussianLikelihood(SimpleLikelihood):
    """ A Gaussian likelihood with homoschedastic noise model. """

    def __init__(self) -> None:
        super(HeteroschedasticGaussianLikelihood, self).__init__(
            in_features=2,
        )

    def condition(
        self, theta: torch.Tensor
    ) -> torch.distributions.Distribution:
        # decompose theta
        assert theta.dim() == 2
        assert theta.shape[1] == 2
        mu = theta[:, 0][:, None]
        log_sigma = theta[:, 1][:, None]
        return torch.distributions.Normal(
            loc=mu,
            scale=log_sigma.exp(),
        )
