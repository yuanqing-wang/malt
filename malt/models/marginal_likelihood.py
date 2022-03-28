import abc
import torch
from typing import Union
from .likelihood import Likelihood
from .regressor import Regressor
from .supervised_model import SupervisedModel

class MarginalLikelihood(torch.nn.Module, abc.ABC):
    """ Base class for marginal likelihood. """

    def __init__(
        self,
        likelihood: Likelihood,
        model: SupervisedModel,
    ) -> None:
        super(MarginalLikelihood, self).__init__()
        self.likelihood = likelihood
        self.model = model

    @abc.abstractmethod
    def forward(self, output, target, *args, **kwargs) -> torch.Tensor:
        """ Computes marginal likelihood given output distribution, target. """
        raise NotImplementedError


class SimpleMarginalLogLikelihood(MarginalLikelihood):
    """ Get predictive posterior distribution. """

    def __init__(
        self,
        likelihood: Likelihood,
        model: SupervisedModel,
    ) -> None:
        super(SimpleMarginalLogLikelihood, self).__init__(
            likelihood=likelihood,
            model=model
        )

    def forward(self, output, target, *args, **kwargs):
        
        mll = output.log_prob(target).mean()
        
        return mll


class ExactMarginalLogLikelihood(MarginalLikelihood):
    """ Exact marginal log likelihood for Gaussian process. """

    def __init__(
        self,
        likelihood: Likelihood,
        model: SupervisedModel,
    ) -> None:
        super(ExactMarginalLogLikelihood, self).__init__(
            likelihood=likelihood,
            model=model
        )

    def forward(self, output, target, *args, **kwargs):

        import math
        
        alpha, l_low = self.model.regressor.mll_vars

        mll = (
            -0.5 * (target.t() @ alpha)
            + torch.trace(l_low)
            + 0.5 * target.shape[0] * math.log(2.0 * math.pi)
        )

        return mll
