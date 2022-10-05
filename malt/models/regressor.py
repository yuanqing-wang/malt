# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
import gpytorch
from typing import Union

# =============================================================================
# BASE CLASSES
# =============================================================================
class Regressor(torch.nn.Module, abc.ABC):
    """Base class for a regressor."""

    def __init__(self, in_features, out_features, *args, **kwargs):
        super(Regressor, self).__init__(*args, **kwargs)
        self.in_features = in_features
        self.out_features = out_features


# =============================================================================
# KERNELS
# =============================================================================
class RBF(torch.nn.Module):
    """A Gaussian Process Kernel that hosts parameters.

    Note
    ----
    l could be either of shape 1 or hidden dim

    """

    def __init__(self, in_features, scale=0.0, variance=0.0, ard=True):

        super(RBF, self).__init__()

        if ard is True:
            self.scale = torch.nn.Parameter(scale * torch.ones(in_features))

        else:
            self.scale = torch.nn.Parameter(torch.tensor(scale))

        self.variance = torch.nn.Parameter(torch.tensor(variance))

    def distance(self, x, x_):
        """ Distance between data points. """
        return torch.norm(x[:, None, :] - x_[None, :, :], p=2, dim=2)

    def forward(self, x, x_=None):
        """ Forward pass. """
        # replicate x if there's no x_
        if x_ is None:
            x_ = x

        # for now, only allow two dimension
        assert x.dim() == 2
        assert x_.dim() == 2

        x = x * torch.exp(self.scale)
        x_ = x_ * torch.exp(self.scale)

        # (batch_size, batch_size)
        distance = self.distance(x, x_)

        # convariant matrix
        # (batch_size, batch_size)
        k = torch.exp(self.variance) * torch.exp(-0.5 * distance)

        return k


# =============================================================================
# MODULE CLASSES
# =============================================================================
class NeuralNetworkRegressor(Regressor):
    """ Regressor with neural network. """

    def __init__(
        self,
        in_features: int = 128,
        hidden_features: int = 128,
        out_features: int = 2,
        depth: int = 2,
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super(NeuralNetworkRegressor, self).__init__(
            in_features=in_features, out_features=out_features
        )
        # bookkeeping
        self.hidden_features = hidden_features
        self.out_features = out_features

        # neural network
        modules = []
        _in_features = in_features
        for idx in range(depth - 1):
            modules.append(torch.nn.Linear(_in_features, hidden_features))
            modules.append(activation)
            _in_features = hidden_features
        modules.append(torch.nn.Linear(hidden_features, out_features))

        self.ff = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.ff(x)

class ExactGaussianProcessRegressor(Regressor, gpytorch.models.ExactGP):

    is_trained = False

    def __init__(
        self,
        train_inputs: Union[torch.Tensor, None] = None,
        train_targets: Union[torch.Tensor, None] = None,
        in_features: int = 32,
        out_features: int = 2,
        *args
    ):

        # it always has to be a Gaussian likelihood anyway
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # prepare training data
        if train_targets is None:
            train_inputs = train_targets
        elif train_inputs is None:
            train_inputs = torch.ones(len(train_targets))

        super(ExactGaussianProcessRegressor, self).__init__(
            in_features,
            out_features,
            train_inputs,
            train_targets,
            likelihood,
        )

        # set debug state to false for DGKL
        gpytorch.settings.debug._state = False
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=in_features)
        )


    def forward(self, x, *args, **kwargs):
        r"""Calculate the predictive distribution given `x_te`.

        Parameters
        ----------
        x_te : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Test input.

        x_tr : `torch.Tensor`, `shape=(n_tr, hidden_dimension)`
            (Default value = None)
            Training input.

        y_tr : `torch.Tensor`, `shape=(n_tr, 1)`
            (Default value = None)
            Test input.

        sampler : `torch.optim.Optimizer` or `pinot.Sampler`
            (Default value = None)
            Sampler.

        Returns
        -------
        distribution : `torch.distributions.Distribution`
            Predictive distribution.
        """
        if self.training:
            self.set_train_data(inputs=x, strict=False)
            self.is_trained = True
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # alias forward
    condition = forward
