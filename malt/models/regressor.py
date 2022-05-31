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
    r"""A Gaussian Process Kernel that hosts parameters.

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


class HardcodedExactGaussianProcessRegressor(Regressor):

    """ Hardcoded Exact GP. """
    
    is_trained = False
    epsilon = 1e-5

    def __init__(
        self,
        train_inputs: Union[torch.Tensor, None] = None,
        train_targets: Union[torch.Tensor, None] = None,
        in_features: int = 32,
        out_features: int = 2,
        kernel_factory: torch.nn.Module = RBF,
        log_sigma: float = -3.0,
    ):
        assert out_features == 2
        super(HardcodedExactGaussianProcessRegressor, self).__init__(
            in_features=in_features,
            out_features=out_features,
        )

        self.train_inputs = train_inputs
        self.train_targets = train_targets

        # construct kernel
        self.kernel = kernel_factory(
            in_features=in_features,
        )

        self.log_sigma = torch.nn.Parameter(
            torch.tensor(log_sigma),
        )

    def _perturb(self, k):
        """Add small noise `epsilon` to the diagonal of covariant matrix.
        Parameters
        ----------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Covariant matrix.
        Returns
        -------
        k : `torch.Tensor`, `shape=(n_data_points, n_data_points)`
            Preturbed covariant matrix.
        """
        # introduce noise along the diagonal
        noise = self.epsilon * torch.eye(*k.shape, device=k.device)
        return k + noise

    def _get_kernel_and_auxiliary_variables(
        self,
        x_tr,
        y_tr,
        x_te,
    ):
        """ Get kernel and auxiliary variables for forward pass. """

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))

        if self.training:
            k_te_te = k_te_tr = k_tr_te = k_tr_tr
        else:
            k_te_te = self._perturb(self.kernel.forward(x_te, x_te))
            k_te_tr = self._perturb(self.kernel.forward(x_te, x_tr))
            k_tr_te = k_te_tr.t()

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
            k_tr_tr.shape[0], device=k_tr_tr.device
        )

        # (batch_size_tr, batch_size_tr)
        l_low = torch.linalg.cholesky(k_plus_sigma)
        l_up = l_low.t()

        if y_tr.dim() != 2:
            y_tr = y_tr.unsqueeze(1)

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr, A=l_low, upper=False
        )

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y, A=l_up, upper=True
        )

        return k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha

    def forward(self, x_te, *args, **kwargs):
        r"""Calculate the predictive distribution given `x_te`.

        Parameters
        ----------
        x : `torch.Tensor`, `shape=(n_te, hidden_dimension)`
            Test input.

        Returns
        -------
        y_pred : `torch.distributions.Distribution`
            Predictive distribution.

        """
        if self.training:
            self.set_train_data(inputs=x_te)
            self.is_trained = True

        x_tr = self.train_inputs
        y_tr = self.train_targets

        # get parameters
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(x_tr, y_tr, x_te)

        # gather inputs for marginal log likelihood
        # if self.training:
        self.mll_vars = alpha, l_low

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(input=k_tr_te, A=l_low, upper=False)

        # (batch_size_te, batch_size_te)
        variance = k_te_te - v.t() @ v

        # ensure symmetric
        variance = 0.5 * (variance + variance.t())

        # construct noise predictive distribution
        y_pred = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                mean.flatten(), variance
            )
        )

        return y_pred

    def set_train_data(
        self, inputs=None, targets=None, *args, **kwargs
    ):
        if inputs is not None:
            self.train_inputs = inputs
        if targets is not None:
            self.train_targets = targets
        return self

    condition = forward


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
