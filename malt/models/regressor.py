# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
import gpytorch

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


class ExactGaussianProcessRegressor(Regressor):
    epsilon = 1e-5

    def __init__(
        self,
        in_features: int = 128,
        out_features: int = 2,
        kernel_factory: torch.nn.Module = RBF,
        log_sigma: float = -3.0,
    ):
        assert out_features == 2
        super(ExactGaussianProcessRegressor, self).__init__(
            in_features=in_features,
            out_features=out_features,
        )

        # construct kernel
        self.kernel = kernel_factory(
            in_features=in_features,
        )

        self.in_features = in_features
        self.log_sigma = torch.nn.Parameter(
            torch.tensor(log_sigma),
        )

    def _get_kernel_and_auxiliary_variables(
        self,
        x_tr,
        y_tr,
        x_te=None,
    ):
        """ Get kernel and auxiliary variables for forward pass. """

        # compute the kernels
        k_tr_tr = self._perturb(self.kernel.forward(x_tr, x_tr))

        if x_te is not None:  # during test
            k_te_te = self._perturb(self.kernel.forward(x_te, x_te))
            k_te_tr = self._perturb(self.kernel.forward(x_te, x_tr))
            # k_tr_te = self.forward(x_tr, x_te)
            k_tr_te = k_te_tr.t()  # save time

        else:  # during train
            k_te_te = k_te_tr = k_tr_te = k_tr_tr

        # (batch_size_tr, batch_size_tr)
        k_plus_sigma = k_tr_tr + torch.exp(self.log_sigma) * torch.eye(
            k_tr_tr.shape[0], device=k_tr_tr.device
        )

        # (batch_size_tr, batch_size_tr)
        l_low = torch.linalg.cholesky(k_plus_sigma)
        l_up = l_low.t()

        # (batch_size_tr. 1)
        l_low_over_y, _ = torch.triangular_solve(
            input=y_tr, A=l_low, upper=False
        )

        # (batch_size_tr, 1)
        alpha, _ = torch.triangular_solve(
            input=l_low_over_y, A=l_up, upper=True
        )

        return k_tr_tr, k_te_te, k_te_tr, k_tr_te, l_low, alpha

    def condition(self, x_te, *args, x_tr=None, y_tr=None, **kwargs):
        r"""Calculate the predictive distribution given `x_te`.

        Note
        ----
        Here we allow the speicifaction of sampler but won't actually
        use it here in this version.

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

        # get parameters
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(x_tr, y_tr, x_te)

        # compute mean
        # (batch_size_te, 1)
        mean = k_te_tr @ alpha

        # (batch_size_tr, batch_size_te)
        v, _ = torch.triangular_solve(input=k_tr_te, A=l_low, upper=False)

        # (batch_size_te, batch_size_te)
        variance = k_te_te - v.t() @ v

        # ensure symetric
        variance = 0.5 * (variance + variance.t())

        # $ p(y|X) = \int p(y|f)p(f|x) df $
        # variance += torch.exp(self.log_sigma) * torch.eye(
        #         *variance.shape,
        #         device=variance.device)

        # construct noise predictive distribution
        distribution = (
            torch.distributions.multivariate_normal.MultivariateNormal(
                mean.flatten(), variance
            )
        )

        return distribution

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

    def loss(self, x_tr, y_tr, *args, **kwargs):
        r"""Compute the loss.
        Note
        ----
        Defined to be negative Gaussian likelihood.
        Parameters
        ----------
        x_tr : `torch.Tensor`, `shape=(n_training_data, hidden_dimension)`
            Input of training data.
        y_tr : `torch.Tensor`, `shape=(n_training_data, 1)`
            Target of training data.
        Returns
        -------
        nll : `torch.Tensor`, `shape=(,)`
            Negative log likelihood.
        """
        # point data to object
        self._x_tr = x_tr
        self._y_tr = y_tr

        # get the parameters
        (
            k_tr_tr,
            k_te_te,
            k_te_tr,
            k_tr_te,
            l_low,
            alpha,
        ) = self._get_kernel_and_auxiliary_variables(x_tr, y_tr)

        import math

        # we return the exact nll with constant
        nll = (
            0.5 * (y_tr.t() @ alpha)
            + torch.trace(l_low)
            + 0.5 * y_tr.shape[0] * math.log(2.0 * math.pi)
        )

        return nll


class GPyTorchExactRegressor(Regressor, gpytorch.models.ExactGP):
    
    train_inputs = torch.ones(1)
    train_targets = torch.ones(1)
    
    def __init__(
        self,
        in_features: int = 32,
        out_features: int = 2,
        *args
    ):

        # it always has to be a Gaussian likelihood anyway
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(GPyTorchExactRegressor, self).__init__(
            in_features,
            out_features,
            self.train_inputs,
            self.train_targets,
            likelihood,
        )

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
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # alias forward
    condition = forward

    def loss(self, x_tr, y_tr, *args, **kwargs):
        r"""Compute the loss.
        Note
        ----
        Defined to be negative Gaussian likelihood.
        Parameters
        ----------
        x_tr : `torch.Tensor`, `shape=(n_training_data, hidden_dimension)`
            Input of training data.
        y_tr : `torch.Tensor`, `shape=(n_training_data, 1)`
            Target of training data.
        Returns
        -------
        nll : `torch.Tensor`, `shape=(,)`
            Negative log likelihood.
        """
        self.set_train_data(x_tr, y_tr.ravel(), strict=False)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        y_pred = self(x_tr)
        return -mll(y_pred, y_tr)