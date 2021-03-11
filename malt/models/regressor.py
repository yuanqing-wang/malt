# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class Regressor(torch.nn.Module, abc.ABC):
    """ Base class for a regressor.

    """
    def __init__(self, out_features=2, *args, **kwargs):
        super(Regressor, self).__init__()
        self.out_features = out_features

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

# =============================================================================
# MODULE CLASSES
# =============================================================================
class NeuralNetworkRegressor(Regressor):
    """ Regressor with neural network. """
    def __init__(
            self,
            in_features: int=128,
            hidden_features: int=128,
            out_features: int=128,
            width: int = 2,
            activation: torch.nn.Module = torch.nn.ReLU(),
        ):
        super(NeuralNetworkRegressor, self).__init__(
            out_features=out_features
        )
        # bookkeeping
        self.width = width
        self.hidden_features = hidden_features

        # neural network
        modules = []
        _in_features=in_features
        for idx in range(width-1):
            modules.append(torch.nn.Linear(_in_features, hidden_features))
            modules.append(activation)
            _in_features = hidden_features
        modules.append(torch.nn.Linear(hidden_features, out_features))

        self.ff = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.ff(x)
        
