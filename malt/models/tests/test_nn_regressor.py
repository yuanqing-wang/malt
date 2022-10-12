import pytest


def test_import():
    from malt.models import regressor


def test_construct_nn():
    import torch
    import malt

    regressor = malt.models.regressor.NeuralNetworkRegressor(
        in_features=128,
    )
    h = torch.zeros(32, 128)
    theta = regressor(h)
    assert isinstance(theta, torch.distributions.Normal)
    assert theta.mean.shape[0] == 32
