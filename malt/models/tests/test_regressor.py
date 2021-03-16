import pytest


def test_import():
    from malt.models import regressor


def test_construct():
    import torch
    import malt

    regressor = malt.models.regressor.NeuralNetworkRegressor(
        in_features=128,
        out_features=2,
    )
    h = torch.zeros(32, 128)
    theta = regressor(h)
    assert isinstance(theta, torch.Tensor)
    assert theta.shape[0] == 32
    assert theta.shape[1] == 2
