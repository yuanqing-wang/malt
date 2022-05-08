import pytest
import torch

def test_import():
    import malt
    from malt.metrics import supervised_metrics


class DumbModel(object):
    def __init__(self, y):
        self.y = y

    def __call__(self, *args, **kwargs):
        return torch.distributions.Normal(
            self.y.ravel().cuda(), 1.0
        )

def test_equality():
    import torch
    import malt
    from malt.metrics.supervised_metrics import MSE, MAPE, RMSE, R2
    from malt.metrics.base_metrics import mse, mape, rmse, r2
    input_ = torch.randn(5, 1)
    target = torch.randn(5, 1)

    ds = malt.Dataset()

    for idx in range(5):
        point = malt.Molecule(smiles="C", metadata={'y': target[idx].item()})
        ds.append(point)

    assert MSE()(DumbModel(input_), ds) == mse(input_, target)
    assert MAPE()(DumbModel(input_), ds) == mape(input_, target)
    assert RMSE()(DumbModel(input_), ds) == rmse(input_, target)
    assert R2()(DumbModel(input_), ds) == r2(input_, target)
