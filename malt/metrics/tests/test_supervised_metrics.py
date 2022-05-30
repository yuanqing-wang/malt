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
            self.y.ravel(), 1.0
        )

def test_equality():
    import torch
    import malt
    from malt.metrics.supervised_metrics import MSE, MAPE, RMSE, R2
    from malt.metrics.base_metrics import mse, mape, rmse, r2
    input_ = torch.randn(5, 1)
    target = torch.randn(5, 1)

    ds = []

    for idx in range(5):
        point = malt.Molecule(smiles="C", metadata={'y': target[idx].item()})
        ds.append(point)

    ds = malt.Dataset(ds)

    assert MSE()(DumbModel(input_), ds).item() == mse(input_, target).item()
    assert MAPE()(DumbModel(input_), ds).item() == mape(input_, target).item()
    assert RMSE()(DumbModel(input_), ds).item() == rmse(input_, target).item()
    assert R2()(DumbModel(input_), ds).item() == r2(input_, target).item()
