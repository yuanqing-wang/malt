import pytest
import torch

def test_import():
    import malt
    from malt.metrics import supervised_metrics


class DumbModel(object):
    def __init__(self, y):
        self.y = y

    def condition(self, *args, **kwargs):
        return torch.distributions.Normal(
            self.y, 1.0
        )

def test_equality():
    import torch
    import malt
    from malt.metrics.supervised_metrics import MSE, MAPE, RMSE, R2
    from malt.metrics.base_metrics import mse, mape, rmse, r2
    input = torch.randn(5, 1)
    target = torch.randn(5, 1)

    ds = malt.Dataset()

    for idx in range(5):
        point = malt.Point(smiles="C", y=target[idx].item())
        ds.append(point)

    assert MSE()(DumbModel(input), ds) == mse(input, target)
    assert MAPE()(DumbModel(input), ds) == mape(input, target)
    assert RMSE()(DumbModel(input), ds) == rmse(input, target)
    assert R2()(DumbModel(input), ds) == r2(input, target)
