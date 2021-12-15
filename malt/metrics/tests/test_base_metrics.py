import pytest

def test_import():
    import malt
    from malt.metrics import base_metrics

def test_equality():
    import torch
    import malt
    from malt.metrics.base_metrics import mse, mape, rmse, r2
    input = target = torch.randn(5, 1)
    assert mse(input, target) == 0.0
    assert mape(input, target) == 0.0
    assert rmse(input, target) == 0.0
    assert r2(input, target) == 1.0
