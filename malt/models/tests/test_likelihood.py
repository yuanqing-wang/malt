import pytest

def test_import():
    from malt.models import likelihood

def test_construct_homoschedastic_gaussian():
    import torch
    import malt
    likelihood = malt.models.likelihood.HomoschedasticGaussianLikelihood()
    theta = torch.zeros(32, 1)
    distribution = likelihood.condition(theta)
    assert distribution.event_shape == torch.Size([])
    assert distribution.batch_shape == torch.Size([32, 1])

def test_construct_heteroschedastic_gaussian():
    import torch
    import malt
    likelihood = malt.models.likelihood.HeteroschedasticGaussianLikelihood()
    theta = torch.zeros(32, 2)
    distribution = likelihood.condition(theta)
    assert distribution.event_shape == torch.Size([])
    assert distribution.batch_shape == torch.Size([32, 1])
