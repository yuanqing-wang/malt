import pytest


def test_import():
    from malt.models import supervised_model


def test_construct():
    import torch
    import malt

    net = malt.models.supervised_model.SimpleSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.NeuralNetworkRegressor(
            in_features=128, out_features=1
        ),
        likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    )


def test_forward():
    import torch
    import malt

    net = malt.models.supervised_model.SimpleSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.NeuralNetworkRegressor(
            in_features=128, out_features=1
        ),
        likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    )

    point = malt.Molecule(smiles="C").featurize()
    distribution = net.condition(point.g)
    assert isinstance(distribution, torch.distributions.Distribution)
    assert distribution.batch_shape == torch.Size([1, 1])

    net.train()
    loss = net.loss(point.g, torch.tensor([0.0])).backward()
