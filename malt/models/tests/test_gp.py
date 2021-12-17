import pytest


def test_construct_gp():
    import torch
    import malt

    regressor = malt.models.regressor.ExactGaussianProcessRegressor(
        in_features=128,
        out_features=2,
    )


def test_blind_condition():
    import torch
    import dgl
    import malt

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    point = malt.Point("C")
    point.featurize()
    graph = dgl.batch([point.g])

    y = net.condition(graph)
    assert y.mean.item() == 0.0

def test_gp_train():
    import torch
    import dgl
    import malt

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    point = malt.Point("C")
    point.featurize()
    graph = dgl.batch([point.g])

    y = net.loss(graph, torch.tensor([[5.0]]))
    y.backward()

    y = net.condition(graph)
    assert y.mean.item() != 0.0


def test_gp_shape():
    import torch
    import dgl
    import malt

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    dataset = malt.data.collections.linear_alkanes(10)
    dataset_loader = dataset.view(batch_size=len(dataset))
    g, y = next(iter(dataset_loader))
    loss = net.loss(g, y)

    y_hat = net.condition(g)
    assert y_hat.mean.shape[0] == 10
    assert len(y_hat.mean.shape) == 1

def test_gp_integrate():
    import malt
    from malt.agents.player import SequentialModelBasedPlayer
    dataset = malt.data.collections.linear_alkanes(10)

    player = SequentialModelBasedPlayer(
       model = malt.models.supervised_model.GaussianProcessSupervisedModel(
           representation=malt.models.representation.DGLRepresentation(
               out_features=128
           ),
           regressor=malt.models.regressor.ExactGaussianProcessRegressor(
               in_features=128, out_features=2,
           ),
           likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
       ),
       policy=malt.policy.Greedy(),
       trainer=malt.trainer.get_default_trainer(),
       merchant=malt.agents.merchant.DatasetMerchant(dataset),
       assayer=malt.agents.assayer.DatasetAssayer(dataset),
    )

    while True:
        if player.step() is None:
            break
