import pytest


def test_construct_gpytorch():
    import torch
    import malt

    dummy_targets = torch.Tensor([0.0])

    regressor = malt.models.regressor.ExactGaussianProcessRegressor(
        dummy_targets,
        in_features=128,
        out_features=2,
    )


def test_blind_condition():
    import torch
    import dgl
    import malt

    dummy_targets = torch.Tensor([0.0])

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            dummy_targets,
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    point = malt.Molecule("C")
    point.featurize()
    graph = dgl.batch([point.g])

    net.eval()
    y = net.condition(graph)
    assert y.mean.item() == 0.0

def test_gpytorch_train():

    import torch
    import dgl
    import malt
    import gpytorch

    dummy_targets = torch.tensor([[4.0]])

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=32
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            train_targets=dummy_targets,
            in_features=32,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    point = malt.Molecule("CCCCCCC")
    point.featurize()
    graph = dgl.batch([point.g])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(net.regressor.likelihood, net)

    net.train()
    y_pred = net(graph)
    loss = mll(y_pred, torch.tensor([[2.0]]))
    loss.backward()

    net.eval()
    y = net(graph)
    assert y.mean.item() != 0.0

def test_gp_shape():
    import torch
    import dgl
    import malt

    dataset = malt.data.collections.linear_alkanes(10)
    dataset_loader = dataset.view(batch_size=len(dataset))
    g, y = next(iter(dataset_loader))

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            train_targets=y,
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    if torch.cuda.is_available():
        net.cuda()

    mll = malt.models.marginal_likelihood.ExactMarginalLogLikelihood(
        net.regressor.likelihood, net
    )

    net.train()
    y_pred = net(g)
    loss = mll(y_pred, y).mean()
    loss.backward()

    net.eval()
    y_hat = net(g)
    assert y_hat.mean.shape[0] == 10
    assert len(y_hat.mean.shape) == 1

def test_gp_integrate():
    import malt
    import torch
    from malt.agents.player import SequentialModelBasedPlayer

    dataset = malt.data.collections.linear_alkanes(10)
    g, y = dataset.batch()
    model = malt.models.supervised_model.GaussianProcessSupervisedModel(
       representation=malt.models.representation.DGLRepresentation(
           out_features=32
       ),
       regressor=malt.models.regressor.ExactGaussianProcessRegressor(
           y, in_features=32, out_features=2,
       ),
       likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )
    if torch.cuda.is_available():
        model.cuda()

    mll = malt.models.marginal_likelihood.ExactMarginalLogLikelihood(
        model.regressor.likelihood,
        model
    )

    player = SequentialModelBasedPlayer(
        model = model,
        policy=malt.policy.Greedy(),
        trainer=malt.trainer.get_default_trainer(),
        marginal_likelihood=mll,
        merchant=malt.agents.merchant.DatasetMerchant(dataset),
        assayer=malt.agents.assayer.DatasetAssayer(dataset),
    )

    while True:
        if player.step() is None:
            break
