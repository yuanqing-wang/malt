import pytest


def test_construct_gp():
    import torch
    import malt

    dummy_targets = torch.Tensor([4.0])
    
    regressor = malt.models.regressor.HardcodedExactGaussianProcessRegressor(
        dummy_targets,
        in_features=128,
        out_features=2,
    )


def test_gp_train():
    import torch
    import dgl
    import malt

    dummy_targets = torch.Tensor([4.0])

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.HardcodedExactGaussianProcessRegressor(
            train_targets=dummy_targets,
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    point = malt.Molecule("C")
    point.featurize()
    graph = dgl.batch([point.g])

    mll = malt.models.marginal_likelihood.HardcodedExactMarginalLogLikelihood(
        net.likelihood, net
    )

    net.train()
    y_hat = net(graph)
    loss = mll(y_hat, torch.tensor([[5.0]]))
    loss.backward()

    net.eval()
    y = net.condition(graph)
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
        regressor=malt.models.regressor.HardcodedExactGaussianProcessRegressor(
            train_targets=y,
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    if torch.cuda.is_available():
        net.cuda()


    mll = malt.models.marginal_likelihood.HardcodedExactMarginalLogLikelihood(
        net.likelihood, net
    )

    net.train()
    try:
        y_hat = net(g)
    except:
        y_hat = net(g)
    loss = mll(y_hat, y)
    loss.backward()

    y_hat = net.condition(g)
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
           out_features=128
       ),
       regressor=malt.models.regressor.HardcodedExactGaussianProcessRegressor(
           train_targets=y, in_features=128, out_features=2,
       ),
       likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )
    if torch.cuda.is_available():
        model.cuda()

    mll = malt.models.marginal_likelihood.HardcodedExactMarginalLogLikelihood(
        model.likelihood, model
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