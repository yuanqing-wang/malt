import pytest


def test_construct_gpytorch():
    import torch
    import malt
    regressor = malt.models.regressor.ExactGaussianProcessRegressor(128)

def test_gpytorch_train():

    import torch
    import dgl
    import malt
    import gpytorch

    dummy_targets = torch.tensor([[4.0]])

    net = malt.models.supervised_model.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=32
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=32,
            num_points=1,
        ),
    )

    point = malt.Molecule("CCCCCCC")
    point.featurize()
    graph = dgl.batch([point.g])

    net.train()
    # y_pred = net(graph)
    # loss = mll(y_pred, torch.tensor([[2.0]]))
    loss = net.loss(graph, torch.tensor([[2.0]]))
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
    print(y.shape)

    net = malt.models.supervised_model.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128,
            num_points = len(y)
        ),
    )

    if torch.cuda.is_available():
        net.cuda()

    net.train()
    loss = net.loss(g, y)
    loss.backward()

    net.eval()
    y_hat = net(g)
    assert y_hat.mean.shape[0] == 10
    assert len(y_hat.mean.shape) == 1
