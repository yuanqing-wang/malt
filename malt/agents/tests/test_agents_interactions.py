import pytest


def test_import():
    import malt
    import malt.agents


def test_player_with_linear_alkane():
    import malt
    import torch
    from malt.agents.player import SequentialModelBasedPlayer
    
    dataset = malt.data.collections.linear_alkanes(10)

    model = malt.models.supervised_model.SimpleSupervisedModel(
               representation=malt.models.representation.DGLRepresentation(
                   out_features=128
               ),
               regressor=malt.models.regressor.NeuralNetworkRegressor(
                   in_features=128, out_features=1
               ),
               likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    )

    mll = malt.models.marginal_likelihood.SimpleMarginalLogLikelihood(
        model.likelihood,
        model
    )

    if torch.cuda.is_available():
        model.cuda()


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
