import pytest


def test_import():
    import malt
    import malt.agents


def test_player_with_linear_alkane():
    import malt
    import torch
    from malt.agents.player import SequentialModelBasedPlayer

    dataset = malt.data.collections.linear_alkanes(10)

    model = malt.models.supervised_model.SupervisedModel(
               representation=malt.models.representation.DGLRepresentation(
                   out_features=128
               ),
               regressor=malt.models.regressor.NeuralNetworkRegressor(
                   in_features=128,
               ),
    )

    player = SequentialModelBasedPlayer(
       model = model,
       policy=malt.policy.Greedy(),
       trainer=malt.trainer.get_default_trainer(),
       merchant=malt.agents.merchant.DatasetMerchant(dataset),
       assayer=malt.agents.assayer.DatasetAssayer(dataset),
    )

    while True:
        if player.step() is None:
            break
