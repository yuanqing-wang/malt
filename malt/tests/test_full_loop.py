import pytest


def test_full_loop():
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
    trainer = malt.trainer.get_default_trainer()
    center = malt.agents.center.NaiveCenter(name="NaiveCenter")
    policy = malt.policy.Greedy()
    player = malt.agents.player.AutonomousPlayer(
        name="AutonomousPlayer",
        center=center,
        policy=policy,
        model=net,
        trainer=trainer,
    )

    p0 = malt.Point(smiles="C", y=1.0).featurize()
    p1 = malt.Point(smiles="CC", y=2.0).featurize()
    player.portfolio += [p0, p1]
    model = player.train()
    assert isinstance(model, torch.nn.Module)
