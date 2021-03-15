import pytest

def test_training():
    import malt

    net = malt.models.supervised_model.SimpleSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(out_features=128),
        regressor=malt.models.regressor.NeuralNetworkRegressor(in_features=128, out_features=1),
        likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    )
    trainer = malt.trainer.get_default_trainer()

    player = malt.agents.player.AutonomousPlayer(
        name="AutonomousPlayer",
        center=None,
        policy=None,
        model=net,
        trainer=trainer,
    )
