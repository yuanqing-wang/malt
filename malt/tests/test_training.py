import pytest

def test_training():
    import malt

    net = malt.models.supervised_model.SimpleSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(out_features=128),
        regressor=malt.models.regressor.NeuralNetworkRegressor(in_features=128, out_features=1),
        likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    )
    trainer = malt.trainer.get_default_trainer()
    center = malt.agents.center.NaiveCenter(name="NaiveCenter")
    player = malt.agents.player.AutonomousPlayer(
        name="AutonomousPlayer",
        center=center,
        policy=None,
        model=net,
        trainer=trainer,
    )

    p0 = malt.Point(smiles="C", y=1.0).featurize()
    p1 = malt.Point(smiles="CC", y=2.0).featurize()
    player.portfolio += [p0, p1]
    player.net = trainer(player)
