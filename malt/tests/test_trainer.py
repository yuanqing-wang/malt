import pytest

def test_training_on_linear_alkane_without_player():
    import malt
    data = malt.data.collections.linear_alkanes(10)
    representation = malt.models.representation.DGLRepresentation(out_features=32)
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=32, out_features=1,
    )
    likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood()
    model = malt.models.supervised_model.SimpleSupervisedModel(
        representation=representation,
        regressor=regressor,
        likelihood=likelihood,
    )

    trainer = malt.trainer.get_default_trainer(without_player=True)
    model = trainer(model, data)

def test_training_on_linear_alkane_with_player():
    import malt
    data = malt.data.collections.linear_alkanes(10)
    merchant = malt.agents.merchant.DatasetMerchant(data)
    assayer = malt.agents.assayer.DatasetAssayer(data)

    representation = malt.models.representation.DGLRepresentation(out_features=32)
    regressor=malt.models.regressor.NeuralNetworkRegressor(
        in_features=32, out_features=1,
    )
    likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood()
    model = malt.models.supervised_model.SimpleSupervisedModel(
        representation=representation,
        regressor=regressor,
        likelihood=likelihood,
    )

    mll = malt.models.marginal_likelihood.ExactMarginalLogLikelihood(
        model.regressor.likelihood,
        model
    )


    player = malt.agents.player.SequentialModelBasedPlayer(
        model=model,
        merchant=merchant,
        assayer=assayer,
        marginal_likelihood=mll,
        policy=malt.policy.Greedy(),
        trainer=malt.trainer.get_default_trainer(),
    )

    player.step()
