def run():
    import torch
    import malt
    data = malt.data.collections.linear_alkanes(10)
    g, y = next(iter(data.view(batch_size=10)))

    model = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            train_targets=torch.zeros(10),
            in_features=128,
            out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )

    trainer = malt.trainer.get_default_trainer(
        without_player=True,
        n_epochs=500,
    )
    model = trainer(model, data)
    model.eval()
    y_hat = model(g)
    print(y_hat.loc)


if __name__ == "__main__":
    run()
