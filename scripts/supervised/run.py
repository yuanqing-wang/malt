import torch
import dgl
import malt 


def run(args):
    data = getattr(malt.data.collections, args.data)()
    data.shuffle(seed=2666)
    ds_tr, ds_vl, ds_te = data.split([8, 1, 1])
    model = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128, out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )
    trainer = malt.trainer.get_default_trainer(without_player=True, n_epochs=3000)
    model = trainer(model, ds_tr)

    r2 = malt.metrics.supervised_metrics.R2()(model, ds_te)
    print(r2)

    rmse = malt.metrics.supervised_metrics.RMSE()(model, ds_te)
    print(rmse)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="esol")
    args = parser.parse_args()
    run(args)
