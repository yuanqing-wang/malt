import torch
import malt

def run(args):
    # data = malt.data.collections.linear_alkanes(10)
    data = getattr(malt.data.collections, args.data)()
    data = data.shuffle(seed=2666)
    data_train, data_valid, data_test = data.split([8, 1, 1])
    g, y = next(iter(data_train.view(batch_size=len(data_train))))
    regressor = getattr(
        malt.models.regressor,
        {
            "nn": "NeuralNetworkRegressor", "gp": "ExactGaussianProcessRegressor"
        }[args.regressor],
    )

    model = malt.models.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            depth=args.depth,
            out_features=args.width,
        ),
        regressor=regressor(
            num_points=len(data_train),
            in_features=args.width,
        ),
    )

    trainer = malt.trainer.get_default_trainer(
        without_player=True,
        n_epochs=2000,
        learning_rate=args.learning_rate,
        reduce_factor=args.reduce_factor,
    )
    model = trainer(model, data_train, data_valid)
    model.eval()

    g, y = next(iter(data_test.view(batch_size=len(data_test))))
    y_hat = model(g).loc
    rmse_test = (y_hat - y).pow(2).mean().pow(0.5)

    g, y = next(iter(data_valid.view(batch_size=len(data_test))))
    y_hat = model(g).loc
    rmse_valid = (y_hat - y).pow(2).mean().pow(0.5)

    import json
    import pandas as pd
    key = dict(vars(args))
    key.pop("out")
    key = json.dumps(key)
    df = pd.DataFrame.from_dict(
        {key: [rmse_valid, rmse_test]},
        orient="index",
        columns=["vl", "te"]
    )
    df.to_csv(args.out, mode="a")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="esol")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--regressor", type=str, default="gp")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--reduce_factor", type=float, default=0.5)
    parser.add_argument("--out", type=str, default="out.csv")
    args = parser.parse_args()
    run(args)
