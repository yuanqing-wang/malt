import torch
import malt

def run(args):
    # data = malt.data.collections.linear_alkanes(10)
    data = getattr(malt.data.collections, args.data)()
    data = data.shuffle(seed=2666)
    data_train, data_valid, data_test = data.split([8, 1, 1])
    g, y = next(iter(data_train.view(batch_size=len(data_train))))

    model = malt.models.SupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            depth=args.depth,
            out_features=args.width,
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
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
    rmse = (y_hat - y).pow(2).mean().pow(0.5)
    print(rmse)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="esol")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--reduce_factor", type=float, default=0.5)
    args = parser.parse_args()
    run(args)
