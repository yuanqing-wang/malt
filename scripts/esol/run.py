import torch
import dgl
import malt

def run(args):
    if args.regressor == "gp":
        net = malt.models.supervised_model.GaussianProcessSupervisedModel(
            representation=malt.models.representation.DGLRepresentation(
                out_features=128,
            ),
            regressor=malt.models.regressor.ExactGaussianProcessRegressor(
                in_features=128, out_features=2,
            ),
            likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
        )

    elif args.regressor == "nn":
        net = malt.models.supervised_model.SimpleSupervisedModel(
            representation=malt.models.representation.DGLRepresentation(
                out_features=128,
            ),
            regressor=malt.models.regressor.NeuralNetworkRegressor(
                in_features=128, out_features=2,
            ),
            likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
        )

    policy = malt.policy.Greedy(
        utility_function=malt.policy.utility_functions.expected_improvement,
        acquisition_size=10,
    )

    trainer = malt.trainer.get_default_trainer(n_epochs=100)
    center = malt.agents.center.NaiveCenter(name="NaiveCenter")
    vendor, assayer = malt.fake_tasks.collections.esol()

    player = malt.agents.player.AutonomousPlayer(
        name="AutonomousPlayer",
        center=center,
        policy=policy,
        model=net,
        trainer=trainer,
    )

    center.register(vendor)
    center.register(assayer)
    center.register(player)

    catalogue = vendor.catalogue()()

    for _ in range(50):
        points_to_acquire = player.prioritize(catalogue-player.portfolio)
        query_receipt = player.query(
            points_to_acquire,
            vendor=vendor,
            assayers=[assayer],
        )
        assert query_receipt is not None
        quote = player.check(query_receipt)
        order_receipt = player.order(quote)
        assert order_receipt is not None
        report = player.check(order_receipt)[0]
        player.portfolio += report.points
        player.train()

    import numpy as np
    idxs = np.array(
        [point.extra["idx"] for point in player.portfolio]
    )

    np.save(args.out + ".npy", idxs)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=str, default="GraphConv")
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--regressor", type=str, default="nn")
    args = parser.parse_args()
    run(args)
