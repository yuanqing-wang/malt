import pytest

def test_construct_gp():
    import torch
    import malt

    regressor = malt.models.regressor.ExactGaussianProcessRegressor(
        in_features=128,
        out_features=2,
    )


def test_blind_condition():
    import torch
    import malt

    net = malt.models.supervised_model.GaussianProcessSupervisedModel(
        representation=malt.models.representation.DGLRepresentation(
            out_features=128
        ),
        regressor=malt.models.regressor.ExactGaussianProcessRegressor(
            in_features=128, out_features=2,
        ),
        likelihood=malt.models.likelihood.HeteroschedasticGaussianLikelihood(),
    )
