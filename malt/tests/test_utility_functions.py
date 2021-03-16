import pytest


def test_import():
    from malt.policy import utility_functions


def test_random():
    import torch
    import malt

    distribution = torch.distributions.Normal(
        torch.ones(5, 1),
        torch.ones(5, 1),
    )
    score = malt.policy.utility_functions.random(distribution)
    assert isinstance(score, torch.Tensor)
    assert score.shape[0] == 5
    assert score.shape[1] == 1


def test_uncertainty():
    import torch
    import malt

    distribution = torch.distributions.Normal(
        torch.ones(5, 1),
        torch.ones(5, 1),
    )
    score = malt.policy.utility_functions.uncertainty(distribution)
    assert torch.all(score == torch.ones(5, 1))


def test_pi():
    import torch
    import malt

    distribution = torch.distributions.Normal(
        torch.ones(5, 1),
        torch.ones(5, 1),
    )
    score = malt.policy.utility_functions.probability_of_improvement(
        distribution
    )
    assert score.shape == torch.Size([5, 1])


def test_ei():
    import torch
    import malt

    distribution = torch.distributions.Normal(
        torch.ones(5, 1),
        torch.ones(5, 1),
    )
    score = malt.policy.utility_functions.expected_improvement(distribution)
    assert score.shape == torch.Size([5, 1])


def test_ucb():
    import torch
    import malt

    distribution = torch.distributions.Normal(
        torch.ones(5, 1),
        torch.ones(5, 1),
    )
    score = malt.policy.utility_functions.upper_confidence_boundary(
        distribution
    )
    assert score.shape == torch.Size([5, 1])
