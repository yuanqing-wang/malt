import pytest


def test_import():
    from malt.models import representation


def test_construct():
    import malt
    import torch

    representation = malt.models.representation.DGLRepresentation()
    assert isinstance(representation, torch.nn.Module)


def test_forward():
    import torch
    import malt

    representation = malt.models.representation.DGLRepresentation(
        out_features=8
    )
    point = malt.Molecule(smiles="C")
    point.featurize()
    h = representation(point.g)
    assert isinstance(h, torch.Tensor)
    assert h.shape[0] == 1
    assert h.shape[1] == 8


def test_batch_forward():
    import torch
    import malt

    representation = malt.models.representation.DGLRepresentation(
        out_features=8,
    )
    if torch.cuda.is_available():
        representation.cuda()

    portfolio = malt.data.dataset.Dataset(
        [malt.Molecule("C"), malt.Molecule("CC")]
    ).featurize_all()

    g = next(iter(portfolio.view(batch_size=2, by='g')))
    h = representation(g)
    assert isinstance(h, torch.Tensor)
    assert h.shape[0] == 2
    assert h.shape[1] == 8
