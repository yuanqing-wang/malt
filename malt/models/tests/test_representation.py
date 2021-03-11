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
    representation = malt.models.representation.DGLRepresentation()
    point = malt.Point(smiles="C")
    point.featurize()
    h = representation(point.g)
    assert isinstance(h, torch.Tensor)
    assert h.shape[0] == 1
    assert h.shape[1] == 1

def test_batch_forward():
    import torch
    import malt
    representation = malt.models.representation.DGLRepresentation()

    portfolio = malt.data.dataset.Dataset(
        [
            malt.Point("C"),
            malt.Point("CC")
        ]
    ).featurize_all()

    g = next(iter(portfolio.view(batch_size=2, collate_fn="batch_of_g")))
    h = representation(g)
    assert isinstance(h, torch.Tensor)
    assert h.shape[0] == 2
    assert h.shape[1] == 1
