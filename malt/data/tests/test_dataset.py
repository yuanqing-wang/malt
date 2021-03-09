import pytest

def test_import():
    from malt.data import dataset

def test_build_dataset():
    from malt.data.dataset import Dataset
    from malt.point import Point
    p1 = Point("C")
    p2 = Point("CC")
    ds = Dataset([p1, p2])
    assert len(ds) == 2
    assert ds[0] == p1

def test_dataset_view():
    import torch
    import dgl
    from malt.data.dataset import Dataset
    from malt.point import Point
    p1 = Point("C", y=0.0)
    p2 = Point("CC", y=0.0)
    ds = Dataset([p1, p2])
    _ds = ds.view(batch_size=2)
    assert isinstance(_ds, torch.utils.data.DataLoader)
    g, y = next(iter(_ds))
    assert isinstance(g, dgl.DGLGraph)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 2
    assert y.shape[1] == 1
