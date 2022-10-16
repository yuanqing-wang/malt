import pytest


def test_import():
    from malt.data import collections

def test_linear_alkane():
    import malt
    dataset = malt.data.collections.linear_alkanes(5)
    for point in dataset:
        print(point)
    assert len(dataset) == 5

def test_esol():
    import torch
    import dgl
    from malt.data import collections

    ds = collections.esol()
    assert len(ds) == 1128
    point = ds[0]
    assert isinstance(point.g, dgl.DGLGraph)
    assert isinstance(point.metadata['y'], float)

    ds = collections.lipophilicity()
    assert len(ds) == 4200
    point = ds[0]
    assert isinstance(point.g, dgl.DGLGraph)
    assert isinstance(point.metadata['y'], float)
