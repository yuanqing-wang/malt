import pytest


def test_import():
    from malt.data import collections

def test_linear_alkane():
    import malt
    dataset = malt.data.collections.linear_alkanes(5)
    assert len(dataset) == 5

def test_esol():
    import torch
    import dgl
    from malt.data import collections

    esol = collections.esol()
    assert len(esol) == 4200
    point = esol[0]
    assert isinstance(point.g, dgl.DGLGraph)
    assert isinstance(point.y, float)
