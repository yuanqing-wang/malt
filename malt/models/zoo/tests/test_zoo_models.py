import pytest

def test_import():
    from malt.models.zoo import __ALL__

def test_dimension():
    import torch
    import dgl
    import malt
    from malt.models.zoo import __ALL__
    g = dgl.rand_graph(5, 8)
    h = torch.randn(5, 3)
    for layer in __ALL__:
        layer = getattr(malt.models.zoo, layer)(3, 4)
        h1 = layer(g, h)
