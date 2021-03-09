import pytest


def test_import():
    from malt import point

def test_build_point():
    from malt.point import Point
    p = Point("C")

def test_featurize():
    import dgl
    from malt.point import Point
    p = Point("C")
    p.featurize()
    assert p.is_featurized()
    assert isinstance(
        p.g,
        dgl.DGLGraph,
    )
