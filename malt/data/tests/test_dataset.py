import pytest


def test_import():
    from malt.data import dataset


def test_build_dataset():
    from malt.data.dataset import Dataset
    from malt.molecule import Molecule

    m1 = Molecule("C")
    m2 = Molecule("CC")
    ds = Dataset([m1, m2])
    assert len(ds) == 2
    assert ds[0] == m1

def test_split_dataset():
    from malt.data.dataset import Dataset
    from malt.molecule import Molecule

    m1 = Molecule("C")
    m2 = Molecule("CC")
    ds = Dataset([m1, m2])
    ds0, ds1 = ds.split([0.5, 0.5])
    assert len(ds0) == 1
    assert len(ds1) == 1

def test_dataset_subtraction():
    from malt.data.dataset import Dataset
    from malt.molecule import Molecule

    p0 = Molecule("C")
    m1 = Molecule("CC")

    dataset0 = Dataset([p0])
    dataset1 = Dataset([p0, m1])
    assert len(dataset1 - dataset0) == 1

def test_dataset_view():
    import torch
    import dgl
    from malt.data.dataset import Dataset
    from malt.molecule import Molecule

    m1 = Molecule("C", metadata={'y': 0.0})
    m2 = Molecule("CC", metadata={'y': 0.0})
    ds = Dataset([m1, m2])
    _ds = ds.view(batch_size=2, by=['g', 'y'])
    assert isinstance(_ds, torch.utils.data.DataLoader)
    g, y = next(iter(_ds))
    assert isinstance(g, dgl.DGLGraph)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 2
    assert y.shape[1] == 1


def test_dataset_view_batch_of_g():
    import dgl
    from malt.data.dataset import Dataset
    from malt.molecule import Molecule

    m1 = Molecule("C", metadata={'y': 0.0})
    m2 = Molecule("CC", metadata={'y': 0.0})
    ds = Dataset([m1, m2])
    _ds = ds.view(batch_size=2, collate_fn="batch_of_g")
    g = next(iter(_ds))
    return type(g)
    assert isinstance(g, dgl.DGLGraph)
    assert (
        g.number_of_nodes() == m1.g.number_of_nodes() + m2.g.number_of_nodes()
    )

def test_dataset_with_assayed_molecule_view():
    import torch
    import dgl
    from malt.data.utils import collate_metadata
    from malt.data.dataset import Dataset
    from malt.molecule import AssayedMolecule

    m1 = AssayedMolecule("C", metadata={'assay': [{'y': 0.0}]})
    m2 = AssayedMolecule("CC", metadata={'assay': [{'y': 0.0}]})
    ds = Dataset([m1, m2])
    _ds = ds.view(
        batch_size=2,
        assay='assay',
        by=['g', 'y'],
        batch_meta=collate_metadata
    )
    assert isinstance(_ds, torch.utils.data.DataLoader)
    g, y = next(iter(_ds))
    assert isinstance(g, dgl.DGLGraph)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == 2
    assert y.shape[1] == 1