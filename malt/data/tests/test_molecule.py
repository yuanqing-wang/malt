import pytest

def test_molecule_import():
    from malt.molecule import Molecule

def test_molecule():
    from malt.molecule import Molecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    mol = Molecule(smiles)

def test_assayed_molecule_import():
    from malt.molecule import AssayedMolecule

def test_assayed_molecule():
    from malt.molecule import AssayedMolecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    metadata = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
        'assay_2': [
            {'k': 32.0},
            {'k': 32.0},
        ]
    }

    mol = AssayedMolecule(
        smiles=smiles,
        metadata=metadata
    )


def test_assayed_molecule__add__():
    from malt.molecule import AssayedMolecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    metadata_1 = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
        'assay_2': [
            {'k': 32.0},
            {'k': 32.0},
        ]
    }

    metadata_2 = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
    }


    mol_1 = AssayedMolecule(
        smiles=smiles,
        metadata=metadata_1
    )

    mol_2 = AssayedMolecule(
        smiles=smiles,
        metadata=metadata_2
    )

    mol_3 = mol_1 + mol_2

    print(mol_3.metadata)

    for assay in mol_1.metadata:
        assert assay in mol_3.metadata

        len_1 = len_2 = 0
        if assay in mol_1.metadata:
            len_1 = len(mol_1[assay])

        if assay in mol_2.metadata:
            len_2 = len(mol_2[assay])

        assert len(mol_3[assay]) == len_1 + len_2


def test_assayed_molecule__getitem__():
    from malt.molecule import AssayedMolecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    metadata = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
    }

    mol = AssayedMolecule(
        smiles=smiles,
        metadata=metadata
    )

    assert metadata['assay_1'] == mol['assay_1']


def test_assayed_molecule__contains__():
    from malt.molecule import AssayedMolecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    metadata = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
    }

    mol = AssayedMolecule(
        smiles=smiles,
        metadata=metadata
    )

    assert 'assay_1' in mol

def test_assayed_molecule__eq__():
    from malt.molecule import AssayedMolecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    metadata = {
        'assay_1': [
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
            {'c': 1e-3, 'y': 1e-2},
        ],
    }

    metadata_2 = {
        'assay_1': [
            {'c': 0.0, 'y': 0.0},
        ],
    }

    mol_1 = AssayedMolecule(
        smiles=smiles,
        metadata=metadata
    )

    mol_2 = AssayedMolecule(
        smiles=smiles,
        metadata=metadata_2
    )

    assert mol_1 == mol_1
    assert mol_1 != mol_2
