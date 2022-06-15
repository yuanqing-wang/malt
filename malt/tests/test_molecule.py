import pytest

def test_molecule_import():
    from malt.molecule import Molecule

def test_molecule():
    from malt.molecule import Molecule

    # make linear alkane
    num_carbons = 10
    smiles = "C" * num_carbons
    mol = Molecule(smiles)
