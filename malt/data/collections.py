# =============================================================================
# IMPORTS
# =============================================================================
import dgllife
from malt.data.dataset import Dataset
from malt.molecule import Molecule, AssayedMolecule
import copy

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _dataset_from_dgllife(dgllife_dataset):
    idx = 0
    ds = []
    for smiles, g, y in dgllife_dataset:
        molecule = Molecule(smiles, g, metadata={'idx': idx, 'y': y.item()})
        idx += 1
        ds.append(molecule)

    ds = Dataset(ds)

    return ds


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
COLLECTIONS = [
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
]

def _get_collection(collection):
    from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
    ds = getattr(dgllife.data, collection)(
        smiles_to_bigraph, CanonicalAtomFeaturizer()
    )
    return _dataset_from_dgllife(ds)

from functools import partial

for collection in COLLECTIONS:
    globals()[collection.lower()] = partial(_get_collection, collection=collection)

def linear_alkanes(max_carbon=10):
    """A toy dataset with linear alkanes from 1 to `max_carbon` carbons.

    Parameters
    ----------
    max_carbon : int
        Maximum number of carbons in the molecules generated.

    Returns
    -------
    Dataset
        A dataset containing alanes.

    Examples
    --------
    dataset = count_carbons(10)


    """

    dataset =  Dataset([Molecule(idx * "C") for idx in range(1, max_carbon+1)])
    def annotate(molecule):
        molecule.metadata = {'y': float(len(molecule.smiles))}
        return molecule
    dataset = dataset.apply(annotate)
    return dataset
