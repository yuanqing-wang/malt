# =============================================================================
# IMPORTS
# =============================================================================
import dgllife
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from .dataset import Dataset
from ..point import Point

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _dataset_from_dgllife(dgllife_dataset):
    return Dataset(
        [Point(smiles, g, y.item()) for smiles, g, y in dgllife_dataset]
    )


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
COLLECTIONS = [
    "ESOL",
    "FreeSolv",
    "Lipophilicity",
]

for collection in COLLECTIONS:

    def _get_collection():
        ds = getattr(dgllife.data, collection)(
            smiles_to_bigraph, CanonicalAtomFeaturizer()
        )
        return _dataset_from_dgllife(ds)

    globals()[collection.lower()] = _get_collection
