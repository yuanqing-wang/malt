# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, Iterable

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def collate_metadata(molecule, key, **kwargs):
    """ Batches metadata from Molecule.
    Parameters
    ----------
    molecule : malt.Molecule
    assay : Union[None, str]
        Filter metadata using assay key.
    key : str
        Attribute of class on which to batch.
    Returns
    -------
    meta : list
        Elements are metadata from assay.
    """
    from malt import AssayedMolecule

    if isinstance(molecule, AssayedMolecule):
        return _collate_assayedmolecule_metadata(molecule, key, **kwargs)
    else:
        return _collate_molecule_metadata(molecule, key)


def _collate_molecule_metadata(
        molecule, key
    ):

    if not molecule.metadata:
        raise RuntimeError(f'No `metadata` associated with `{molecule.smiles}`')

    if key not in molecule.metadata:
        raise RuntimeError(f'`{key}` not found in `metadata`')

    meta = molecule.metadata[key]
    if isinstance(meta, Iterable):
        return meta
    else:
        return [meta]
    

def _collate_assayedmolecule_metadata(molecule, key, assay: Union[None, str] = None):
    """ Batches metadata from AssayedMolecule.
    Parameters
    ----------
    molecule : malt.Molecule
    assay : Union[None, str]
        Filter metadata using assay key.
    key : str
        Attribute of class on which to batch.
    Returns
    -------
    meta : list
        Elements are metadata from assay.
    """
    meta = []
    for record in molecule[assay]:
        if key not in record:
            raise RuntimeError(f'`{key}` not found in `metadata`')
        meta.append(record[key])
    return meta
