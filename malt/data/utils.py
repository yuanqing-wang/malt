# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def batch_metadata(molecule, assay, key):
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