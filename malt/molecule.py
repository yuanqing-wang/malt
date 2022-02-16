# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union
import functools
import dgl
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def validate_metadata(metadata):
    if not metadata:
        return metadata
    else:
        metadata_values = iter(metadata.values())
        length = len(next(metadata_values))
        if all(len(v) == length for v in metadata_values):
            return metadata
        else:
            raise RuntimeError('Length of values in `metadata` do not match.')


# =============================================================================
# MODULE CLASSES
# =============================================================================
class Molecule(object):
    """ Models information associated with a molecule.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule.

    g : dgl.DGLGraph or None, default=None
        The DGL graph of the molecule.

    metadata : dict, default={}
        Metadata associated with the compound

    featurizer : callable, default=CanonicalAtomFeaturizer(
        atom_data_field='feat')
        The function which maps the SMILES string to a DGL graph.

    Methods
    -------
    featurize()
        Convert the SMILES string to a graph if there isn't one.

    """

    def __init__(
        self,
        smiles: str,
        g: Union[dgl.DGLGraph, None] = None,
        metadata: dict = {},
        featurizer: callable = functools.partial(
            smiles_to_bigraph,
            node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        ),
    ) -> None:
        self.smiles = smiles
        self.g = g
        self.metadata = metadata
        self.featurizer = featurizer

    def __repr__(self):
        return self.smiles

    def __eq__(self, other):
            return (
                self.g == other.g
                and self.metadata == other.metadata
            )
    
    def __getattr__(self, name):
        if name not in self.metadata:
            raise RuntimeError(
                f'`{name}` is not associated with this Molecule.'
            )
        return self.metadata[name]
    
    def __add__(self, other):
        if self.smiles != other.smiles:
            raise RuntimeError(
                f'SMILES must match; `{other.smiles}` != `{self.smiles}`.'
            )
        else:
            for key in other.metadata:
                if key not in self.metadata:
                    raise RuntimeError(
                        f'All keys must match; `{key}` not associated with "{self.smiles}"'
                    )
                else:
                    self.metadata[key] += other.metadata[key]
        return self
        
    def __getitem__(self, idx):
        if not self.metadata:
            raise RuntimeError("No data associated with Molecule.")
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().flatten().cpu().numpy().tolist()
        if isinstance(idx, list):
            metadata = {k: list(map(v.__getitem__, idx))
                        for k, v in self.metadata.items()}
        if isinstance(idx, int):
            metadata = {k: [v[idx]] for k, v in self.metadata.items()}        
        elif isinstance(idx, slice):
            metadata = {k: v[idx] for k, v in self.metadata.items()}
        else:
            raise RuntimeError("The slice is not recognized.")
        return self.__class__(
                smiles = self.smiles,
                g = self.g,
                metadata = metadata
        )

    def featurize(self):
        """Featurize the SMILES string to get the graph.

        Returns
        -------
        dgl.DGLGraph : The resulting graph.

        """
        # if there is already a graph, raise an error
        if self.is_featurized():
            raise RuntimeError("Point is already featurized.")

        # featurize
        self.g = self.featurizer(self.smiles)

        return self

    def is_featurized(self):
        return self.g is not None