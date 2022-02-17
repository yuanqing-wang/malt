# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, Any
import functools
import dgl
import torch
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

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

    metadata : Any
        Metadata associated with the molecule.

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
        metadata: Any = None,
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

    def erase_annotation(self):
        self.metadata = None
        return self


class AssayedMolecule(Molecule):
    """ Models information associated with a molecule.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule.

    g : dgl.DGLGraph or None, default=None
        The DGL graph of the molecule.

    metadata : dict, default={}
        Metadata from assays associated with the molecule.

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

        super(AssayedMolecule, self).__init__(
            smiles = smiles,
            g = g,
            metadata = metadata,
            featurizer = featurizer
        )

    def __eq__(self, other):
            return (
                self.g == other.g
                and self.metadata == other.metadata
            )

    def __getitem__(self, idx):
        if not self.metadata:
            raise RuntimeError("No data associated with Molecule.")
        elif isinstance(idx, str):
            return self.metadata[idx]
        elif idx is None and len(self.metadata) == 1:
            return list(self.metadata.values())[0]
        else:
            raise NotImplementedError

    def __contains__(self, key):
        if not self.metadata:
            raise RuntimeError("No data associated with Molecule.")
        return key in self.metadata

    def __add__(self, other):
        if self.smiles != other.smiles:
            raise RuntimeError(
                f'SMILES must match; `{other.smiles}` != `{self.smiles}`.'
            )
        else:
            for key in other.metadata:
                if key not in self.metadata:
                    self.metadata[key] = other.metadata[key]
                else:
                    self.metadata[key] += other.metadata[key]
        return self

    def erase_annotation(self):
        self.metadata = {}
        return self