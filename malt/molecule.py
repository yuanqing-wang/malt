# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, Any, Optional, Mapping, Callable, Sequence
import functools
import dgl
import copy
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

    Notes
    -----
    * The current CanonicalAtomFeaturizer has implicit Hs.

    Examples
    --------
    >>> molecule = Molecule("C")
    >>> molecule.g.number_of_nodes()
    1

    """
    def __init__(
        self,
        smiles: str,
        g: Optional[dgl.DGLGraph] = None,
        metadata: Optional[Mapping] = None,
        featurizer: Optional[Callable] = functools.partial(
            smiles_to_bigraph,
            node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        ),
    ) -> None:
        self.smiles = smiles
        self.g = g
        self.metadata = metadata
        self.featurizer = featurizer

        # featurize the first thing after init
        self.featurize()

    def __repr__(self) -> str:
        return self.smiles

    def featurize(self) -> None:
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

    def is_featurized(self) -> bool:
        """Returns whether this molecule is attached with a graph. """
        return self.g is not None

    def __eq__(self, other: Any):
        """Determine if two AssayedMolecule objects are equal.

        Parameters
        ----------
        other : Any
            The other object

        Returns
        -------
        bool
            If the two objects are identical.

        Examples
        --------
        >>> molecule = AssayedMolecule("C", metadata={"name": "john"})

        Type mismatch:
        >>> molecule == "john"
        False

        Graph mismatch:
        >>> molecule == AssayedMolecule("CC", metadata={"name": "john"})
        False

        Metadata mismatch:
        >>> molecule == AssayedMolecule("C", metadata={"name": "jane"})
        False

        Both graph and metadata match:
        >>> molecule == AssayedMolecule("C", metadata={"name": "john"})
        True

        """
        # if not a molecule, fuggedaboutit
        if not isinstance(other, type(self)):
            return False

        # NOTE(yuanqing-wang):
        # Equality is not well-defined for DGL graph
        # Use networx isomorphism instead.
        import networkx as nx
        return (
            nx.is_isomorphic(self.g.to_networkx(), other.g.to_networkx())
            and self.metadata == other.metadata
        )

    def erase_annotation(self) -> Any:
        """Erase the metadata. """
        self.metadata = None
        return self
