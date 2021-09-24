# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union
import functools
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Point(object):
    """ Models a datapoint.

    Parameters
    ----------
    smiles : str
        SMILES of the molecule.

    g : dgl.DGLGraph or None, default=None
        The DGL graph of the molecule.

    y : float or None, default=None
        The measurement.

    extra : dict, default={}
        The extra data.
        
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
        y: Union[float, None] = None,
        extra: dict = {},
        featurizer: callable = functools.partial(
            smiles_to_bigraph,
            node_featurizer=CanonicalAtomFeaturizer(atom_data_field="h"),
        ),
    ) -> None:
        self.smiles = smiles
        self.g = g
        self.y = y
        self.extra = extra
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
        self.y = None
        return self
