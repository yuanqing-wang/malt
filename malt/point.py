# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
from typing import Union, List
from dgllife.utils import CanonicalAtomFeaturizer

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
            featurizer: callable = CanonicalAtomFeaturizer(atom_data_field="h"),
        ) -> None:
        self.smiles = smiles
        self.g = g
        self.y = y
        self.extra = extra
        self.featurizer = featurizer

    def featurize(self):
        """ Featurize the SMILES string to get the graph.

        Returns
        -------
        dgl.DGLGraph : The resulting graph.

        """
        # if there is already a graph, raise an error
        if self.is_featurized():
            raise RuntimeError("Point is already featurized.")

        # featurize
        self.g = self.featurizer(self.smiles)

        return self.g

    def is_featurized(self):
        return (self.g is not None)

class Portfolio(torch.utils.data.Dataset):
    """ A collection of Points with functionalities to be compatible with
    training and optimization.

    Parameters
    ----------
    points : List[Point]
        A list of points in the portfolio.

    Methods
    -------

    """
    def __init__(self, points: Union[None, List[Point]]=None) -> None:
        self.points = points

    def __len__(self):
        if self.points is None:
            return 0
        return len(self.points)

    def __getitem__(self, idx):
        if self.points is None:
            raise RuntimeError("Empty Portfolio.")
        return self.__class__(points=self.points[idx])

    @staticmethod
    def featurize(points):
        for point in points:
            if not point.is_featurized():
                point.featurize()
        return points

    @staticmethod
    def _batch_as_tuple_of_g_and_y(points):
        # initialize results
        gs = []
        ys = []

        # loop through the points
        for point in points:
            if not point.is_featurized():
                point.featurize()
            gs.append(point.g)
            ys.append(point.y)

        g = dgl.batch(gs)
        y = torch.tensor(ys)[:, None]
        return g, y

    def view(self, collate_fn: Union[callable, None]=None, *args, **kwargs):
        """ Provide a data loader from portfolio.

        Parameters
        ----------
        collate_fn : None or callable
            The function to gather data points.

        Returns
        -------
        torch.utils.data.DataLoader
            Resulting data loader.

        """
        # provide default collate function
        if collate_fn is None:
            collate_fn = self._batch_as_tuple_of_g_and_y

        return torch.utils.data.DataLoader(
            dataset=self,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )
