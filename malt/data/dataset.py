# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
from typing import Union, List
from ..point import Point

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(torch.utils.data.Dataset):
    """A collection of Points with functionalities to be compatible with
    training and optimization.

    Parameters
    ----------
    points : List[Point]
        A list of points.

    Methods
    -------
    featurize(points)
        Featurize all points in the dataset.
    view()
        Generate a torch.utils.data.DataLoader from this Dataset.

    """
    _lookup = None

    def __init__(self, points) -> None:
        super(Dataset, self).__init__()
        self.points = points

    def _construct_lookup(self):
        self._lookup = {point.smiles: point for point in self.points}

    @property
    def lookup(self):
        if self._lookup is None:
            self._construct_lookup()
        return self._lookup

    def equal(self, points):
        if not isinstance(points, self.__class__):
            return False
        return self.points == points.points

    def __len__(self):
        if self.points is None:
            return 0
        return len(self.points)

    def __getitem__(self, idx):
        if self.points is None:
            raise RuntimeError("Empty Portfolio.")
        if isinstance(idx, int):
            return self.points[idx]
        elif isinstance(idx, str):
            return self.lookup[idx]
        elif isinstance(idx, torch.Tensor):
            idx = idx.detach().flatten().cpu().numpy().tolist()

        if isinstance(idx, list):
            return self.__class__(points=[
                self.points[_idx] for _idx in idx
            ])

        return self.__class__(points=self.points[idx])

    def __add__(self, points):
        """ Combine two datasets. """
        if isinstance(points, list):
            return self.__class__(points=self.points + points)

        elif isinstance(points, Dataset):
            return self.__class__(points=self.points + points.points)

        else:
            raise RuntimeError("Addition only supports list and Dataset.")

    def __sub__(self, points):
        if isinstance(points, list):
            points = self.__class__(points)
        return self.__class__(
            [
                point
                for point in self.points if point.smiles not in points.lookup
            ]
        )

    def __iter__(self):
        return iter(self.points)

    def append(self, point):
        """Append a point to the dataset.

        Parameters
        ----------
        point : Point
            The data point to be appended.
        """
        self.points.append(point)
        return self

    def featurize_all(self):
        """ Featurize all points in dataset. """
        for point in self.points:
            if not point.is_featurized():
                point.featurize()

        return self

    @staticmethod
    def batch_of_g_and_y(points):
        # initialize results
        gs = []
        ys = []

        # loop through the points
        for point in points:
            if not point.is_featurized():  # featurize
                point.featurize()
            if point.y is None:
                raise RuntimeError("No data associated with data. ")
            gs.append(point.g)
            ys.append(point.y)

        g = dgl.batch(gs)
        y = torch.tensor(ys)[:, None]
        return g, y

    @staticmethod
    def batch_of_g(points):
        # initialize results
        gs = []

        # loop through the points
        for point in points:
            if not point.is_featurized():  # featurize
                point.featurize()
            gs.append(point.g)

        g = dgl.batch(gs)
        return g

    def view(
        self,
        collate_fn: Union[callable, str] = "batch_of_g_and_y",
        *args,
        **kwargs
    ):
        """Provide a data loader from portfolio.

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
        if isinstance(collate_fn, str):
            collate_fn = getattr(self, collate_fn)

        return torch.utils.data.DataLoader(
            dataset=self,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )
