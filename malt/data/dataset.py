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
    """ A collection of Points with functionalities to be compatible with
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
    def __init__(self, points: Union[None, List[Point]]=None) -> None:
        self.points = points

    def __len__(self):
        if self.points is None:
            return 0
        return len(self.points)

    def __getitem__(self, idx):
        if self.points is None:
            raise RuntimeError("Empty Portfolio.")
        if isinstance(idx, int):
            return self.points[idx]
        else:
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
            if not point.is_featurized(): # featurize
                point.featurize()
            if point.y is None:
                raise RuntimeError("No data associated with data. ")
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
