# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
from typing import Union, Iterable
from malt.point import Point

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(torch.utils.data.Dataset):
    """A collection of Molecules with functionalities to be compatible with
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
    _extra = None

    def __init__(self, points=[]) -> None:
        super(Dataset, self).__init__()
        assert all(isinstance(point, Point) for point in points)
        self.points = points

    def __repr__(self):
        return "%s with %s points" % (self.__class__.__name__, len(self))

    def _construct_lookup(self):
        from collections import defaultdict
        self._lookup = defaultdict(list)
        for point in self.points:
            self._lookup[point.smiles].append(point)
        self._lookup = dict(self._lookup)

    @property
    def lookup(self):
        if self._lookup is None:
            self._construct_lookup()
        return self._lookup

    def __contains__(self, point):
        return point.smiles in self.lookup

    def apply(self, function):
        self.points = [function(point) for point in self.points]
        return self

    def __eq__(self, points):
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
            return self.__class__(points=self.lookup[idx])
        elif isinstance(idx, Point):
            return self.lookup[idx.smiles]
        elif isinstance(idx, torch.Tensor):
            idx = idx.detach().flatten().cpu().numpy().tolist()

        if isinstance(idx, list):
            return self.__class__(points=[self.points[_idx] for _idx in idx])

        elif isinstance(idx, slice):
            return self.__class__(points=self.points[idx])

        else:
            raise RuntimeError("The slice is not recognized.")

        return self.__class__(points=self.points[idx])

    def shuffle(self, seed=None):
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.points)
        return self


    def split(self, partition):
        """Split the dataset according to some partition.
        Parameters
        ----------
        partition : sequence of integers or floats
        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        ds = []
        idx = 0
        for p_size in partition:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds

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
                for point in self.points
                if point.smiles not in points.lookup
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

    @property
    def y(self):
        return [point.y for point in self.points]

    @property
    def smiles(self):
        return [point.smiles for point in self.points]

    def _construct_extra(self):
        from collections import defaultdict
        self._extra = defaultdict(list)
        for point in self.points:
            for key, value in point.extra.items():
                self._extra[key].append(value)
        self._extra = dict(self._extra)

    @property
    def extra(self):
        if self._extra is None:
            self._construct_extra()
        return self._extra

    def batch(self, points=None, by=['g', 'y']):
        """Batches points by provided keys.

        Parameters
        ----------
        points : list of Points
            Defaults to all points in Dataset if none provided.
        by : Union[Iterable, str]
            Attributes of class on which to batch.

        Returns
        -------
        ret : Union[tuple, dgl.Graph, torch.Tensor]
            Batched data, in order of keys passed in `by` argument.

        """
        
        from collections import defaultdict
        ret = defaultdict(list)

        if points is None:
            points = self.points

        # guarantee keys are a list
        by = [by] if isinstance(by, str) else by

        # loop through points
        for point in points:
            for key in by:
                if key is 'g':
                    # featurize graphs
                    if not point.is_featurized():
                        point.featurize()
                    ret['g'].append(point.g)
                
                elif key is 'y':
                    if point.y is None:
                        raise RuntimeError(
                            'No targets associated with data.'
                        )
                    ret['y'].append(point.y)
                
                else:
                    if key not in point.extra:
                        raise RuntimeError(f'`{key}` not found in `extra`')
                    ret[key].append(point.extra[key])

        # collate batches
        for key in by:
            if key == 'g':
                ret['g'] = dgl.batch(ret['g'])
            else:
                ret[key] = torch.tensor(ret[key])[:,None]
        
        # return batches
        ret = (*ret.values(), )
        if len(ret) < 2:
            ret = ret[0]
        
        return ret


    def batch_all_g(self):
        return next(
            iter(
                self.view(
                    by='g',
                    batch_size=len(self)
                )
            )
        )

    def erase_annotation(self):
        for point in self.points:
            point.erase_annotation()
        return self

    def clone(self):
        """ Return a copy of self. """
        import copy
        return self.__class__(copy.deepcopy(self.points))

    def view(
        self,
        collate_fn: Union[callable, str] = batch,
        group: Union[None, str] = None,
        by: Union[Iterable, str] = ['g', 'y'],
        *args,
        **kwargs,
    ):
        """Provide a data loader from portfolio.

        Parameters
        ----------
        collate_fn : None or callable
            The function to gather data points.
        group : Union[None, str]
            If a group is provided (e.g., 'smiles'), batches data by SMILES groupings.
        by : Union[Iterable, str]


        Returns
        -------
        torch.utils.data.DataLoader
            Resulting data loader.

        """
        from functools import partial
        
        def _get_smiles_batch_indices():
            import numpy as np
            cumul_data = np.cumsum([
                len(v) for v in self.lookup.values()
            ])
            return np.split(
                np.arange(len(self)),
                indices_or_sections=cumul_data[:-1],
            )

        # provide default collate function
        collate_fn = self.batch

        if group == 'smiles':
            batch_sampler = _get_smiles_batch_indices()
        elif group is None:
            batch_sampler = None

        return torch.utils.data.DataLoader(
            dataset=self.points,
            collate_fn=partial(collate_fn, by=by),
            batch_sampler=batch_sampler,
            *args,
            **kwargs,
        )