# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import malt
import torch
from malt.molecule import Molecule
from malt.data.utils import collate_metadata
from typing import Union, Iterable

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(torch.utils.data.Dataset):
    """A collection of Molecules with functionalities to be compatible with
    training and optimization.
    Parameters
    ----------
    molecules : List[malt.Molecule]
        A list of Molecules.
    Methods
    -------
    featurize(molecules)
        Featurize all molecules in the dataset.
    view()
        Generate a torch.utils.data.DataLoader from this Dataset.
    """

    _lookup = None
    _extra = None

    def __init__(self, molecules=[]) -> None:
        super(Dataset, self).__init__()
        assert all(isinstance(molecule, Molecule) for molecule in molecules)
        self.molecules = molecules

    def __repr__(self):
        return "%s with %s molecules" % (self.__class__.__name__, len(self))

    def _construct_lookup(self):
        self._lookup = {mol.smiles: mol for mol in self.molecules}

    @property
    def lookup(self):
        if self._lookup is None:
            self._construct_lookup()
        return self._lookup

    def __contains__(self, molecule):
        return molecule.smiles in self.lookup

    def apply(self, function):
        self.molecules = [function(molecule) for molecule in self.molecules]
        return self

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.molecules == other.molecules

    def __len__(self):
        if self.molecules is None:
            return 0
        return len(self.molecules)

    def __getitem__(self, idx):
        if self.molecules is None:
            raise RuntimeError("Empty Portfolio.")
        if isinstance(idx, int):
            return self.molecules[idx]
        elif isinstance(idx, str):
            return self.__class__(molecules=self.lookup[idx])
        elif isinstance(idx, Molecule):
            return self.lookup[idx.smiles]
        elif isinstance(idx, torch.Tensor):
            idx = idx.detach().flatten().cpu().numpy().tolist()
        if isinstance(idx, list):
            return self.__class__(molecules=[self.molecules[_idx] for _idx in idx])
        elif isinstance(idx, slice):
            return self.__class__(molecules=self.molecules[idx])
        else:
            raise RuntimeError("The slice is not recognized.")

        return self.__class__(molecules=self.molecules[idx])

    def shuffle(self, seed=None):
        import random
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.molecules)
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

    def __add__(self, molecules):
        """ Combine two datasets. """
        if isinstance(molecules, list):
            return self.__class__(molecules=self.molecules + molecules)

        elif isinstance(molecules, Dataset):
            return self.__class__(molecules=self.molecules + molecules.molecules)

        else:
            raise RuntimeError("Addition only supports list and Dataset.")

    def __sub__(self, molecules):
        if isinstance(molecules, list):
            molecules = self.__class__(molecules)

        return self.__class__(
            [
                molecule
                for molecule in self.molecules
                if molecule.smiles not in molecules.lookup
            ]
        )

    def __iter__(self):
        return iter(self.molecules)

    def append(self, molecule):
        """Append a molecule to the dataset.
        Parameters
        ----------
        molecule : molecule
            The data molecule to be appended.
        """
        self.molecules.append(molecule)
        return self

    def featurize_all(self):
        """ Featurize all molecules in dataset. """
        for molecule in self.molecules:
            if not molecule.is_featurized():
                molecule.featurize()

        return self

    @property
    def smiles(self):
        return [molecule.smiles for molecule in self.molecules]

    @staticmethod
    def _batch(
        molecules=None, by=['g', 'y'], assay=None,
        batch_meta=collate_metadata, use_gpu=True,
        **kwargs
    ):
        """ Batches molecules by provided keys.
        Parameters
        ----------
        molecules : list of molecules
            Defaults to all molecules in Dataset if none provided.
        assay : Union[None, str]
            Filter metadata using assay key.
        by : Union[Iterable, str]
            Attributes of molecule on which to batch.
        Returns
        -------
        ret : Union[tuple, dgl.Graph, torch.Tensor]
            Batched data, in order of keys passed in `by` argument.
        """
        from collections import defaultdict
        ret = defaultdict(list)

        # guarantee keys are a list
        by = [by] if isinstance(by, str) else by

        # loop through molecules
        for molecule in molecules:
            
            for key in by:
                if key == 'g':
                    # featurize graphs
                    if not molecule.is_featurized():
                        molecule.featurize()
                    ret['g'].append(molecule.g)

                else:
                    m = batch_meta(molecule, key, assay=assay)
                    ret[key].extend(m)

        # collate batches
        for key in by:
            if key == 'g':
                ret['g'] = dgl.batch(ret['g'])
            else:
                ret[key] = torch.tensor(ret[key])
            if torch.cuda.is_available():
                ret[key] = ret[key].to(torch.cuda.current_device())
        
        # return batches
        ret = (*ret.values(), )
        if len(ret) < 2:
            ret = ret[0]
        
        return ret


    def batch(self, **kwargs):
        return self._batch(molecules=self.molecules, **kwargs)

    def erase_annotation(self):
        for molecule in self.molecules:
            molecule.erase_annotation()
        return self

    def clone(self):
        """ Return a copy of self. """
        import copy
        return self.__class__(copy.deepcopy(self.molecules))

    def view(
        self,
        collate_fn: Union[callable, str] = batch,
        assay: Union[None, str] = None,
        by: Union[Iterable, str] = ['g', 'y'],
        batch_meta: callable = collate_metadata,
        *args,
        **kwargs,
    ):
        """Provide a data loader from portfolio.
        Parameters
        ----------
        collate_fn : None or callable
            The function to gather data molecules.
        assay : Union[None, str]
            Batch data from molecules using key provided to filter metadata.
        by : Union[Iterable, str]
        Returns
        -------
        torch.utils.data.DataLoader
            Resulting data loader.
        """
        from functools import partial
        
        # provide default collate function
        collate_fn = self._batch

        return torch.utils.data.DataLoader(
            dataset=self.molecules,
            collate_fn=partial(
                collate_fn,
                by=by,
                assay=assay,
                batch_meta=batch_meta
            ),
            *args,
            **kwargs,
        )
