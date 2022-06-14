# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import malt
import torch
from malt.molecule import Molecule
from malt.data.utils import collate_metadata
from typing import Union, Iterable, Optional, List, Any

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

    def __init__(self, molecules: Optional[List]=None) -> None:
        super(Dataset, self).__init__()
        if molecules is None:
            molecules = []
        assert isinstance(molecules, List)
        assert all(isinstance(molecule, Molecule) for molecule in molecules)
        self.molecules = molecules

    def __repr__(self):
        return "%s with %s molecules" % (self.__class__.__name__, len(self))

    def _construct_lookup(self):
        """Construct lookup table for molecules."""
        self._lookup = {mol.smiles: mol for mol in self.molecules}

    @property
    def lookup(self):
        """Returns the mapping between the SMILES and the molecule. """
        if self._lookup is None:
            self._construct_lookup()
        return self._lookup

    def __contains__(self, molecule):
        """Check if a molecule is in the dataset.

        Parameters
        ----------
        molecule : malt.Molecule

        Examples
        --------
        >>> molecule = Molecule("CC")
        >>> dataset = Dataset([molecule])
        >>> Molecule("CC") in dataset
        True
        >>> Molecule("C") in dataset
        False

        """
        return molecule.smiles in self.lookup

    def apply(self, function):
        """Apply a function to all molecules in the dataset.

        Parameters
        ----------
        function : Callable
            The function to be applied to all molecules in this dataset
            in place.

        Examples
        --------
        >>> molecule = Molecule("CC")
        >>> dataset = Dataset([molecule])
        >>> from ..molecule import AssayedMolecule
        >>> fn = lambda molecule: AssayedMolecule(
        ...     smiles=molecule.smiles, metadata={"name": "john"},
        ... )
        >>> dataset = dataset.apply(fn)
        >>> dataset[0]["name"]
        'john'
        """

        self.molecules = [function(molecule) for molecule in self.molecules]
        return self

    def __eq__(self, other):
        """Determin if two objects are identical."""
        if not isinstance(other, self.__class__):
            return False
        return self.molecules == other.molecules

    def __len__(self):
        """Return the number of molecules in the dataset."""
        if self.molecules is None:
            return 0
        return len(self.molecules)

    def __getitem__(self, key: Any):
        """Get item from the dataset.

        Parameters
        ----------
        key : Any

        Notes
        -----
        * If the key is integer, return the single molecule indexed.
        * If the key is a string, return a dataset of all molecules with
            this SMILES.
        * If the key is a molecule, extract the SMILES string and index by
            its SMILES.
        * If the key is a tensor, flatten it to treat it as a list.
        * If the key is a list, return a dataset with molecules indexed by
            the elements in the list.
        * If the key is a slice, slice the range and treat at as a list.

        """
        if self.molecules is None:
            raise RuntimeError("Empty Portfolio.")
        if isinstance(key, int):
            return self.molecules[key]
        elif isinstance(key, str): # NOTE(yuanqing-wang): Are we settled?
            return self.__class__(molecules=[self.lookup[key]])
        elif isinstance(key, Molecule):
            return self.lookup[key.smiles]
        elif isinstance(key, torch.Tensor):
            key = key.detach().flatten().cpu().numpy().tolist()
        elif isinstance(key, list):
            return self.__class__(
                molecules=[self.molecules[_idx] for _idx in key]
            )
        elif isinstance(key, slice):
            return self.__class__(molecules=self.molecules[key])
        else:
            raise RuntimeError("The slice is not recognized.")

    def shuffle(self, seed=None):
        """ Shuffle the dataset and return it. """
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

        Returns
        -------
        List[Dataset]
            List of datasets split according to the partition.

        Examples
        --------
        >>> dataset = Dataset([Molecule("CC"), Molecule("C")])
        >>> dataset0, dataset1 = dataset.split([1, 1])
        >>> dataset0[0].smiles
        'CC'
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
        """Combine two datasets and return a new one.

        Parameters
        ----------
        molecules : Union[List[Molecule], Dataset]
            Molecules to be added to the dataset.

        Returns
        -------
        >>> dataset0 = Dataset([Molecule("C")])
        >>> dataset1 = Dataset([Molecule("CC")])
        >>> dataset = dataset0 + dataset1
        >>> len(dataset)
        2

        """
        if isinstance(molecules, list):
            return self.__class__(molecules=self.molecules + molecules)

        elif isinstance(molecules, Dataset):
            return self.__class__(
                molecules=self.molecules + molecules.molecules
            )

        else:
            raise RuntimeError("Addition only supports list and Dataset.")

    def __sub__(self, molecules):
        """ Subtract a list of molecules from a dataset and return a new one.

        Parameters
        ----------
        molecules : Union[list[Molecule], Dataset]
            Molecules to be subtracted from the dataset.

        Returns
        -------
        Dataset
            The resulting dataset.

        Examples
        --------
        >>> dataset = Dataset([Molecule("CC"), Molecule("C")])
        >>> dataset -= [Molecule("C")]
        >>> len(dataset)
        1
        """
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
        """Alias of iter for molecules. """
        return iter(self.molecules)

    def append(self, molecule):
        """Append a molecule to the dataset.

        Alias of append for molecules.

        Note
        ----
        * This append in-place.

        Parameters
        ----------
        molecule : molecule
            The data molecule to be appended.

        """
        self.molecules.append(molecule)
        return self

    def featurize_all(self):
        """ Featurize all molecules in dataset. """
        (molecule.featurize() for molecule in self.molecules())
        return self

    @property
    def smiles(self):
        """Return the list of SMILE strings in the datset. """
        return [molecule.smiles for molecule in self.molecules]

    @staticmethod
    def _batch(
        molecules=None, by=['g', 'y'], assay=None,
        batch_meta=collate_metadata, device='cuda',
        **kwargs,
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
                ret[key] = torch.tensor(ret[key])[:,None]
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
