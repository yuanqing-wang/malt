import abc
from .agent import Agent
from malt.data.dataset import Dataset

class Assayer(Agent):
    """Assayer. Takes a point and annotates `y` field as well as (optionally)
    extra.

    Methods
    -------
    assay(dataset)
        Annotate `y` in the dataset.

    """
    def __init__(self):
        super(Assayer, self).__init__()

    @abc.abstractmethod
    def assay(self, *args, **kwargs):
        raise NotImplementedError

class DatasetAssayer(Assayer):
    """Simulated assayer based on dataset.

    Parameters
    ----------
    dataset : Dataset

    Methods
    -------
    assay(dataset)
        Assay a dataset (persumably without `y`).

    Examples
    --------
    >>> import malt
    >>> dataset = malt.data.collections.linear_alkanes(5)
    >>> dataset_assayer = malt.agents.assayer.DatasetAssayer(dataset)
    >>> assayed_dataset = dataset_assayer.assay(dataset)
    >>> assert assayed_dataset == dataset

    """

    def __init__(self, dataset: Dataset):
        super(DatasetAssayer, self).__init__()
        self.dataset = dataset

    def assay(self, dataset, by=['y']):
        """ Assay based on a given dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset to assay.

        Returns
        -------
        dataset : Dataset
            Assayed dataset, with all `y` annotated.

        """
        for molecule in dataset:
            
            assert molecule in self.dataset
            
            if molecule.metadata is None:
                molecule.metadata = {}
            
            for key in by:
                molecule.metadata[key] = self.dataset[molecule].metadata[key]
        
        return dataset

class DockingAssayer(Assayer):
    """Simulated assayer based on docking score.

    Parameters
    ----------
    protein : str
        PDB or file name of a protein.

    Methods
    -------
    assay(dataset)
        Assay a dataset (persumably without `y`).

    """
    def __init__(self, protein: str):
        super(DockingAssayer, self).__init__()
        self.protein = protein

    def dock(self, point):
        from malt.utils.docking import vina_docking
        return vina_docking(smiles=point.smiles, protein=self.protein)

    def assay(self, dataset):
        for point in dataset:
            point.y = self.dock(point)
        return dataset
