import abc
from malt.data.dataset import Dataset
from .agent import Agent

class Merchant(Agent):
    """ Base class for all merchants.

    Methods
    -------
    catalogue (*args, **kwargs):
        Return an iterator over the catalogue of the merchant.

    merchandize (*args, **kwargs):
        Place an order from the merchant.

    """

    def __init__(self):
        super(Merchant, self).__init__()

    @abc.abstractmethod
    def catalogue(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def merchandize(self, *args, **kwargs):
        raise NotImplementedError

class DatasetMerchant(Merchant):
    """ Merchant with a candidate pool.

    Parameters
    ----------
    dataset : Dataset
        A dataset of Points.

    Examples
    --------
    >>> import malt
    >>> dataset = malt.data.collections.linear_alkanes(5)
    >>> dataset_merchant = malt.agents.merchant.DatasetMerchant(dataset)
    >>> catalogue = dataset_merchant.catalogue()
    >>> assert catalogue == dataset.clone().erase_annotation()
    >>> dataset_with_the_first_point = Dataset([dataset[0]])
    >>> dataset_merchant.merchandize(dataset_with_the_first_point)
    Dataset with 1 points
    >>> assert len(dataset_merchant.dataset) == len(dataset) - 1

    """
    def __init__(
        self,
        dataset: Dataset,
    ):
        super(DatasetMerchant, self).__init__()
        self.dataset = dataset.clone().erase_annotation()

    def catalogue(self):
        return self.dataset

    def merchandize(self, dataset):
        """ Order molecules in subset.

        Parameters
        ----------
        dataset : malt.Dataset
            A dataset to be merchandized.

        """
        self.dataset -= dataset
        return dataset
