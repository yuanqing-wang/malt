# =============================================================================
# IMPORTS
# =============================================================================
import abc
from typing import Union, Callable
from .agent import Agent
from .assayer import Assayer
from .messages import QueryReceipt, OrderReceipt, Quote, VendorToAssayerNote
from ..data.dataset import Dataset

# =============================================================================
# BASE CLASSES
# =============================================================================
class Vendor(Agent):
    """Base class for a vendor that sells small molecules.

    Methods
    -------
    quote
        Offer a quote for the small molecule.

    """

    def __init__(self, *args, **kwargs):
        super(Vendor, self).__init__(*args, **kwargs)

    @abc.abstractmethod
    def query(self, points: Dataset) -> Quote:
        """Generate a quote for a molecule with a SMILES string.

        Parameters
        ----------
        molecules : list
            List of molecules to be quoted.

        Returns
        -------
        Quote
            Quote for the molecule.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def order(
        self, quote: Quote, assayer: Assayer, *args, **kwargs
    ) -> VendorToAssayerNote:
        """Execute with a quote and a downstream assayer.

        Parameters
        ----------
        quote : Quote
            Quote that was generated for the molecules.
        assayer : Assayer
            Downstream assayer.

        Returns
        -------
        VendorToAssayerNote
            Note from vendor to assayer.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def catalogue(self) -> Callable:
        """Offer a catalogue of all the available function.

        Returns
        -------
        callable
            A function that generate the catalogue.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _check_query(self, query_receipt: QueryReceipt) -> Union[None, Quote]:
        raise NotImplementedError

    @abc.abstractmethod
    def _check_order(
        self, order_receipt: OrderReceipt
    ) -> Union[None, VendorToAssayerNote]:
        raise NotImplementedError

    def check(self, receipt):
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)
