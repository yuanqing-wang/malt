# =============================================================================
# IMPORTS
# =============================================================================
import abc
from typing import Union
from .agent import Agent
from .assayer import Assayer
from .letters import QueryReceipt, OrderReceipt, Quote, MerchantToAssayerNote

# =============================================================================
# BASE CLASSES
# =============================================================================
class Merchant(Agent):
    """ Base class for a merchant that sells small molecules.

    Methods
    -------
    quote
        Offer a quote for the small molecule.

    """
    def __init__(self):
        super(Merchant, self).__init__()

    @abc.abstractmethod
    def quote(self, molecules: list) -> Quote:
        """ Generate a quote for a molecule with a SMILES string.

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
    def sale(
        self, quote: Quote, assayer: Assayer, *args, **kwargs
    ) -> MerchantToAssayerNote:
        """ Execute with a quote and a downstream assayer.

        Parameters
        ----------
        quote : Quote
            Quote that was generated for the molecules.
        assayer : Assayer
            Downstream assayer.

        Returns
        -------
        MerchantToAssayerNote
            Note from merchant to assayer.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def catalogue(self) -> callable:
        """ Offer a catalogue of all the available function.

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
        ) -> Union[None, MerchantToAssayerNote]:
        raise NotImplementedError

    def check(self, receipt):
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)
