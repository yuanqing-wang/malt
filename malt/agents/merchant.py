# =============================================================================
# IMPORTS
# =============================================================================
import abc
from .agent import Agent
from .letters import QueryReceipt, OrderReceipt, Quote, MerchantToAssayerNote

# =============================================================================
# BASE CLASSES
# =============================================================================
class Merchant(Agent, abc.ABC):
    """ Base class for a merchant that sells small molecules.

    Methods
    -------
    quote
        Offer a quote for the small molecule.

    """
    def __init__(self):
        super(Merchant, self).__init__()

    @abc.abstractmethod
    def quote(self, molecules: list, *args, **kwargs) -> Quote:
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
        self, molecules: list, assayer: Agent, *args, **kwargs
    ) -> MerchantToAssayerNote:
        """ Execute an order with a molecule with a SMILES string and a
        downstream assayer.

        Parameters
        ----------
        molecules : list
            List of molecules to be sold.

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
