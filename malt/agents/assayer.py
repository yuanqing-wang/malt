# =============================================================================
# IMPORTS
# =============================================================================
import abc
from .agent import Agent
from .messages import (
    QueryReceipt,
    OrderReceipt,
    Quote,
    MerchantToAssayerNote,
    Dataset,
)

# =============================================================================
# BASE CLASSES
# =============================================================================
class Assayer(Agent, abc.ABC):
    """ Models an assayer. """

    def __init__(self, *args, **kwargs):
        super(Assayer, self).__init__(*args, **kwargs)

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
        self,
        quote: Quote,
    ) -> OrderReceipt:
        """Conduct a sale to assay a list of compounds."""
        raise NotImplementedError

    def receive_note(
        self,
        merchant_to_assayer_note: MerchantToAssayerNote,
    ):
        pass

    @abc.abstractmethod
    def _check_query(self, query_receipt):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_order(self, order_receipt):
        raise NotImplementedError

    def check(self, receipt):
        """ Check the status of an order or a query given a receipt. """
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)
