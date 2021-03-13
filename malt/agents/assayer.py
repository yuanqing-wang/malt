# =============================================================================
# IMPORTS
# =============================================================================
import abc
from .agent import Agent
from .messages import QueryReceipt, OrderReceipt, Quote, MerchantToAssayerNote

# =============================================================================
# BASE CLASSES
# =============================================================================
class Assayer(Agent, abc.ABC):
    """ Models an assayer. """
    def __init__(self):
        super(Assayer, self).__init__()

    @abc.abstractmethod
    def sale(
            self,
            merchant_to_assayer_note: MerchantToAssayerNote,
    ) -> OrderReceipt:
        """ Conduct a sale to assay a list of compounds.
        """
        raise NotImplementedError
