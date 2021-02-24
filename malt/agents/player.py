# =============================================================================
# IMPORTS
# =============================================================================
import abc
from typing import Union
from .agent import Agent
from .center import Center
from .merchant import Merchant
from .assayer import Assayer
from .letters import Quote, QueryReceipt, OrderReceipt, Report

# =============================================================================
# BASE CLASSES
# =============================================================================
class Player(Agent):
    """ Base classes for player.

    Methods
    -------
    query(molecules, merchant, assayer)
        Query about

    Attributes
    ----------
    center : Center
        Distribution center.

    """
    def __init__(self, center):
        super(Player, self).__init__()
        self.center = center

    def query(
        self, molecules: list, merchant: Merchant, assayer: Assayer
    ) -> QueryReceipt:
        self.center.receive_query(
            molecules=molecules,
            merchant=merchant,
            assayer=assayer,
        )

    def _check_query(self, query_receipt: QueryReceipt) -> Union[None, Quote]:
        self.center.check_query(query_receipt=query_receipt)

    def _check_order(self, order_receipt: OrderReceipt) -> Union[None, Report]:
        self.center.check_order(order_receipt=order_receipt)

    def check(self, receipt):
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)

    @abc.abstractmethod
    def order(
        self, quote: Quote
    ) -> OrderReceipt:
        raise NotImplementedError
