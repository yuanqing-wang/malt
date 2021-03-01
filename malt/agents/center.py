# =============================================================================
# IMPORTS
# =============================================================================
import abc
from .letters import QueryReceipt, OrderReceipt
from .player import Player
from .merchant import Merchant
from .assayer import Assayer

# =============================================================================
# BASE CLASSES
# =============================================================================
class Center(abc.ABC):
    """ Base class for center.

    Methods
    -------
    register(agent)
        Register an agent in the center.

    receive_query(player, query)
        Receive a query from a player and distribute it to merchant and assayer.

    


    """
    def __init__(self, name="center"):
        super(Center, self).__init__()
        self.name = name

    def register(self, agent):
        if isinstance(agent, Player):
            return self._register_player(agent)
        elif isinstance(agent, Merchant):
            return self._register_merchant(agent)
        elif isinstance(agent, Assayer):
            return self._register_assayer(agent)

    @abc.abstractmethod
    def _register_player(self, agent):
        raise NotImplementedError

    @abc.abstractmethod
    def _register_merchant(self, agent):
        raise NotImplementedError

    @abc.abstractmethod
    def _register_assayer(self, agent):
        raise NotImplementedError

    @abc.abstractmethod
    def receive_query(self, player, query):
        raise NotImplementedError

    def check(self, receipt):
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)

    @abc.abstractmethod
    def _check_query(self, query_receipt):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_order(self, order_receipt):
        raise NotImplementedError


# =============================================================================
# EXAMPLE MODULES
# =============================================================================
class DumbCenter(Center):
    def __init__(self):
        super(DumbCenter, self).__init__()
