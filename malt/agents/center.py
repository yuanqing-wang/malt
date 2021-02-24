# =============================================================================
# IMPORTS
# =============================================================================
import abc

# =============================================================================
# BASE CLASSES
# =============================================================================
class Center(abc.ABC):
    """ Base class for center. """
    def __init__(self):
        super(Center, self).__init__()

    @abc.abstractmethod
    def register_agent(self, agent):
        raise NotImplementedError

    @abc.abstractmethod
    def receive_query(self, query):
        raise NotImplementedError

    @abc.abstractmethod
    def check_query(self, query):
        raise NotImplementedError

    @abc.abstractmethod
    def check_order(self, order):
        raise NotImplementedError
