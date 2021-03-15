# =============================================================================
# IMPORTS
# =============================================================================
import abc

# =============================================================================
# BASE CLASSES
# =============================================================================
class Policy(abc.ABC):
    """ Base class for policy. """
    def __init__(self):
        self.history = []

    @abc.abstractmethod
    def acquire(self, smiles):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError
