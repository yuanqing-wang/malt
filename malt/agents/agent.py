# =============================================================================
# IMPORTS
# =============================================================================
import abc
import logging
_logger = logging.getLogger(__name__)

# =============================================================================
# BASE CLASSES
# =============================================================================
class Agent(abc.ABC):
    """ Base class for an agent.

    """
    cache = {}
    def __init__(self, center=None, name=None):
        self.center = center
        self.id = id(self)

        if name is None:
            name = "%s (%s)" % (self.__class__.__name__, self.id)
            
        self.name = name

        _logger.debug(
            "%s initialized. " % str(self)
        )

    def __repr__(self):
        return self.name
