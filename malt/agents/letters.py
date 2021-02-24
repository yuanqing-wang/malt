# =============================================================================
# IMPORTS
# =============================================================================
import abc

# =============================================================================
# BASE CLASS
# =============================================================================
class Letter(abc.ABC):
    """ Base class for letters between merchant, assayer, and player.

    Attributes
    ----------
    id : int
        The identification for the letter.

    """
    def __init__(self, *args, **kwargs):
        pass


# =============================================================================
# MODULE CLASSES
# =============================================================================
class QueryReceipt(Letter):
    """ Receipt for query. """
    def __init__(self, id):
        self.id = id

class OrderReceipt(Letter):
    """ Receipt for order. """
    def __init__(self, id):
        self.id = id

class Quote(Letter):
    """ Receipt for quote. """
    def __init__(self, id):
        self.id = id

class Quote(Letter):
    """ A quote. """
    def __init__(self, id):
        self.id = id
