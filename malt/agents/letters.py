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
    def __init__(self, data={}):
        self.data = data

# =============================================================================
# MODULE CLASSES
# =============================================================================
class QueryReceipt(Letter):
    """ Receipt for query. """
    def __init__(self, data):
        super(QueryReceipt, self).__init__(data)

class OrderReceipt(Letter):
    """ Receipt for order. """
    def __init__(self, data):
        super(OrderReceipt, self).__init__(data)

class MerchantQuote(Letter):
    """ Quote from merchant. """
    def __init__(self, data):
        super(MerchantQuote, self).__init__(data)

class AssayerQuote(Letter):
    """ Quote from assayer. """
    def __init__(self, data):
        super(AssayerQuote, self).__init__(data)

class Quote(Letter):
    """ Receipt for quote. """
    def __init__(self, data):
        super(MerchantQuote, self).__init__(data)

class MerchantToAssayerNote(Letter):
    """ A note from merchant to assayer. """
    def __init__(self, data):
        super(MerchantToAssayerNote, self).__init__(data)

class Report(Letter):
    """ A report. """
    def __init__(self, data):
        super(Report, self).__init__(data)
