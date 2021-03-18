# =============================================================================
# IMPORTS
# =============================================================================
import abc
from typing import Union
from .agent import Agent
from ..data.dataset import Dataset

# =============================================================================
# BASE CLASS
# =============================================================================
class Message(abc.ABC):
    """Base class for messages between merchant, assayer, and player.

    Parameters
    ----------
    to : Agent
        Sender of message.
    fro : Agent
        Addressee of message.
    points : List[Point]
        A list of molecules with attributes.
    id : int
        The identification for the Message.

    """

    def __init__(
        self,
        to: Agent,
        fro: Agent,
        points: Union[None, Dataset]=None,
        extra={},
    ):
        super(Message, self).__init__()
        if points is None:
            points = Dataset([])

        self.points = points
        self.id = id(self)
        self.extra = extra
        self.to = to
        self.fro = fro

    def __repr__(self):
        return self.__class__.__name__

# =============================================================================
# MODULE CLASSES
# =============================================================================
class QueryReceipt(Message):
    """ Receipt for query. """

    def __init__(self, *args, **kwargs):
        super(QueryReceipt, self).__init__(*args, **kwargs)


class OrderReceipt(Message):
    """ Receipt for order. """

    def __init__(self, *args, **kwargs):
        super(OrderReceipt, self).__init__(*args, **kwargs)

class Quote(Message):
    """ Receipt for quote. """

    def __init__(self, *args, **kwargs):
        super(Quote, self).__init__(*args, **kwargs)


class MerchantToAssayerNote(Message):
    """ A note from merchant to assayer. """

    def __init__(self, *args, **kwargs):
        super(MerchantToAssayerNote, self).__init__(*args, **kwargs)

class Report(Message):
    """ Report. """

    def __init__(self, *args, **kwargs):
        super(Report, self).__init__(*args, **kwargs)
