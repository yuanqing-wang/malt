# =============================================================================
# IMPORTS
# =============================================================================
import abc
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
        points: Dataset = Dataset([]),
        id: str = None,
        extra={},
    ):
        print([point.smiles for point in points.points])
        self.points = points
        if id is None:
            import time

            id = int(time.time() * 100000)
        self.id = id
        self.extra = extra
        self.to = to
        self.fro = fro


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
