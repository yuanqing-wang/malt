# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, List, Callable
from .agent import Agent
from .vendor import Vendor
from .assayer import Assayer
from .messages import QueryReceipt, OrderReceipt, Quote, Report, Message
from ..data.dataset import Dataset
from ..policy import Policy
from ..models.supervised_model import SupervisedModel
from ..point import Point
import logging
_logger = logging.getLogger(__name__)

# =============================================================================
# BASE CLASSES
# =============================================================================
class Player(Agent):
    """Base classes for player.

    Methods
    -------
    query(molecules, vendor, assayer)
        Query about molecules with specific vendor and assayer.

    check(receipt)
        Check the status of a query or an order.

    order(quote)
        Place an order based on a quote.

    Attributes
    ----------
    center : Center
        Distribution center.

    name : str
        Name (ID) of the player.

    """

    portfolio = Dataset([])

    def __init__(
        self,
        center=None,
        name: str = "",
        quote_review_function: Callable = (lambda x: True),
    ):
        super(Player, self).__init__()
        self.center = center
        self.name = name
        self.center.register(self)
        self.quote_review_fuction = quote_review_function

    def query(
        self, points: List[Point], vendor: Vendor, assayers: List[Assayer]
    ) -> QueryReceipt:

        _logger.debug(
            "%s queries %s with %s and %s" % (
                self, points, vendor, assayers
            )
        )

        return self.center.query(
            points=points,
            player=self,
            vendor=vendor,
            assayers=assayers,
        )

    def check(self, receipt: Message) -> Union[None, Quote]:
        return self.center.check(
            receipt=receipt,
        )

    def order(self, quote: Quote) -> OrderReceipt:

        _logger.debug(
            "%s orders %s" % (
                self, quote,
            )
        )

        return self.center.order(quote=quote)

    def append(self, points) -> None:
        if isinstance(points, Point):
            self.portfolio.append(points)
        elif isinstance(points, Dataset) or isinstance(points, List):
            self.portfolio += points


# =============================================================================
# MODULE CLASSES
# =============================================================================
class AutonomousPlayer(Player):
    """A player that explores the chemical space with a model and a policy.

    Parameters
    ----------
    center : Center
        The control center the player is attached to.
    name : str
        The name of the player.
    model : malt.Model
        A model that takes a batch of molecules and form a function for
        predictive distribution.
    policy : Callable
        A policy object that takes a predictive distribution and choose
        among the data points.
    trainer : callable

    Methods
    -------
    train()
        Train the model with some training_kwargs as speicification.

    """

    def __init__(
        self,
        name: str,
        center,
        model: SupervisedModel,
        policy: Callable,
        trainer: Callable,
    ) -> None:
        super(AutonomousPlayer, self).__init__(
            name=name,
            center=center,
        )
        self.model = model
        self.policy = policy
        self.trainer = trainer

    def train(self):
        self.model = self.trainer(self)
        return self.model

    def prioritize(self, points):
        g = next(
            iter(
                points.view(collate_fn="batch_of_g", batch_size=len(points))
            )
        )
        distribution = self.model.condition(g)
        idxs = self.policy(distribution)
        return points[idxs]
