# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union
from .agent import Agent
from .center import Center
from .merchant import Merchant
from .assayer import Assayer
from .letters import Letter
from ..data.dataset import Dataset
from ..policy import Policy
from ..models.model import Model

# =============================================================================
# BASE CLASSES
# =============================================================================
class Player(Agent):
    """ Base classes for player.

    Methods
    -------
    query(molecules, merchant, assayer)
        Query about molecules with specific merchant and assayer.

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
    def __init__(self, center: Center, name: str = ""):
        super(Player, self).__init__()
        self.center = center
        self.name = name
        self.center.register(self)

    def query(
        self, molecules: list, merchant: Merchant, assayer: Assayer
    ) -> Letter:
        return self.center.receive_query(
            molecules=molecules,
            player=self,
            merchant=merchant,
            assayer=assayer,
        )

    def check(self, receipt):
        return self.center.check(
            player=self,
            receipt=receipt,
        )

    def order(
        self, quote: Letter
    ) -> Letter:
        return self.center.order(quote=quote)

# =============================================================================
# MODULE CLASSES
# =============================================================================
class AutonomousPlayer(Player):
    """ A player that explores the chemical space with a model and a policy.

    Parameters
    ----------
    center : Center
        The control center the player is attached to.
    name : str
        The name of the player.
    model : malt.Model
        A model that takes a batch of molecules and form a function for
        predictive distribution.
    policy : malt.Policy
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
        center: Center,
        model: Model,
        policy: Policy,
        trainer: callable,
    ) -> None:
        super(AutonomousPlayer, self).__init__(
            center=center, model=model
        )
        self.model = model
        self.policy = policy
        self.trainer = trainer
        self.history = Dataset([])

    def train(self, trainer):
        self.model = trainer(self)
        return self.model

    
