import abc
from typing import Union, Callable
from .agent import Agent, Merchant, Assayer
from malt.models.supervised_model import SupervisedModel
from malt.data.dataset import Dataset

class Player(Agent):
    """ Base class for players.

    Methods
    -------
    merchandize(points, merchant):
        Conduct merchanidization.

    assay(assayer):
        Conduct assay.


    """
    def __init__(self):
        super(Player, self).__init__()

    def merchandize(
        self,
        dataset: Dataset,
        merchant: Union[Merchant, None],
    ):
        return merchant.merchandize(dataset)

    def assay(
        self,
        dataset: Dataset,
        assayer: Union[Assayer, None],
    ):
        return assayer.assay(dataset)

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

class ModelBasedPlayer(Player):
    """ Player with model.

    Parameters
    ----------
    model : SupervisedModel

    """
    def __init__(
            self,
            model: SupervisedModel,
            policy: Callable,
            trainer: Callable,
            merchant: Merchant,
            assayer: Assayer,
            knowledge: Union[Dataset, None]=None,
        ):
        super(ModelBasedPlayer, self).__init__()
        self.model = model
        self.policy = policy
        self.trainer = trainer
        self.merchant = merchant
        self.assayer = assayer
        if knowledge is None:
            knowledge = Dataset([])

    def merchandize(
        self,
        dataset: Dataset,
        merchant: Merchant,
    ):
        super().merchandize(dataset=dataset, merchant=merchant)

    def assay(
        self,
        dataset: Dataset,
        assayer: Assayer,
    ):
        dataset = assayer.assay(dataset)
        self.portfolio += dataset
        return dataset

    def train(self):
        self.model = self.trainer(self)
        return self.model
