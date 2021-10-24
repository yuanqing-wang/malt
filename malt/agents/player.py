import abc
from typing import Union, Callable
from .agent import Agent
from .merchant import Merchant
from .assayer import Assayer
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
        merchant: Union[Merchant, None]=None,
    ):
        if merchant is None:
            merchant = self.merchant
        return merchant.merchandize(dataset)

    def assay(
        self,
        dataset: Dataset,
        assayer: Union[Assayer, None]=None,
    ):
        if assayer is None:
            assayer = self.assayer
        return assayer.assay(dataset)

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        raise NotImplementedError

class ModelBasedPlayer(Player):
    """ Player with model.

    Parameters
    ----------
    model : SupervisedModel
        Model to predict properties based on structure.

    policy : Callable
        Policy to rank candidates.

    trainer: Callable
        Function to train a model.

    merchant : Merchant
        Merchant that merchandizes candidates.

    assayer : Assayer
        Assayer that assays the candidates.

    portfolio : Dataset
        Inititial knowledge about data points.

    Note
    ----
    1. Perfolio respects order and could be used to analyze acquisition
        trajectory.

    """
    def __init__(
            self,
            model: SupervisedModel,
            policy: Callable,
            trainer: Callable,
            merchant: Merchant,
            assayer: Assayer,
            portfolio: Union[Dataset, None]=None,
        ):
        super(ModelBasedPlayer, self).__init__()
        self.model = model
        self.policy = policy
        self.trainer = trainer
        self.merchant = merchant
        self.assayer = assayer
        if portfolio is None:
            portfolio = Dataset([])
        self.portfolio = portfolio

    def merchandize(
        self,
        dataset: Dataset,
    ):
        return super().merchandize(dataset=dataset)

    def assay(
        self,
        dataset: Dataset,
    ):
        dataset = super().assay(dataset=dataset)
        self.portfolio += dataset
        return dataset

    def train(self):
        self.model = self.trainer(self)
        return self.model

    def prioritize(self):
        if len(self.merchant.catalogue()) == 0:
            return None
        posterior = self.model.condition(
            next(iter(self.merchant.catalogue().view("batch_of_g")))
        )
        score = self.policy(posterior)
        return self.merchant.catalogue()[int(score.argmax().item())]

class SequentialModelBasedPlayer(ModelBasedPlayer):
    """Model based player with step size equal one.

    Examples
    --------
    >>> import malt
    >>> player = SequentialModelBasedPlayer(
    ...    model = malt.models.supervised_model.SimpleSupervisedModel(
    ...        representation=malt.models.representation.DGLRepresentation(
    ...            out_features=128
    ...        ),
    ...        regressor=malt.models.regressor.NeuralNetworkRegressor(
    ...            in_features=128, out_features=1
    ...        ),
    ...        likelihood=malt.models.likelihood.HomoschedasticGaussianLikelihood(),
    ...    ),
    ...    policy=malt.policy.Greedy(),
    ...    trainer=malt.trainer.get_default_trainer(),
    ...    merchant=malt.agents.merchant.DatasetMerchant(
    ...        malt.data.collections.linear_alkanes(10),
    ...    ),
    ...    assayer=malt.agents.assayer.DatasetAssayer(
    ...        malt.data.collections.linear_alkanes(10),
    ...    )
    ... )

    >>> while True:
    ...     if player.step() is None:
    ...         break
    """
    def __init__(self, *args, **kwargs):
        super(SequentialModelBasedPlayer, self).__init__(
            *args, **kwargs
        )

    def step(self):
        best = self.prioritize()
        if best is None:
            return None
        best = Dataset([best])
        best = self.merchandize(best)
        best = self.assay(best)
        return best
