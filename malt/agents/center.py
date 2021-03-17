# =============================================================================
# IMPORTS
# =============================================================================
import abc
from typing import List
from .messages import Message, QueryReceipt, OrderReceipt, Quote
from .player import Player
from .merchant import Merchant
from .assayer import Assayer
from ..point import Point
from ..data.dataset import Dataset

# =============================================================================
# BASE CLASSES
# =============================================================================
class Center(abc.ABC):
    """Base class for center.

    Methods
    -------
    register(agent)
        Register an agent in the center.

    receive_query(player, query)
        Receive a query from a player and distribute it to merchant and assayer.

    """

    cache = {}
    players = []
    merchants = []
    assayers = []

    def __init__(self, name="center"):
        super(Center, self).__init__()
        self.name = name

    def register(self, agent):
        """ Register an agent in the center. """
        if isinstance(agent, Player):
            return self._register_player(agent)
        elif isinstance(agent, Merchant):
            return self._register_merchant(agent)
        elif isinstance(agent, Assayer):
            return self._register_assayer(agent)

    def _register_player(self, agent):
        self.players.append(agent)

    def _register_merchant(self, agent):
        self.merchants.append(agent)

    def _register_assayer(self, agent):
        self.assayers.append(agent)

    @abc.abstractmethod
    def query(self, player, query):
        raise NotImplementedError

    def check(self, receipt):
        """ Check the status of an order or a query given a receipt. """
        if isinstance(receipt, QueryReceipt):
            return self._check_query(receipt)
        elif isinstance(receipt, OrderReceipt):
            return self._check_order(receipt)

    @abc.abstractmethod
    def order(self, player, quote):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_query(self, query_receipt):
        raise NotImplementedError

    @abc.abstractmethod
    def _check_order(self, order_receipt):
        raise NotImplementedError

    def _clean_cache(self):
        self.cache = {}


# =============================================================================
# EXAMPLE MODULES
# =============================================================================
class NaiveCenter(Center):
    def __init__(self, name="center"):
        super(NaiveCenter, self).__init__(name=name)

    def query(
        self,
        points: Dataset,
        player: Player,
        merchant: Merchant,
        assayers: List[Assayer],
    ):

        # check that all of the agents are registered
        assert player in self.players
        assert merchant in self.merchants
        for assayer in assayers:
            assert assayer in self.assayers

        # query from both merchant and assayer
        merchant_query_receipt = merchant.query(points)
        assayer_query_receipts = [
            assayer.query(points) for assayer in assayers
        ]

        # initialize query receipt
        query_receipt = QueryReceipt(
            points=points,
            to=player,
            fro=self,
        )

        # save copy in the cache
        self.cache[query_receipt.id] = {
            "merchant_query_receipt": merchant_query_receipt,
            "assayer_query_receipts": assayer_query_receipts,
        }

        return query_receipt

    def _check_query(self, query_receipt: QueryReceipt):
        # get cached query
        _cached_query = self.cache[query_receipt.id]
        merchant_query_receipt = _cached_query["merchant_query_receipt"]
        assayer_query_receipts = _cached_query["assayer_query_receipts"]

        # get quote from merchant and assayers
        merchant_quote = merchant_query_receipt.fro.check(
            merchant_query_receipt
        )

        assayer_quotes = [
            _assayer_query_receipt.fro.check(_assayer_query_receipt)
            for _assayer_query_receipt in assayer_query_receipts
        ]

        if merchant_quote is not None:
            if all(
                [
                    (_assayer_quote is not None)
                    for _assayer_quote in assayer_query_receipts
                ]
            ):
                return self._combine_quote(
                    merchant_quote=merchant_quote,
                    assayer_quotes=assayer_quotes,
                    player=query_receipt.to,
                )

        return None

    def _combine_quote(
        self,
        merchant_quote: Quote,
        assayer_quotes: List[Quote],
        player: Player,
    ):
        # initialize quote
        quote = Quote(to=player, fro=self)

        # assert molecules are consistent
        for _assayer_quote in assayer_quotes:
            assert _assayer_quote.points == merchant_quote.points

        # assign points to quote
        quote.points = merchant_quote.points

        # add price
        quote.extra["price"] = merchant_quote.extra["price"] + sum(
            [
                _assayer_quote.extra["price"]
                for _assayer_quote in assayer_quotes
            ]
        )

        # put merchant and assayer into extra
        quote.extra["merchant"] = merchant_quote.fro
        quote.extra["assayers"] = [
            _assayer_quote.fro for _assayer_quote in assayer_quotes
        ]

        # make cache
        self.cache[quote.id] = {
            "merchant_quote": merchant_quote,
            "assayer_quotes": assayer_quotes,
        }

        return quote

    def order(self, quote: Quote):
        # grab cache
        _cache = self.cache[quote.id]
        merchant_quote = _cache["merchant_quote"]
        assayer_quotes = _cache["assayer_quotes"]

        # order
        merchant_order_receipt = merchant_quote.fro.order(merchant_quote)

        assayer_order_receipts = [
            assayer_quote.fro.order(assayer_quote)
            for assayer_quote in assayer_quotes
        ]

        # initialize order receipt
        order_receipt = OrderReceipt(to=quote.to, fro=self)

        self.cache[order_receipt.id] = {
            "merchant_order_receipt": merchant_order_receipt,
            "assayer_order_receipts": assayer_order_receipts,
        }

        return order_receipt

    def _check_order(self, order_receipt: OrderReceipt):
        # grab cache
        _cache = self.cache[order_receipt.id]
        merchant_order_receipt = _cache["merchant_order_receipt"]
        assayer_order_receipts = _cache["assayer_order_receipts"]

        if (
            merchant_order_receipt.fro.check(merchant_order_receipt)
            is not None
        ):
            if all(
                [
                    (_assayer_order_receipt is not None)
                    for _assayer_order_receipt in assayer_order_receipts
                ]
            ):
                return [
                    _assayer_order_receipt.fro.check(_assayer_order_receipt)
                    for _assayer_order_receipt in assayer_order_receipts
                ]
