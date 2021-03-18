# =============================================================================
# IMPORTS
# =============================================================================
from ..agents.center import Center
from ..agents.player import Player
from ..agents.merchant import Merchant
from ..agents.assayer import Assayer
from ..agents.messages import (
    Quote,
    Message,
    MerchantToAssayerNote,
    OrderReceipt,
    QueryReceipt,
    Report,
)

# =============================================================================
# MODULE CLASSES
# =============================================================================
class FakeMerchant(Merchant):
    def __init__(self, dataset, *args, **kwargs):
        super(FakeMerchant, self).__init__(*args, **kwargs)
        self.dataset = dataset

    def query(self, points):
        assert all([point in self.dataset for point in points])
        quote = Quote(fro=self, to=self.center)
        quote.extra["price"] = 0.0
        quote.points = points

        # generate receipt
        query_receipt = QueryReceipt(to=self.center, fro=self)
        self.cache[query_receipt.id] = quote

        return query_receipt

    def order(self, quote):
        # generate order receipt

        order_receipt = OrderReceipt(
            fro=self,
            to=self.center,
        )

        return order_receipt

    def _check_query(self, query_receipt: QueryReceipt):
        return self.cache[query_receipt.id]

    def _check_order(self, order_receipt: OrderReceipt):
        return True

    def catalogue(self):
        return lambda: self.dataset


class FakeAssayer(Assayer):
    """ Fake assayer. """

    def __init__(self, dataset, *args, **kwargs):
        super(FakeAssayer, self).__init__()
        self.dataset = dataset

    def query(self, points):
        assert all([point.smiles in self.dataset.lookup for point in points])

        quote = Quote(fro=self, to=self.center)
        quote.extra["price"] = 0.0
        quote.points = points

        # generate receipt
        query_receipt = QueryReceipt(to=self.center, fro=self)
        self.cache[query_receipt.id] = quote

        return query_receipt

    def order(self, quote: Quote):
        order_receipt = OrderReceipt(
            fro=self,
            to=self.center,
        )
        order_receipt.points = quote.points
        self.cache[order_receipt.id] = order_receipt

        return order_receipt

    def _check_query(self, query_receipt: QueryReceipt):
        return self.cache[query_receipt.id]

    def _check_order(self, order_receipt: OrderReceipt):
        points = order_receipt.points
        assert all([point.smiles in self.dataset.lookup for point in points])

        report = Report(
            fro=self,
            to=self.center,
        )

        for point in points:
            report.points.append(self.dataset[point.smiles])
        return report
