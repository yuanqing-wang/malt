import abc


class Merchant(abc.ABC):
    """ Base class for all merchants.

    Methods
    -------
    catalogue (*args, **kwargs):
        Return an iterator over the catalogue of the merchant.

    order (*args, **kwargs):
        Place an order from the merchant.

    """

    def __init__(self):
        super(Merchant, self).__init__()

    @abc.abstractmethod
    def catalogue(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def order(self, *args, **kwargs):
        raise NotImplementedError
