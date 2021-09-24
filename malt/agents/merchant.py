import abc


class Merchant(abc.ABC):
    """ Base class for all merchants.

    Methods
    -------
    catalogue (*args, **kwargs):
        Return an iterator over the catalogue of the merchant.

    
    """
