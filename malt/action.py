# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union
from .receipt import QueryReceipt, OrderReceipt, Quote

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def query(smiles: str) -> QueryReceipt:
    raise NotImplementedError

def get_quote(query_receipt: QueryReceipt) -> Union[None, Quote]:
    raise NotImplementedError

def order(quote: Quote) -> OrderReceipt:
    raise NotImplementedError
