# =============================================================================
# IMPORTS
# =============================================================================
from typing import Union, List
from .receipt import QueryReceipt, OrderReceipt, Quote
from .agent import Agent

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def query(
        smiles: str,
        assayer: Agent,
        synthesizer: Union[Agent, str] = "ALL",
    ) -> QueryReceipt:
    raise NotImplementedError

def get_quote(query_receipt: QueryReceipt) -> List:
    raise NotImplementedError

def order(quote: Quote) -> OrderReceipt:
    raise NotImplementedError
