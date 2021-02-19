# =============================================================================
# IMPORTS
# =============================================================================
from .agent import Agent

# =============================================================================
# BASE CLASSES
# =============================================================================

class FakeAgent(Agent):
    """ Base class for fake agent. """
    def __init__(self):
        super(FakeAgent, self).__init__()

class FakeMerchant(FakeAgent):
    """ Fake synthesizer. """
    def __init__(self):
        super(FakeMerchant, self).__init__()

class FakeAssayer(FakeAgent):
    """ Fake characterizer. """
    def __init__(self):
        super(FakeAssayer, self).__init__()
