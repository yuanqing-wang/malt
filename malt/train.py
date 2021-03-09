# =============================================================================
# IMPORTS
# =============================================================================
import torch

# =============================================================================
# DEFAULT CONSTANTS
# =============================================================================
OPTIMIZER = "Adam"
LEARNING_RATE = 1e-3
N_EPOCHS = 10
BATCH_SIZE = -1

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def _prepare_training_kwargs(training_kwargs):
    # all the keys needed to be filled
    keys = ["optimizer", "learning_rate", "n_epochs", "batch_size"]

    # initialize an empty dictionary for the speicifications
    _training_kwargs = {}

    # enumerate through the keys
    for key in keys:
        if key in training_kwargs: # if specified
            _training_kwargs[key] = training_kwargs[key]
        else:
            _training_kwargs = globals()[key.upper()]

    return _training_kwargs

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================
def train(net, training_kwargs):
    """ Train the model with speicifications.

    Parameters
    ----------
    net : torch.nn.Module
        Supervised model.
    training_kwargs : dict
        A dictionary of the training specifications.

    Returns
    -------
    torch.nn.Module
        A trained model.
    """
    # get the missing values
    training_kwargs = _prepare_training_kwargs(
        training_kwargs
    )

    # get optimizer
    optimizer = getattr(
        torch.optim,
        training_kwargs["optimizer"]
    )(
        net.parameters(),
        training_kwargs["learning_rate"]
    )

    
