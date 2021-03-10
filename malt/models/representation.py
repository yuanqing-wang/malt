# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch

# =============================================================================
# BASE CLASSES
# =============================================================================
class Representation(torch.nn.Module, abc):
    """ Base class for a regressor.

    """
    def __init__(self, *args, **kwargs):
        super(Representation, self).__init__()

    @staticmethod
    def forward(self, g):
        """ Forward pass.

        Parameters
        ----------
        g : dgl.DGLBatchedGraph
            Input graph.
        """
        raise NotImplementedError
