# =============================================================================
# IMPORTS
# =============================================================================
import abc
import torch
import dgl
import functools
from dgl.nn.pytorch import GraphConv

# =============================================================================
# BASE CLASSES
# =============================================================================
class Representation(torch.nn.Module, abc.ABC):
    """Base class for a representation.

    Methods
    -------
    forward(g)
        Project a graph onto a fixed-dimensional space.

    """

    def __init__(self, out_features, *args, **kwargs) -> torch.Tensor:
        super(Representation, self).__init__()
        self.out_features = out_features

    @abc.abstractmethod
    def forward(self, g) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLBatchedGraph
            Input graph.
        """
        raise NotImplementedError


# =============================================================================
# MODULE CLASSES
# =============================================================================
class DGLRepresentation(Representation):
    """ Representation with DGL layer. """

    def __init__(
        self,
        layer: type = functools.partial(GraphConv, allow_zero_in_degree=True),
        in_features: int = 74,
        hidden_features: int = 128,
        out_features: int = 1,
        depth: int = 3,
        activation: callable = torch.nn.SiLU(),
        global_pool: str = "sum",
    ):
        super(DGLRepresentation, self).__init__(out_features=out_features)
        
        self.embedding_in = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            activation,
        )

        # construct model
        for idx in range(depth):
            setattr(
                self,
                "gn%s" % idx,
                layer(hidden_features, hidden_features),
            )

        self.embedding_out = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, hidden_features),
            # activation,
        )

        # output
        self.ff = torch.nn.Sequential(
            # torch.nn.Linear(hidden_features, hidden_features),
            # activation,
            torch.nn.Linear(hidden_features, out_features),
        )

        self.depth = depth
        self.global_pool = getattr(dgl, "%s_nodes" % global_pool)
        self.activation = activation

    def forward(self, g, field="h"):
        """Forward pass.

        Parameters
        ----------
        g : dgl.DGLGraph
            Input graph.

        """
        # make local copy
        g = g.local_var()
        h = g.ndata[field]
        h = self.embedding_in(h)

        # loop through the depth
        for idx in range(self.depth):
            h = getattr(self, "gn%s" % idx)(g, h)
            h = self.activation(h)

        h = self.embedding_out(h)
        g.ndata[field] = h

        # global pool
        h = self.global_pool(g, field)

        # final feedforward
        h = self.ff(h)

        return h
