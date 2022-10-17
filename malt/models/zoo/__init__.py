"""Model zoo with all flavors of graph neural network layers."""
from functools import partial
import dgl

GCN = partial(dgl.nn.GraphConv, allow_zero_in_degree=True)
GCN.__doc__ = dgl.nn.GraphConv.__doc__

GraphSAGE = partial(dgl.nn.SAGEConv, aggregator="mean")
GraphSAGE.__doc__ = dgl.nn.SAGEConv.__doc__
