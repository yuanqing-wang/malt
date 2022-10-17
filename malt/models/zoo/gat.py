from functools import partial
import torch
import dgl

class ConcatenationAttentionHeads(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        layer: type=dgl.nn.GATConv,
    ):
        super().__init__()
        self.layer = layer(in_features, out_features // num_heads, num_heads)
        self.__doc__ = self.layer.__doc__

    def forward(self, graph, feat):
        feat = self.layer(graph, feat)
        feat = feat.flatten(-2, -1)
        return feat


GAT = partial(ConcatenationAttentionHeads, layer=dgl.nn.GATConv)
GATDot = partial(ConcatenationAttentionHeads, layer=dgl.nn.DotGatConv)
