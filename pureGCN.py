# NOTE: run with baselinePyG conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#################################################################################
from torch_geometric.nn import GCNConv

from utils import *

class pureGCN(torch.nn.Module) :  # Only for inference
    def __init__(self, in_channels, hidden_channels, out_channels, args) :
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False,
                             normalize=not args.use_gdc, aggr=args.aggr)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False,
                             normalize=not args.use_gdc, aggr=args.aggr)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        return x
