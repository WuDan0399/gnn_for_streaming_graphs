#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
#################################################################################

import copy

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from utils import *
from load_dataset import load_dataset


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args) :
        super().__init__()
        self.save_int = args.save_int
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=args.aggr, save_intermediate=args.save_int))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=args.aggr, save_intermediate=args.save_int))

    def forward(self, x, edge_index):
        out_per_layer = {}
        intermediate_result_per_layer = {}

        for i, conv in enumerate(self.convs):
            x, intermediate_result = conv(x, edge_index)

            if self.save_int:
                # must contains time info, could contain intermediate
                intermediate_result_per_layer[f"layer{i+1}"] = intermediate_result

            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            
            # since out_per_layer is rarely used, I simply remove it.

        return x, out_per_layer, intermediate_result_per_layer
