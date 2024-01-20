# From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py

import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool

from utils import *


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        num_layers = 5
        self.save_int = args.save_int

        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            out_c = 64 if i != num_layers - 1 else out_channels
            mlp = MLP([in_channels, 64, out_c], norm=None)  # original 32
            self.convs.append(GINConv(mlp, train_eps=False,
                              save_intermediate=self.save_int, aggr=args.aggr))
            in_channels = 64


    def forward(self, x, edge_index):
        out_per_layer = {}
        intermediate_result_per_layer = defaultdict(
            lambda: defaultdict(lambda: torch.empty((0))))

        for i, conv in enumerate(self.convs):
            x, intermediate_result = conv(x, edge_index)

            if self.save_int:
                intermediate_result_per_layer[f"layer{i+1}"] = intermediate_result

            x = x.relu()

        return None, out_per_layer, intermediate_result_per_layer
