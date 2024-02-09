# original code from : https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py

import argparse
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import SubsetRandomSampler

from ogb.nodeproppred import DglNodePropPredDataset
import os.path as osp
import re
import time
from utils import save, load, general_parser, timing_sampler
from configs import *
from load_dataset import load_dataset_dgl


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_worker):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_worker,  
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device) 
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                input_nodes.to(device)
                output_nodes.to(device)
                blocks[0].to(device)
                x = feat[input_nodes]
                h = layer(blocks[0], x) 
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_worker):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_worker, 
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)  
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                input_nodes.to(device)
                output_nodes.to(device)
                blocks[0].to(device)
                x = feat[input_nodes]
                h = layer(blocks[0], x) 
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))

    def forward(self, x):
        h = x
        h = F.relu(self.linears[0](h))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hid_size, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        num_layers = 5
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(input_dim, hid_size, hid_size)
            elif layer == num_layers - 1:
                mlp = MLP(hid_size, hid_size, output_dim)
            else:
                mlp = MLP(hid_size, hid_size, hid_size)
            self.layers.append(
                GINConv(mlp, learn_eps=False)
            ) 
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self,blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size, num_worker):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_worker, 
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)  
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                input_nodes.to(device)
                output_nodes.to(device)
                blocks[0].to(device)
                x = feat[input_nodes]
                h = layer(blocks[0], x)  
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes):  # full graph inference
    batch_size = loader_configs["batch_size"]
    num_worker = loader_configs["num_workers"]
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        pred = model.inference(
            graph, device, batch_size, num_worker
        ) 
        print(f"Inference time: {time.perf_counter()-start} seconds.")
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def mini_batch_infer(device, graph, nid, model, num_classes):
    model.eval()
    nid.to(device)
    batch_size = loader_configs["batch_size"]
    num_worker = loader_configs["num_workers"]
    sampler = NeighborSampler(
        [-1, -1], 
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    dataloader = DataLoader(
        g,
        nid,
        sampler,
        device=device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_worker,
        use_uva=use_uva,
    )
    with torch.no_grad():
        acc = evaluate(model, g, dataloader, num_classes)
        print(
            "Accuracy {:.4f} ".format( acc.item() )
        )

def train(args, device, g, dataset, model, num_classes):
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [10, 10],  
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
    return acc


if __name__ == "__main__":
    no_load = True  
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Using {args.mode} mode.")

    print("Loading data")
    dataset = load_dataset_dgl(args)
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    num_nodes = g.ndata["feat"].shape[0]
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes


    available_model = []

    if args.model == "SAGE":
        model = SAGE(in_size, 256, out_size).to(device)
        name_prefix = f"{args.dataset}_SAGE_{args.aggr}_dgl"
    elif args.model == "GCN":
        model = GCN(in_size, 256, out_size).to(device)
        name_prefix = f"{args.dataset}_GCN_{args.aggr}_dgl"
    elif args.model == "GIN":
        model = GIN(in_size, 64, out_size).to(device)
        name_prefix = f"{args.dataset}_GCN_{args.aggr}_dgl"
    else:
        print("unsupported model")

    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)

    if no_load == False and len(available_model) == 0:  # no available model, train from scratch
        print("Training...")
        acc = train(args, device, g, dataset, model, num_classes)
        save(model.state_dict(), args.epochs, acc, name_prefix)
    else:  # load the model, then inference
        if no_load:
            print("use random initialized weights")
        else:
            print("loading model ", available_model[0])
            model = load(model, available_model[0])
        timing_sampler(g, args)
        g = dgl.add_self_loop(g)
        print("Inference for whole graph...")
        acc = layerwise_infer(
            device, g, range(num_nodes), model, num_classes
        )
        print("Test Accuracy {:.4f}".format(acc.item()))
