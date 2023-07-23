# From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import MLP, GINConv, global_add_pool

from utils import *


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(5):
            mlp = MLP([in_channels, 64, 64])  # original 32
            self.convs.append(GINConv(mlp, train_eps=False, save_intermediate=args.save_int))
            in_channels = 64

        self.mlp = MLP([64, 64, out_channels], norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            print(x)
            x = x.relu()

        x = global_add_pool(x, batch)
        return self.mlp(x)


def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # out = model(data.x, data.edge_index, data.batch)
        out = model(data.x, data.edge_index,torch.zeros((data.x.shape[0],), dtype=torch.int64))
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        # pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        pred = model(data.x, data.edge_index, torch.zeros((data.x.shape[0],), dtype=torch.int64)).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    model = GIN(dataset.num_features, dataset.num_classes, args).to(device)

    print_dataset(dataset)
    data = dataset[0].to(device)
    add_mask(data)
    print_data(data)

    # Compile the model into an optimized version:
    # Note that `compile(model, dynamic=True)` does not work yet in PyTorch 2.0, so
    # we use `transforms.Pad` and static compilation as a current workaround.
    # See: https://github.com/pytorch/pytorch/issues/94640
    # model = torch_geometric.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    available_model = []
    name_prefix = f"{args.dataset}_GIN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        best_test_acc = 0
        best_model_state_dict = None
        patience = args.patience
        it_patience = 0
        train_loader, val_loader, test_loader = data_loader(data, num_layers=5, num_neighbour_per_layer=-1,
                                                            separate=True, disjoint=True)
        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer)
            train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
                  f'Test: {test_acc:.4f}')
            if best_test_acc < test_acc :
                best_test_acc = test_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else :
                it_patience = it_patience + 1
                if it_patience >= patience :
                    print(f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, best_test_acc, name_prefix)