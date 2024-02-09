# From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py

import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool

from utils import *
from load_dataset import load_dataset

class pureGIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        num_layers = 5
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            out_c = 64 if i!=num_layers-1 else out_channels
            mlp = MLP([in_channels, 64, out_c], norm=None) 
            self.convs.append(GINConv(mlp, train_eps=False, aggr=args.aggr))
            in_channels = 64

    def forward(self, x, edge_index, batch=None):
        if isinstance(edge_index, list):
            for i, (graph_edge_index, _, size) in enumerate(edge_index):
                x = self.convs[i](x, graph_edge_index)
                x = F.relu(x)
            return x
        else:  
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = x.relu()
            return x


def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    total_examples = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        y = data.y[:data.batch_size]
        if len(data.y.shape) == 1:
            y = torch.nn.functional.one_hot(y.long().to(device), num_classes=out.shape[1])
        elif data.y.shape[1] == 1:
            y = torch.nn.functional.one_hot(y.long().flatten().to(device), num_classes=out.shape[1])
        loss = F.cross_entropy(out[:data.batch_size], y.float())
        loss.backward()
        optimizer.step()
        total_examples += data.batch_size
        total_loss += float(loss) * data.batch_size
    return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_examples = 0
    total_correct = 0
    for data in tqdm(loader):
        batch_size = data.batch_size
        data = data.to(device)
        out = model(data.x, data.edge_index)
        batch_y = data.y[:batch_size]
        if len(batch_y.shape) == 1: 
            pred = out.argmax(dim=-1) 
            total_correct += int((pred[:batch_size] == batch_y).sum())
            total_examples += batch_size
        elif batch_y.shape[1] == 1: 
            pred = out.argmax(dim=-1)
            total_correct += int((pred[:batch_size] == batch_y.flatten()).sum())
            total_examples += batch_size
        else:  
            pred = (out > 1).float()
            total_correct += int((pred[:batch_size] == batch_y).sum())
            total_examples += batch_size * batch_y.shape[1]

    return total_correct / total_examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]
    timing_sampler(data, args)

    available_model = []
    name_prefix = f"{args.dataset}_GIN_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)

    if len(available_model) == 0:  # no available model, train from scratch
        best_test_acc = 0
        best_model_state_dict = None
        patience = args.patience
        it_patience = 0

        train_loader, val_loader, test_loader = data_loader(data, num_layers=5, num_neighbour_per_layer=10)

        sample_batch = next(iter(train_loader))
        if args.dataset == 'papers':
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes + 1, args).to(device)
        else:
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer)
            test_acc = test(model, test_loader)
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
            if best_test_acc > test_acc:
                best_test_acc = test_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience + 1
                if it_patience >= patience:
                    print(
                        f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, best_test_acc, name_prefix)
    else:
        timing_sampler(data, args)
        accuracy = [float(re.findall("[0-9]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-9]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)

        num_eval_nodes = 100000
        node_indices = torch.randperm(data.num_nodes)[:num_eval_nodes]
        loader = data_loader(data, num_layers=5, num_neighbour_per_layer=-1,
                             separate=False, input_nodes=node_indices)

        sample_batch = next(iter(loader))
        if args.dataset == 'papers':
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes + 1, args).to(device)
        else:
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes, args).to(device)
        model = load(model, available_model[index_best_model])

        start = time.perf_counter()
        test(model, loader)
        end = time.perf_counter()
        print(
            f'Full Graph. Inference time: {(end - start)*(data.num_nodes/num_eval_nodes):.4f} seconds, averaged for {num_eval_nodes} nodes.')
