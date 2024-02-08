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
            mlp = MLP([in_channels, 64, out_c], norm=None)  # original 32
            self.convs.append(GINConv(mlp, train_eps=False, aggr=args.aggr))
            in_channels = 64
        # original implementation with global pooling
        # for _ in range(5):
        #     mlp = MLP([in_channels, 64, 64], norm=None)  # original 32
        #     self.convs.append(GINConv(mlp, train_eps=False, aggr=args.aggr))
        #     in_channels = 64
        #
        # self.mlp = MLP([64, 64, out_channels], dropout=0.5)

    def forward(self, x, edge_index, batch=None):
        if isinstance(edge_index, list):  # used for quiver loader
            for i, (graph_edge_index, _, size) in enumerate(edge_index):
                # x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i](x, graph_edge_index)
                x = F.relu(x)
            return x
        else:  # edge_index is torch_sparse.SparseTensor, torch.Tensor, or torch.sparse.Tensor
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                x = x.relu()
            # x = global_add_pool(x, batch)
            # return self.mlp(x)
            return x


def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    total_examples = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.to(device), data.edge_index.to(device))
        # classification task with 1 scalar label, cannot be used in training
        y = data.y[:data.batch_size]
        if len(data.y.shape) == 1:
            # out = out.argmax(dim=-1).float()
            y = torch.nn.functional.one_hot(y.long().to(device), num_classes=out.shape[1])
        elif data.y.shape[1] == 1: # 2d array but 1 element in each row.
            # out = out.argmax(dim=-1).reshape((-1,1)).float()
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
        if len(batch_y.shape) == 1:  # single label classification
            pred = out.argmax(dim=-1)  # one-hot
            total_correct += int((pred[:batch_size] == batch_y).sum())
            total_examples += batch_size
        elif batch_y.shape[1] == 1: # single label classification
            pred = out.argmax(dim=-1)  # one-hot
            total_correct += int((pred[:batch_size] == batch_y.flatten()).sum())
            total_examples += batch_size
        else:  # multi-label classification
            pred = (out > 1).float()
            # element-wise compare for each task.
            total_correct += int((pred[:batch_size] == batch_y).sum())
            # classification for each task
            total_examples += batch_size * batch_y.shape[1]

    return total_correct / total_examples


if __name__ == '__main__':
    # args = FakeArgs(dataset="reddit", aggr="max", binary=True, epochs=1)
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]
    # add_mask(data)
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

        # node_indices = torch.arange(data.num_nodes)
        # if data.num_nodes > 20000:
        #     train_indices = node_indices[:int(data.num_nodes * 0.3)]
        #     test_indices = node_indices[int(
        #         data.num_nodes * 0.3):int(data.num_nodes * 0.35)]
        # else:
        #     train_indices = node_indices[:int(data.num_nodes * 0.8)]
        #     test_indices = node_indices[int(data.num_nodes * 0.8):]
        # train_loader = EgoNetDataLoader(data, train_indices, k=5, **kwargs)
        # test_loader = EgoNetDataLoader(data, test_indices, k=5, **kwargs)

        train_loader, val_loader, test_loader = data_loader(data, num_layers=5, num_neighbour_per_layer=10)

        sample_batch = next(iter(train_loader))
        # model = pureGIN(sample_batch.x.shape[1], 2, args).to(device)  # For binary graph classification
        if args.dataset == 'papers':
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes + 1, args).to(device)
        else:
            model = pureGIN(sample_batch.x.shape[1], dataset.num_classes, args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer)
            # train_acc = test(model, train_loader)
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
        # loader = EgoNetDataLoader(data, node_indices, **kwargs)
        loader = data_loader(data, num_layers=5, num_neighbour_per_layer=-1,
                             separate=False, input_nodes=node_indices)

        sample_batch = next(iter(loader))
        # model = pureGIN(sample_batch.x.shape[1], 2).to(device)
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
        # print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)
        clean(available_model)
