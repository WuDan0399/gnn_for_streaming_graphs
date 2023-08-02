# From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py

import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool

from utils import *


class pureGIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(5):
            mlp = MLP([in_channels, 64, 64], norm=None)  # original 32
            self.convs.append(GINConv(mlp, train_eps=False))
            in_channels = 64

        self.mlp = MLP([64, 64, out_channels], dropout=0.5)

    def forward(self, x, edge_index, batch):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = x.relu()

        x = global_add_pool(x, batch)
        return self.mlp(x)


def train(model, train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
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
        pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)



if __name__ == '__main__':
    # args = FakeArgs(dataset="reddit", aggr="max", binary=True, epochs=1)
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]
    add_mask(data)
    timing_sampler(data, args)

    # kwargs = {'batch_size' : 1, 'num_workers' : 1, 'persistent_workers' : False}
    kwargs = {'batch_size' : 16, 'num_workers' : 4, 'persistent_workers' : False}

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

        node_indices = torch.arange(data.num_nodes)
        if data.num_nodes > 20000 :
            train_indices = node_indices[:int(data.num_nodes * 0.3)]
            test_indices = node_indices[int(data.num_nodes * 0.3):int(data.num_nodes * 0.35)]
        else:
            train_indices = node_indices[:int(data.num_nodes * 0.8)]
            test_indices = node_indices[int(data.num_nodes * 0.8):]

        train_loader = EgoNetDataLoader(data, train_indices, k=5, **kwargs)
        test_loader = EgoNetDataLoader(data, test_indices, k=5, **kwargs)

        sample_batch = next(iter(train_loader))
        model = pureGIN(sample_batch.x.shape[1], 2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer)
            # train_acc = test(model, train_loader)
            test_acc = test(model, test_loader)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f}')
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
    else:
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        
        node_indices = torch.arange(data.num_nodes)
        loader = EgoNetDataLoader(data, node_indices, **kwargs)

        sample_batch = next(iter(loader))
        model = pureGIN(sample_batch.x.shape[1], 2).to(device)
        model = load(model, available_model[index_best_model])

        start = time.perf_counter()
        for _ in tqdm(range(10)):
            test_acc = test(model, loader)
        end = time.perf_counter()
        print(f'Full Graph. Inference time: {(end - start)/10:.4f} seconds for 10 iterations.')
        print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)
        clean(available_model)