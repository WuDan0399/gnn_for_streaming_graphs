#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
#################################################################################
import torch_sparse
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from tqdm import tqdm
from utils import *
from load_dataset import load_dataset


class pureSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=args.aggr))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=args.aggr))

    def forward(self, x, edge_index):
        if isinstance(edge_index, list):  # used for quiver loader
            for i, (graph_edge_index, _, size) in enumerate(edge_index):
                # x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i](x, graph_edge_index)
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                    x = F.dropout(x, p=0.5, training=self.training)
            return x
        else:  # edge_index is torch_sparse.SparseTensor, torch.Tensor, or torch.sparse.Tensor
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < len(self.convs) - 1:
                    x = x.relu()
                    x = F.dropout(x, p=0.5, training=self.training)
            return x


def train(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_examples = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size].to(device)
        y_hat = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
        if len(batch.y.shape) == 1:
            # out = out.argmax(dim=-1).float()
            y = torch.nn.functional.one_hot(y.long(), num_classes=y_hat.shape[1]).float()
        elif batch.y.shape[1] == 1: # 2d array but 1 element in each row.
            # out = out.argmax(dim=-1).reshape((-1,1)).float()
            y = torch.nn.functional.one_hot(y.long().flatten(), num_classes=y_hat.shape[1]).float()
        loss = F.cross_entropy(y_hat, y.float())
        loss.backward()
        optimizer.step()

        total_examples += batch.batch_size
        total_loss += float(loss) * batch.batch_size
    return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_examples = 0
    total_correct = 0
    for batch in tqdm(loader):
    # for batch in loader:
        batch_size = batch.batch_size
        out = model(batch.x.to(device), batch.edge_index.to(device))[
            :batch.batch_size]
        batch_y = batch.y[:batch_size].to(device)
        if len(batch_y.shape) == 1:  # single label classification
            pred = out.argmax(dim=-1)  # one-hot
            total_correct += int((pred[:batch_size] == batch_y).sum())
            total_examples += batch_size
        elif batch_y.shape[1] == 1:  # single label classification
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


def main():
    # args = FakeArgs(dataset="papers", aggr='min')
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    # print_dataset(dataset)
    data = dataset[0]
    timing_sampler(data, args)
    # add_mask(data)

    if args.dataset == 'papers':
        model = pureSAGE(dataset.num_features, 256, dataset.num_classes+1, args).to(device)
    else:
        model = pureSAGE(dataset.num_features, 256, dataset.num_classes, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    available_model = []
    name_prefix = f"{args.dataset}_SAGE_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)

    if len(available_model) == 0:  # no available model, train from scratch
        best_test_acc = 0
        best_loss = torch.nan
        best_model_state_dict = None
        patience = args.patience
        it_patience = 0
        train_loader, val_loader, test_loader = data_loader(data, num_layers=2, num_neighbour_per_layer=10,
                                                            separate=True)
        for epoch in range(1, args.epochs + 1):
            loss = train(model, train_loader, optimizer, epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
            test_acc = test(model, test_loader)
            print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f}')
            # if best_test_acc < test_acc:
            #     best_test_acc = test_acc
            if epoch == 1 or best_loss > loss:
                best_loss = loss
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience+1
                if it_patience >= patience:
                    print(
                        # f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                        f"No accuracy improvement {best_loss} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, best_loss, name_prefix)
        # save(best_model_state_dict, epoch, best_test_acc, name_prefix)

    else:  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-9]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-9]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])

        if args.dataset in ['papers', "products"]:
            num_eval_nodes = 100000
            node_indices = torch.randperm(data.num_nodes)[:num_eval_nodes]
            # loader = EgoNetDataLoader(data, node_indices, **kwargs)
            loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1,
                                 separate=False, input_nodes=node_indices)
            start = time.perf_counter()
            test(model, loader)
            end = time.perf_counter()
            print(f"total nodes in dataset: {data.num_nodes}, sampled {num_eval_nodes}.")
            print(
                f'Full Graph. Inference time: {(end - start)* (data.num_nodes/num_eval_nodes):.4f} seconds, averaged for {num_eval_nodes} nodes.')

        else:
            loader = data_loader(data, separate=False)
            start = time.perf_counter()
            num_iter = 5
            for _ in range(num_iter):
                test(model, loader)
            end = time.perf_counter()

            print(
                f'Full Graph. Inference time: {(end - start)/num_iter:.4f} seconds, averaged for {num_iter} iterations.')
        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()

