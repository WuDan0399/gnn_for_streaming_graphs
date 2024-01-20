# NOTE: run with baselinePyG conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#################################################################################
from torch_geometric.nn import GCNConv

from utils import *
from load_dataset import load_dataset
from tqdm import tqdm
class pureGCN(torch.nn.Module):  # Only for inference
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False,
                             normalize=False, aggr=args.aggr)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False,
                             normalize=False, aggr=args.aggr)

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, list):  # used for quiver loader
            graph_edge_index_1 = edge_index[0][0]
            graph_edge_index_2 = edge_index[1][0]
            # size_1 = edge_index[0][2]
            # size_2 = edge_index[1][2]
            # x_target_1 = x[:size_1[1]]  # Target nodes are always placed first.
            # x_target_2 = x[:size_2[1]]

            x = self.conv1(x, graph_edge_index_1)
            x = x.relu()
            x = self.conv2(x, graph_edge_index_2)
            return x

        else: # edge_index is torch_sparse.SparseTensor, torch.Tensor, or torch.sparse.Tensor
            x = self.conv1(x, edge_index, edge_weight)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_weight)
            return x


@torch.no_grad()
def test(model, loader):
    model.eval()

    total_examples = total_correct = 0
    for batch in tqdm(loader):
    # for batch in loader:
        batch.to(device)
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)[
            :batch.batch_size]
        if len(batch.y.shape) != 1:
            pred = (out > 1).float()
            # element-wise compare for each task.
            total_correct += int((pred[:batch_size]
                                 == batch.y[:batch_size]).sum())
            # classification for each task
            total_examples += batch_size*batch.y.shape[1]
        else:
            pred = out.argmax(dim=-1)  # one-hot
            total_correct += int((pred[:batch_size]
                                 == batch.y[:batch_size]).sum())
            total_examples += batch_size
        total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())
        total_examples += batch_size
    return total_correct / total_examples


def main():
    # args = FakeArgs(dataset="PubMed", aggr='min')
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    # print_dataset(dataset)
    data = dataset[0]
    # timing_sampler(data, args)
    add_mask(data)

    if args.dataset == 'papers':
        model = pureGCN(dataset.num_features, 256, dataset.num_classes +1, args).to(device)
    else:
        model = pureGCN(dataset.num_features, 256, dataset.num_classes, args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file):
            available_model.append(file)

    if len(available_model) == 0:  # no available model, train from scratch
        print(f"No model available for GCN {args.dataset} {args.aggr}.")

    else:  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
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
            print(
                f'Full Graph. Inference time: {(end - start) * (data.num_nodes/num_eval_nodes) :.4f} seconds, averaged for {num_eval_nodes} nodes.')

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

