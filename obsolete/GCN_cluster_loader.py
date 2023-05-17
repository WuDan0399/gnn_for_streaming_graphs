#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#  Cluster data loader from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/cluster_gcn_reddit.py

# Need Metis support in Pytorch Sparse package. Failed to link metis when building
# pytorch sparse *RocM* build on AMD GPUs.
# Commands:
# python setup.py bdist_wheel
# cd dist
# pip install --upgrade --no-deps --force-reinstall torch_sparse-0.6.17-cp38-cp38-linux_x86_64.whl
# cd ../test
# python test_metis.py
#################################################################################

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

from tqdm import tqdm
from utils import *

class GCN(torch.nn.Module) :
    def __init__(self, in_channels, hidden_channels, out_channels, args) :
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False,
                             normalize=not args.use_gdc, aggr=args.aggr)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False,
                             normalize=not args.use_gdc, aggr=args.aggr)

    def forward(self, x, edge_index, edge_weight=None) :
        out_per_layer = {}
        out_per_layer["input"] = x.detach()
        x = F.dropout(x, p=0.5, training=self.training)  # prevent overfitting, cannot sparsify the input\network for inference
        x = self.conv1(x, edge_index, edge_weight).relu()
        out_per_layer["conv1"] = x.detach()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        out_per_layer["conv2"] = x.detach()
        return x, out_per_layer


    def inference(self, x_all, subgraph_loader, device) :
        pbar = tqdm(total=x_all.size(0) * 2)  # 2 for number of layers
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.

        out_per_layer = {}
        out_per_layer["input"] = x_all.detach()
        # 1st conv layer
        xs = []
        for batch_size, n_id, adj in subgraph_loader : #  a list of bipartite graph objects via the tuple (edge_index, e_id, size)
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x = self.conv1(x, edge_index).relu()
            target_result = x[:size[1]]  # target nodes and their sources are loaded. sources are isolated, remove for inference.
            xs.append(target_result.cpu())
            pbar.update(batch_size)
        x_all = torch.cat(xs, dim=0)

        out_per_layer["conv1"] = x_all.detach()

        # 2nd conv layer
        xs = []
        for batch_size, n_id, adj in subgraph_loader :  # a list of bipartite graph objects via the tuple (edge_index, e_id, size)
            edge_index, _, size = adj.to(device)
            x = x_all[n_id].to(device)
            x = self.conv2(x, edge_index)
            target_result = x[:size[
                1]]  # target nodes and their sources are loaded. sources are isolated, remove for inference.
            xs.append(target_result.cpu())
            pbar.update(batch_size)
        x_all = torch.cat(xs, dim=0)

        out_per_layer["conv2"] = x_all.detach()

        pbar.close()

        return x_all, out_per_layer


def train(model, train_loader, device, optimizer):
    model.train()

    total_loss = total_nodes = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


@torch.no_grad()
def test(model, data, subgraph_loader, device):  # Inference should be performed on the full graph.
    model.eval()

    out, _ = model.inference(data.x, subgraph_loader, device)
    y_pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = y_pred[mask].eq(data.y[mask]).sum().item()
        accs.append(correct / mask.sum().item())
    return accs


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
               hidden_channels=args.hidden_channels, device=device)

    print_dataset(dataset)
    data = dataset[0]
    add_mask(data)
    print_data(data)

    num_parts = 1500
    batch_size = 20

    print(f"num_parts: {num_parts}, batch size: {batch_size} clusters, "
          f"around {data.num_nodes/num_parts*batch_size:.2f} nodes {data.num_edges/num_parts*batch_size:.2f} edges.")
    cluster_data = ClusterData(data, num_parts=1500, recursive=False,
                               save_dir=dataset.processed_dir)
    train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=True,
                                 num_workers=12)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1], batch_size=1024,
                                      shuffle=False, num_workers=12)


    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)
    model, data = model.to(device), data.to(device)
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.


    if args.use_gdc :
        transform = T.GDC(
            self_loop_weight=1,
            normalization_in='sym',
            normalization_out='col',
            diffusion_kwargs=dict(method='ppr', alpha=0.05),
            sparsification_kwargs=dict(method='topk', k=128, dim=0),
            exact=True,
        )
        data = transform(data)


    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        best_test_acc = 0
        best_model_state_dict = None
        patience = args.patience
        it_patience = 0
        for epoch in range(1, args.epochs + 1) :
            loss = train(model, train_loader, device, optimizer)
            train_acc, val_acc, tmp_test_acc = test(model, data, subgraph_loader, device)
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=tmp_test_acc)
            if best_test_acc < tmp_test_acc:
                best_test_acc = tmp_test_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience+1
                if it_patience >= patience:
                    print(f"No accuracy improvement {best_test_acc:.5f} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, tmp_test_acc, name_prefix)

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])
        train_acc, val_acc, test_acc = test(model, data, subgraph_loader, device)
        print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)
        clean(available_model)

if __name__ == '__main__':
    main()