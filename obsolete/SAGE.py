#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
#################################################################################

import copy

import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from dynamic.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--epochs', type=int, default=20)
args = parser.parse_args()

if args.dataset == "Cora":
    # 2,708,  10,556,  1,433 , 7
    dataset = Planetoid("../datasets/Planetoid", "Cora")
elif args.dataset == 'reddit':
    dataset = Reddit("../datasets/Reddit")
elif args.dataset == "cora":
    dataset = CitationFull("../datasets/CitationFull", "Cora")
elif args.dataset == "yelp":
    dataset = Yelp("../datasets/Yelp")
elif args.dataset == "amazon":
    dataset = AmazonProducts("../datasets/Amazon")

# Already send node features/labels to GPU for faster access during sampling:
data = dataset[0].to(device, 'x', 'y')

if not hasattr(data, "train_mask"):
    train_ratio, test_ratio, validation_ratio = 0.75, 0.1, 0.15
    rand_choice = np.random.rand(len(data.y))
    data.train_mask = torch.tensor(
        [True if x < train_ratio else False for x in rand_choice], dtype=torch.bool)
    data.test_mask = torch.tensor([True if x >= train_ratio and x <
                                   train_ratio+test_ratio else False for x in rand_choice], dtype=torch.bool)
    data.val_mask = torch.tensor([True if x >= train_ratio +
                                  test_ratio else False for x in rand_choice], dtype=torch.bool)


kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                              num_neighbors=[25, 10], shuffle=True, **kwargs)

subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                 num_neighbors=[-1], shuffle=False, **kwargs)

# No need to maintain these features during evaluation:
del subgraph_loader.data.x, subgraph_loader.data.y
# Add global node index information.
subgraph_loader.data.num_nodes = data.num_nodes
subgraph_loader.data.n_id = torch.arange(data.num_nodes)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="max"))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="max"))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu_()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader):
        pbar = tqdm(total=len(subgraph_loader.dataset) * len(self.convs))
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all


model = SAGE(dataset.num_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs


available_model = []
name_prefix = f"{args.dataset}_SAGE"
for file in os.listdir("../examples/trained_model"):
    if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file):
        available_model.append(file)

if len(available_model) == 0:  # no available model, train from scratch

    for epoch in range(1, 11):  # 11
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')
        save(model.state_dict(), epoch, test_acc, name_prefix)

    best_val_acc = 0
    best_model_state_dict = None
    patience = 3
    it_patience = 0
    for epoch in range(1, args.epochs + 1) :
        loss = train(model, data, optimizer)
        train_acc, val_acc, tmp_test_acc = test(model, data)

        if val_acc > best_val_acc :
            best_val_acc = val_acc
            best_model_state_dict = model.state_dict()
            it_patience = 0
        else :
            it_patience = it_patience + 1
            if it_patience >= patience :
                save(best_model_state_dict, epoch, tmp_test_acc, name_prefix)
                print(f"No accuracy improvement {best_val_acc} in {patience} epochs. Early stopping.")
                break
else:  # choose the model with the highest test acc
    accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
    index_best_model = np.argmax(accuracy)
    model = load(model, available_model[index_best_model])
    train_acc, val_acc, test_acc = test()
    print(f'Test: {test_acc:.4f}')

    available_model.pop(index_best_model)
    clean(available_model)
