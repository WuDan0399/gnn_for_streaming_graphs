#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/reddit.py
#################################################################################

import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
# from tqdm import tqdm
from utils import *


class pureSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr="min"))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr="min"))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()

    # pbar = tqdm(total=int(len(train_loader.dataset)))
    # pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_examples += batch.batch_size
        total_loss += float(loss) * batch.batch_size
    #     pbar.update(batch.batch_size)
    # pbar.close()

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
    print(torch.get_num_threads())
    model.eval()

    total_examples = total_correct = 0
    # for batch in tqdm(loader):
    for batch in loader:
        batch_size = batch.batch_size
        out = model(batch.x.to(device), batch.edge_index.to(device))[:batch.batch_size]
        if len(batch.y.shape) !=1:
            pred = (out > 1).float()
            total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())  # element-wise compare for each task.
            total_examples += batch_size*batch.y.shape[1]  # classification for each task
        else:
            pred = out.argmax(dim=-1)  # one-hot
            total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())
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
    data = dataset[0].to(device)
    timing_sampler(data, args)
    add_mask(data)
    # print_data(data)

    model = pureSAGE(dataset.num_features, 256, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    available_model = []
    name_prefix = f"{args.dataset}_SAGE_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file):
            available_model.append(file)

    if len(available_model) == 0:  # no available model, train from scratch
        best_test_acc = 0
        best_model_state_dict = None
        patience = args.patience
        it_patience = 0
        train_loader, val_loader, test_loader = data_loader(data, num_layers=2, num_neighbour_per_layer=25,
                                                            separate=True)
        for epoch in range(1, args.epochs + 1) :
            loss = train(model, train_loader, optimizer, epoch)
            print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')
            test_acc = test(model, test_loader)
            print(f'Epoch: {epoch:02d}, Test: {test_acc:.4f}')
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience+1
                if it_patience >= patience:
                    print(f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, best_test_acc, name_prefix)

    else:  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])
        loader = data_loader(data, separate=False)
        start = time.perf_counter()
        for _ in range(10):
            test(model, loader)
        end = time.perf_counter()
        print(f'Full Graph. Inference time: {(end - start)/10:.4f} seconds for 10 iterations.')

        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()