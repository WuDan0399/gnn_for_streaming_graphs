# NOTE: run with baselinePyG conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#################################################################################

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

from utils import *

class GCN(torch.nn.Module) :
    def __init__(self, in_channels, hidden_channels, out_channels, args) :
        super().__init__()
        self.save_int = args.save_int
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=False,
                             normalize=False, aggr=args.aggr, save_intermediate=args.save_int)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=False,
                             normalize=False, aggr=args.aggr, save_intermediate=args.save_int)

    def forward(self, x, edge_index, edge_weight=None):
        out_per_layer = {}
        intermediate_result_per_layer = {}

        if self.save_int:
            out_per_layer["input"] = x.detach()

        x = F.dropout(x, p=0.5, training=self.training)   # prevent overfitting, cannot sparsify the input\network for inference
        x, intermediate_result = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        if self.save_int :
            intermediate_result_per_layer["layer1"] = intermediate_result
            out_per_layer["conv1"] = x.detach()

        x = F.dropout(x, p=0.5, training=self.training)
        x, intermediate_result = self.conv2(x, edge_index, edge_weight)

        if self.save_int:
            intermediate_result_per_layer["layer2"] = intermediate_result
            out_per_layer["conv2"] = x.detach()

        return x, out_per_layer, intermediate_result_per_layer


def train(model, data, optimizer) :
    model.train()
    optimizer.zero_grad()
    result = model(data.x, data.edge_index, data.edge_attr)
    out = result[0]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data) :
    model.eval()
    result = model(data.x, data.edge_index, data.edge_attr)
    out = result[0]
    pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask] :
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

@torch.no_grad()
def test_all(model, data) :
    model.eval()
    result = model(data.x, data.edge_index, data.edge_attr)
    out = result[0]
    pred = out.argmax(dim=-1)
    return int((pred == data.y).sum()) / len(pred)


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    init_wandb(name=f'GCN-{args.dataset}', lr=args.lr, epochs=args.epochs,
               hidden_channels=args.hidden_channels, device=device)

    print_dataset(dataset)
    data = dataset[0]
    add_mask(data)
    print_data(data)

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
            loss = train(model, data, optimizer)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=tmp_test_acc)
            if best_test_acc < tmp_test_acc:
                best_test_acc = tmp_test_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience+1
                if it_patience >= patience:
                    print(f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, tmp_test_acc, name_prefix)

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])
        train_acc, val_acc, test_acc = test(model, data)
        print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)
        clean(available_model)

if __name__ == '__main__':
    main()