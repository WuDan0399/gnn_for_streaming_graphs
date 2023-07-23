# NOTE: run with baselinePyG conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#  Data Loader from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py
#################################################################################

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.logging import init_wandb, log

from tqdm import tqdm
from utils import *
from GCN import GCN


def train(model, train_loader, optimizer) :
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader) :
        optimizer.zero_grad()
        batch = batch.to(device, 'x', 'edge_index')
        batch_size = batch.batch_size
        out, _, _ = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader) :
    model.eval()
    total_examples = total_correct = 0

    t_loader, t_aggr, t_comb = 0,0,0
    sync(device)
    start_data_load = time.perf_counter()

    for batch in tqdm(loader) :
        batch = batch.to(device, 'x', 'edge_index')
        batch_size = batch.batch_size
        sync(device)
        end_data_load = time.perf_counter()
        t_loader = t_loader + (end_data_load - start_data_load)

        out, _, inter_and_time = model(batch.x, batch.edge_index)
        t_comb = t_comb + inter_and_time["layer1"]["t_comb"] + inter_and_time["layer2"]["t_comb"]
        t_aggr = t_aggr + inter_and_time["layer1"]["t_aggr"] + inter_and_time["layer2"]["t_aggr"]

        if len(batch.y.shape) !=1:
            pred = (out > 1).float()
            total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())  # element-wise compare for each task.
            total_examples += batch_size*batch.y.shape[1]  # classification for each task
        else:
            pred = out.argmax(dim=-1) # one-hot
            total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())
            total_examples += batch_size

        sync(device)
        start_data_load = time.perf_counter()
    print(f"[Time] Sampling (s)\tAggregation (s)\tCombination(s)")
    print(f"{t_loader}\t{t_aggr}\t{t_comb}")
    return total_correct / total_examples


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

    train_loader, val_loader, test_loader = data_loader(data, num_layers=2, num_neighbour_per_layer=10, separate=True)

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
            loss = train(model, train_loader, optimizer)
            val_acc = test(model, val_loader)
            log(Epoch=epoch, Loss=loss, Val=val_acc)
            if best_test_acc < val_acc:
                best_test_acc = val_acc
                best_model_state_dict = model.state_dict()
                it_patience = 0
            else:
                it_patience = it_patience+1
                if it_patience >= patience:
                    print(f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
                    break
        save(best_model_state_dict, epoch, best_test_acc, name_prefix)

    else :  # choose the model with the highest test acc
        print("Inference with full graph as input.")
        if args.interval > 0:
            timing_sampler(data, args)  # get the latest 'interval' edges for inference.
            print(data.num_edges)
        else:
            print("Disable edge sampling according to creation time.")

        loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1, separate=False)  # load all neighbours
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model]).to(device)
        test_acc = test(model, loader)
        print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)
        clean(available_model)

if __name__ == '__main__':
    main()