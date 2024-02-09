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
from pureGCN import pureGCN

import torch.profiler
from torch.profiler import profile, record_function, ProfilerActivity

from load_dataset import load_dataset

def train(model, train_loader, optimizer) :
    model.train()

    total_examples = total_loss = 0
    for batch in tqdm(train_loader) :
        optimizer.zero_grad()
        batch = batch.to(device, 'x', 'y', 'edge_index')
        batch_size = batch.batch_size
        out = model(batch.x, batch.edge_index)
        y = batch.y[:batch_size]
        if len(batch.y.shape) == 1:
            y = torch.nn.functional.one_hot(y.long(), num_classes=out.shape[1]).float()
        elif batch.y.shape[1] == 1: 
            y = torch.nn.functional.one_hot(y.long().flatten(), num_classes=out.shape[1]).float()
        loss = F.cross_entropy(out[:batch_size], y.float())
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()
def test(model, loader):
    model.eval()
    total_examples = total_correct = 0
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        batch_y = batch.y[:batch_size]
        out = model(batch.x, batch.edge_index)
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
            total_correct += int((pred[:batch_size] == batch_y).sum())
            total_examples += batch_size * batch_y.shape[1]
    return total_correct / total_examples


@torch.no_grad()
def test_w_profiler(model, loader):
    model.eval()
    total_examples = total_correct = 0
    data_iter = iter(loader)
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=1000, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True
    ) as prof:
        print(f"Loader iterations: {len(loader)}")
        for i in tqdm(range(len(loader))):
            if i > 1200:
                break
            with torch.profiler.record_function("Subgraph consitution"):
                batch = next(data_iter)

            with torch.profiler.record_function("Data Trasfer"):
                batch = batch.to(device)

            batch_size = batch.batch_size

            with torch.profiler.record_function("model_inference"):
                out = model(batch.x, batch.edge_index)

            if len(batch.y.shape) != 1:
                pred = (out > 1).float()
                total_correct += int(
                    (pred[:batch_size] == batch.y[:batch_size]).sum())  # element-wise compare for each task.
                total_examples += batch_size * batch.y.shape[1]  # classification for each task
            else:
                pred = out.argmax(dim=-1)  # one-hot
                total_correct += int((pred[:batch_size] == batch.y[:batch_size]).sum())
                total_examples += batch_size

            del batch
            torch.cuda.empty_cache()

            prof.step()  # Call this at the end of each step to record stats for the step

    print(prof.key_averages().table())
    for avg in prof.key_averages():
        print(f"{avg.key}: {avg.cpu_time_total}")
        print(f"{avg.key}: {avg.cuda_time_total}")

    return total_correct / total_examples

def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    print("try load dataset")
    dataset = load_dataset(args)

    data = dataset[0]
    add_mask(data)

    train_loader, val_loader, test_loader = data_loader(data, num_layers=2, num_neighbour_per_layer=10, separate=True)
    torch.where(data.test_mask == True)
    if args.dataset == 'papers':
        model = pureGCN(dataset.num_features, args.hidden_channels, dataset.num_classes+1, args)
    else:
        model = pureGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)
    model, data = model.to(device), data
    optimizer = torch.optim.Adam([
        dict(params=model.conv1.parameters(), weight_decay=5e-4),
        dict(params=model.conv2.parameters(), weight_decay=0)
    ], lr=args.lr)  # Only perform weight-decay on first convolution.

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file) :
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
            print(f"Num Edges: {data.num_edges}")
        else:
            print("Disable edge sampling according to creation time.")

        loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1, separate=False)  # load all neighbours
        accuracy = [float(re.findall("[0-9]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-9]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model]).to(device)
        test_acc = test(model, loader)
        print(f'Test: {test_acc:.4f}')

        available_model.pop(index_best_model)

if __name__ == '__main__':
        main()
