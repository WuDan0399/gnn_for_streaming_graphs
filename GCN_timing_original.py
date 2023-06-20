# NOTE: run with baselinePyG conda env!
# from tqdm import tqdm

from pureGCN import *
from utils import *
from theoretical_est import EdgelistToGraph, Mergable

@torch.no_grad()
def inference(model, data:Union[pyg.data.Data, pyg.loader.NeighborLoader], folder:str, prefix:str, log) :
    # Could be pyg.Data or NeighbourLoader
    model.eval()
    if isinstance(data, pyg.data.Data):
        print("Using Full Data")
        log.write("Using Full Data\n")
        start = time.time_ns()
        out = model(data.x, data.edge_index, data.edge_attr)
        end = time.time_ns()
        log.write(f"time: {(end-start)/10**9:.4f} seconds\n")
        torch.save(out, osp.join(folder, prefix + ".pt"))

    elif isinstance(data, pyg.loader.NeighborLoader):
        print("Using Neighbour Loader")
        log.write("Using Neighbour Loader\n")
        start = time.time()
        out = torch.tensor([])
        # for batch in tqdm(data) :
        for batch in data:
            batch = batch.to(device, 'edge_index')
            batch_out = model(batch.x, batch.edge_index)
            out = torch.cat((out, batch_out.cpu()), 0) if len(out) != 0 else batch_out.cpu()

        end = time.time()
        log.write(f"time: {(end - start)} seconds\n")
        torch.save(out, osp.join(folder, prefix + ".pt"))


def inference_affected(model, data:pyg.data.Data, edge_dict: dict, folder:str, prefix:str, batch_size:int, log) :
    nlayers = 2  # number of GNN layers
    niters = 10  # number of batches, to get an average performance
    print(f"Using Neighbour Loader for Affected Area Inference with batch size {batch_size}, averaged for {niters} iters.")
    log.write(f"Using Neighbour Loader for Affected Area Inference with batch size {batch_size}, averaged for {niters} iters.\n")

    connected_nodes = []
    for _ in range(niters):
        index_sampled_edges = np.random.choice(data.num_edges, batch_size, replace=False)
        unique_ends = torch.unique(data.edge_index[:, index_sampled_edges].flatten()).cpu()
        unique_ends = [element.item() for element in unique_ends]  # transform from torch tensor to list of int\fp
        connected_nodes.append(unique_ends)

    start = time.time_ns()

    for batch_connected_nodes in connected_nodes:  # run multiple times to avoid extreme cases.
        # out = np.array([])
        # get affected nodes (2 hop) given connected_nodes
        total_affected_nodes = set(batch_connected_nodes)
        for _ in range(nlayers):
            copy_total_affected_nodes = total_affected_nodes.copy()
            for node in total_affected_nodes:
                copy_total_affected_nodes.update(edge_dict[node])
            total_affected_nodes = copy_total_affected_nodes  #.copy()
        batch_affected_nodes = torch.LongTensor(list(total_affected_nodes))
        loader = data_loader(data, input_nodes=batch_affected_nodes, num_layers=nlayers, num_neighbour_per_layer=-1,
                             separate=False, persistent_workers=False)
        # for batch in tqdm(loader):
        for batch in loader:
            batch = batch.to(device, 'edge_index')
            batch_out = model(batch.x, batch.edge_index)
            # out = concate_by_id(out, batch_out[:batch_size].cpu(), batch.input_id)  # Global node index of each node in batch.
            # out = np.concatenate((out, batch_out), axis=0) if len(out) != 0 else batch_out
        del loader
    end = time.time_ns()
    log.write(f"time: {(end-start)/10**9/niters:.4f} seconds\n")
    # torch.save(out, osp.join(folder, prefix + ".pt"))


def main():

    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    print_dataset(dataset)
    data = dataset[0].to(device)
    use_loader = is_large(data) or args.use_loader

    model = pureGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        if not use_loader:
            print(f"No available model. Please run `python GCN.py --dataset {args.dataset} --aggr {args.aggr}`")
        else:
            print(f"No available model. Please run `python GCN_neighbor_loader.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])

        if args.range == "full":
            out_folder = osp.join("examples", "result", "ground_truth")
            create_directory(out_folder)
            prefix = "_".join([args.dataset, args.aggr, "T" if use_loader else "F"])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            if not use_loader:
                inference(model, data, out_folder, prefix, log)
            else:
                loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1, separate=False)  # load all neighbours
                inference(model, loader, out_folder, prefix, log)

        elif args.range == "affected":
            out_folder = osp.join("examples", "result", "affected")
            create_directory(out_folder)
            prefix = "_".join([args.dataset, args.aggr])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            edge_dict = to_dict(data.edge_index)
            if args.perbatch >= 1:
                batch_size = int(args.perbatch)
            else:
                batch_size = int(args.perbatch * data.num_edges)
            inference_affected(model, data, edge_dict, out_folder, prefix, batch_size, log)

        # remove redundant saved models, only save the one with the highest accuracy
        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()