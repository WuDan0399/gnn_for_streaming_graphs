# NOTE: run with baselinePyG conda env!

from pureGCN import *
from utils import *
from tqdm import tqdm

@torch.no_grad()
def inference(model, data:Union[pyg.data.Data, pyg.loader.NeighborLoader], folder:str, prefix:str, log) :
    # Could be pyg.Data or NeighbourLoader
    model.eval()
    if isinstance(data, pyg.data.Data):
        print("Using Full Data")
        log.write("Using Full Data\n")
        start = time.perf_counter()
        out = model(data.x, data.edge_index, data.edge_attr)
        end = time.perf_counter()
        log.write(f"time: {(end-start):.4f} seconds\n")
        torch.save(out, osp.join(folder, prefix + ".pt"))

    elif isinstance(data, pyg.loader.NeighborLoader):
        print("Using Neighbour Loader")
        log.write("Using Neighbour Loader\n")
        start = time.perf_counter()
        out = torch.tensor([])
        # for batch in tqdm(data) :
        for batch in data:
            batch = batch.to(device, 'x', 'edge_index')
            batch_out = model(batch.x, batch.edge_index)
            out = torch.cat((out, batch_out.cpu()), 0) if len(out) != 0 else batch_out.cpu()

        end = time.perf_counter()
        log.write(f"time: {(end - start)} seconds\n")
        torch.save(out, osp.join(folder, prefix + ".pt"))

def inference_affected(data, model, edge_dict:dict, connected_nodes, nlayers):
    """
    :param data: pyg.data.Data instance.
    :param model: pyg model
    :param edge_dict: dict for out-going edges. {src:[dest]}
    :param connected_nodes: destinations of changed edges
    :param nlayers: number of model layer. Used to find all affected nodes.
    :return:
    """
    start = time.perf_counter()
    total_affected_nodes = set(connected_nodes)
    for _ in range(nlayers - 1) :
        copy_total_affected_nodes = total_affected_nodes.copy()
        for node in total_affected_nodes :
            if node in edge_dict :
                copy_total_affected_nodes.update(edge_dict[node])
        total_affected_nodes = copy_total_affected_nodes  # .copy()
    batch_affected_nodes = torch.LongTensor(list(total_affected_nodes))
    loader = data_loader(data, input_nodes=batch_affected_nodes, num_layers=nlayers, num_neighbour_per_layer=25,
                         separate=False, persistent_workers=False)
    # for batch in tqdm(loader):
    for batch in loader :
        batch = batch.to(device, 'x', 'edge_index')
        model(batch.x, batch.edge_index)
    end = time.perf_counter()
    return end - start

def batch_inference_affected(model, data:pyg.data.Data, batch_size:int, log, folder:str=None) :
    nlayers = 2  # number of GNN layers
    total_time = 0
    if folder == None:
        edge_dict = to_dict(data.edge_index)
        niters = int(1000 // batch_size)  # number of batches, to get an average performance
        for _ in range(niters):
            index_sampled_edges = np.random.choice(data.num_edges, batch_size, replace=False)
            unique_ends = torch.unique(data.edge_index[:, index_sampled_edges].flatten()).cpu()
            unique_ends = [element.item() for element in unique_ends]  # transform from torch tensor to list of int\fp
            # call inferecnce_affected
            total_time = total_time + inference_affected(data, model, edge_dict, unique_ends, nlayers)
    else:
        # load from saved files in subdirs of folder
        entries = os.listdir(folder)
        data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))]
        niters = len(data_folders)
        for data_dir in tqdm(data_folders):
            final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
            edge_dict = to_dict(final_edges)
            unique_ends = []
            if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")) :
                inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
                unique_ends = unique_ends + [dst.item() for dst in inserted_edges[1]]
            if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")) :
                removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
                unique_ends = unique_ends + [dst.item() for dst in removed_edges[1]]
            total_time = total_time + inference_affected(data, model, edge_dict, unique_ends, nlayers)

    log.write(f"sage: {(total_time)/niters:.4f} seconds, averaged with {niters} iterations.\n")
    # torch.save(out, osp.join(folder, prefix + ".pt"))


def main():

    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    print_dataset(dataset)
    data = dataset[0].to(device)
    use_loader = is_large(data) or args.use_loader

    model = pureGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)

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
        model = load(model, available_model[index_best_model]).to(device)

        if args.range == "full":
            timing_sampler(data, args)  # Only sample for full graph inference
            out_folder = osp.join("examples", "timing_result", "ground_truth")
            create_directory(out_folder)
            prefix = "_".join(["SAGE", args.dataset, args.aggr, "T" if use_loader else "F"])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            if not use_loader:
                data.to(device)
                inference(model, data, out_folder, prefix, log)
            else:
                loader = data_loader(data, num_layers=2, num_neighbour_per_layer=25, separate=False)  # load all neighbours
                inference(model, loader, out_folder, prefix, log)

        elif args.range == "affected":
            timing_sampler(data, args)
            out_folder = osp.join("examples", "timing_result", "affected")
            create_directory(out_folder)
            prefix = "_".join(["SAGE", args.dataset, args.aggr, str(args.perbatch), args.stream])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            if args.perbatch >= 1:
                batch_size = int(args.perbatch)
            else:
                batch_size = int(args.perbatch * data.num_edges)

            intr_result_dir = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                                       f"batch_size_{batch_size}")
            batch_inference_affected(model, data, batch_size, log, folder=intr_result_dir)

        # remove redundant saved models, only save the one with the highest accuracy
        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()