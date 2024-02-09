# NOTE: run with baselinePyG conda env!

from pureGCN import *
from pureSAGE import *
from pureGIN import *
from utils import *
from tqdm import tqdm
from load_dataset import load_dataset
import quiver
from quiver.pyg import GraphSageSampler

@torch.no_grad()
def inference(model, data: Union[pyg.data.Data, pyg.loader.NeighborLoader, torch.utils.data.DataLoader],
              folder: str, prefix: str, log, x: quiver.Feature = None): 
    model.eval()
    n_repeat = 3
    if isinstance(data, pyg.data.Data):
        start = time.perf_counter()
        for _ in range(n_repeat):
            out = model(data.x, data.edge_index, data.edge_attr)
        end = time.perf_counter()
        log.write(f"time: {(end-start)/n_repeat:.4f} seconds for {n_repeat} iterations\n")
        torch.save(out, osp.join(folder, prefix + ".pt"))

    elif isinstance(data, pyg.loader.NeighborLoader):
        start = time.perf_counter()
        for _ in range(n_repeat):
            for batch in data:
                batch = batch.to(device, 'x', 'edge_index')
                batch_out = model(batch.x, batch.edge_index)
        end = time.perf_counter()
        log.write(f"time: {(end-start)/n_repeat:.4f} seconds for {n_repeat} iterations\n")

    elif isinstance(data, torch.utils.data.DataLoader): 
        csr_topo = quiver.CSRTopo(data.edge_index) 
        num_layer = count_layers(model)
        quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[-1]*num_layer, device=0, mode='GPU')
        subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                          batch_size=loader_configs["batch_size"], shuffle=False,
                                          num_workers=loader_configs["num_workers"])
        start = time.perf_counter()
        for _ in range(n_repeat):
            for seeds in data: 
                n_id, batch_size, adjs = quiver_sampler.sample(seeds)  
                adjs = [adj.to(device) for adj in adjs]
                out = model(x[n_id], adjs)
        end = time.perf_counter()
        log.write(f"time: {(end-start)/n_repeat:.4f} seconds for {n_repeat} iterations\n")

    else:
        print(
            "To evaluation time for full graph inference for GIN, please run `python pureGIN.py --dataset[dataset] --aggr [aggregator] --binary`")


@torch.no_grad()
def inference_affected(data, model, edge_dict: dict, connected_nodes, nlayers, egonet=False,
                       x: quiver.Feature = None, quiver_sampler:quiver.pyg.GraphSageSampler=None):   
    """
    :param data: pyg.data.Data instance.
    :param model: pyg model
    :param edge_dict: dict for out-going edges. {src:[dest]}
    :param connected_nodes: destinations of changed edges
    :param nlayers: number of model layer. Used to find all affected nodes.
    :return:
    """
    model.eval()
    start = time.perf_counter()
    total_affected_nodes = set(connected_nodes)
    for _ in range(nlayers - 1):
        copy_total_affected_nodes = total_affected_nodes.copy()
        for node in total_affected_nodes:
            if node in edge_dict:
                copy_total_affected_nodes.update(edge_dict[node])
        total_affected_nodes = copy_total_affected_nodes 
    batch_affected_nodes = torch.LongTensor(list(total_affected_nodes))
    print(f"time for geting affected area {time.perf_counter()-start} s")
    if x is not None:
        if quiver_sampler is None:
            raise ValueError("quiver_sampler is not provided, but quiver feature is provided.")
        # print("using Quiver Sampler")
        loader = data_loader(data, input_nodes=batch_affected_nodes, num_layers=nlayers, num_neighbour_per_layer=-1,
                             separate=False, persistent_workers=False, loader_type="quiver")
        for seeds in tqdm(loader):
            n_id, batch_size, adjs = quiver_sampler.sample(seeds) 
            adjs = [adj.to(device) for adj in adjs]
            out = model(x[n_id], adjs)
    else:
        if quiver_sampler is not None:
            raise ValueError("quiver_sampler is provided, but quiver feature is not.")
        loader = data_loader(data, input_nodes=batch_affected_nodes, num_layers=nlayers, num_neighbour_per_layer=-1,
                             separate=False, persistent_workers=False)
        for batch in tqdm(loader):
            batch = batch.to(device, 'x', 'edge_index')
            model(batch.x, batch.edge_index)


    end = time.perf_counter()
    return end - start


def batch_inference_affected(model, data: pyg.data.Data, log, folder: str, egonet=False, loader:str="default",
                             num_samples=None):
    nlayers = count_layers(model)
    total_time = 0

    entries = os.listdir(folder)
    data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))]
    it = 0
    for data_dir in data_folders:
        it += 1
        final_edges = torch.load(
            osp.join(folder, data_dir, "final_edges.pt"))
        data.edge_index = final_edges
        edge_dict = to_dict_wiz_cache(final_edges, osp.join(folder, data_dir), f'final_out_edge_dict.pickle')
        unique_ends = []
        if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")):
            inserted_edges = torch.load(
                osp.join(folder, data_dir, "inserted_edges.pt"))
            unique_ends = unique_ends + \
                [dst.item() for dst in inserted_edges[1]]
        if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")):
            removed_edges = torch.load(
                osp.join(folder, data_dir, "removed_edges.pt"))
            unique_ends = unique_ends + \
                [dst.item() for dst in removed_edges[1]]
        if loader == "default":
            inference_time = inference_affected(data, model, edge_dict,
                               unique_ends, nlayers, egonet)
            print(f"{inference_time:.4f} seconds for data dir {data_dir}\n")
            log.write(
                f"{inference_time:.4f} seconds for data dir {data_dir}\n")
            total_time = total_time + inference_time

        elif loader == "quiver":
            csr_topo = quiver.CSRTopo(data.edge_index)
            quiver_sampler = quiver.pyg.GraphSageSampler(csr_topo, sizes=[-1]*nlayers, device=0, mode='GPU') 
            x = quiver.Feature(rank=0, device_list=[0], device_cache_size="4G", cache_policy="device_replicate",
                               csr_topo=csr_topo)  
            x.from_cpu_tensor(data.x) 
            total_time = total_time + \
                inference_affected(data, model, edge_dict,
                               unique_ends, nlayers, False, x, quiver_sampler)
            if it%10 == 0:
                log.write(
                    f"{(total_time) / it:.4f} seconds, averaged with {it} iterations.\n")
                log.flush()
    log.write(
        f"{(total_time)/num_samples:.4f} seconds, averaged with {num_samples} iterations.\n")


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]
    use_loader = True

    if args.loader == "quiver":
        csr_topo = quiver.CSRTopo(data.edge_index)  
        x = quiver.Feature(rank=0, device_list=[0], device_cache_size="4G", cache_policy="device_replicate",
                           csr_topo=csr_topo)  
        x.from_cpu_tensor(data.x) 

    out_channels = dataset.num_classes if args.dataset != "papers" else dataset.num_classes + 1
    if args.model == "GCN":

        model = pureGCN(dataset.num_features,
                        args.hidden_channels, out_channels, args)
    elif args.model == "SAGE":
        model = pureSAGE(dataset.num_features,
                         args.hidden_channels, out_channels, args)
    elif args.model == "GIN":
        model = pureGIN(dataset.x.shape[1], out_channels, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_{args.model}_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)

    if len(available_model) == 0:  # no available model, train from scratch
        print(
            f"No available model. Please run `python pure{args.model}.py --dataset {args.dataset} --aggr {args.aggr}`")

    else:  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-9]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-9]\.[0-9]+", model_name)) != 0]
    use_loader = is_large(data) or args.use_loader
    
    if args.model=="GCN": 
        model = pureGCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)
    elif args.model=="SAGE":
        model = pureSAGE(dataset.num_features, args.hidden_channels, dataset.num_classes)
    elif args.model=="GIN":
        model = pureGIN(data.x.shape[1], 2).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_{args.model}_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        print(f"No available model. Please run `python pure{args.model}.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model]).to(device)

        if args.range == "full":
            out_folder = osp.join("examples", "timing_result", "ground_truth")
            create_directory(out_folder)
            prefix = "_".join(
                [args.model, args.dataset, args.aggr, "T" if use_loader else "F"])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            if not use_loader:
                data.to(device)
                inference(model, data, out_folder, prefix, log)
            else:
                loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1,
                                     separate=False, persistent_workers=False)  
                inference(model, loader, out_folder, prefix, log)
            log.close()

        elif args.range == "affected":
            create_directory(out_folder)
            prefix = "_".join(
                [args.model, args.dataset, args.aggr, str(args.perbatch), args.stream, args.loader])
            timing_sampler(data, args)
            out_folder = osp.join("examples", "timing_result", "affected")
            create_directory(out_folder)
            prefix = "_".join([args.model, args.dataset, args.aggr, str(args.perbatch), args.stream])
            log = open(osp.join(out_folder, f"{prefix}.log"), 'a')
            if args.perbatch >= 1:
                batch_size = int(args.perbatch)
            else:
                batch_size = int(args.perbatch * data.num_edges)

            batch_sizes = defaultConfigs.batch_sizes
            num_samples = defaultConfigs.num_samples
            num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None
            intr_result_dir = osp.join("examples", "intermediate", args.dataset, "min", args.stream,
                                       f"batch_size_{batch_size}")
            if args.model == "GIN":
                num_sample = max(10, num_sample//10)
            batch_inference_affected(
                model, data, log, folder=intr_result_dir, egonet=args.binary, loader=args.loader, num_samples=num_sample)
            intr_result_dir = osp.join("examples", "intermediate", args.dataset, "min", args.stream,
                                       f"batch_size_{batch_size}")
            batch_inference_affected(model, data, batch_size, log, folder=intr_result_dir, egonet=args.binary)
            log.close()


if __name__ == '__main__':
    main()
