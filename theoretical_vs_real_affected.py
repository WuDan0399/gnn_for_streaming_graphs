# NOTE: run with gnnEnv conda env!
from tqdm import tqdm
from GCN import *
from utils import *
from torch_geometric.data import Data

@torch.no_grad()
def inference_with_intermediate_value(model, data: Union[pyg.data.Data, pyg.loader.NeighborLoader]) :
    # Could be pyg.Data or NeighbourLoader
    model.eval()
    if isinstance(data, pyg.data.Data):
        # print("Using Full Data")
        _, result_each_layer, _ = model(data.x, data.edge_index, data.edge_attr)

    elif isinstance(data, pyg.loader.NeighborLoader):
        # print("Using Neighbour Loader")
        # inference batches with loader. 不然会因为batch sampling的randomness导致不匹配。
        result_each_layer = {}
        for batch in data:
            batch = batch.to(device, 'edge_index')
            batch_size = batch.batch_size
            _, batch_result_each_layer, _ = model(batch.x, batch.edge_index)

            for layer in batch_result_each_layer:
                # 把batch_result_each_layer 按照batch.input_id  concate在一起。
                if layer not in result_each_layer :
                    result_each_layer[layer] = batch_result_each_layer[layer][:batch_size].cpu()
                else:
                    result_each_layer[layer] = torch.concat((result_each_layer[layer],
                         batch_result_each_layer[layer][:batch_size].cpu()))
    return result_each_layer

def main():
    difference_ratio = []
    nlayer = 2
    verification_tolerance = 1e-05

    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0].to(device)
    use_loader = is_large(data) or args.use_loader

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        if not use_loader :
            print(f"No available model. Please run `python GCN.py --dataset {args.dataset} --aggr {args.aggr}`")
        else :
            print(
                f"No available model. Please run `python GCN_neighbor_loader.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model]).to(device)

        if args.perbatch < 1 :
            batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
        else :
            batch_size = int(args.perbatch)

        folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                          f"batch_size_{batch_size}")

        entries = os.listdir(folder)
        data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))]

        # Multiple different data (initial state info and final state info) directory
        for data_dir in tqdm(data_folders[args.it*100: (args.it+1)*100]):

            initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt"))
            final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
            inserted_edges, removed_edges = [], []
            if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")) :
                inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
                inserted_edges = [(src.item(), dst.item()) for src, dst in zip(inserted_edges[0], inserted_edges[1])]
            if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")) :
                removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
                removed_edges = [(src.item(), dst.item()) for src, dst in zip(removed_edges[0], removed_edges[1])]
            init_out_edge_dict = to_dict(initial_edges)
            final_out_edge_dict = to_dict(final_edges)

            direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
            # calculate theoretical affected nodes, use as input_nodes.
            affected_nodes = affected_nodes_each_layer([init_out_edge_dict, final_out_edge_dict],
                                                       direct_affected_nodes, depth=nlayer - 1)
            last_layer_affected_nodes = torch.LongTensor(list(affected_nodes[nlayer - 1]))

            # inference for initial graph
            data2 = Data(x=data.x, edge_index=initial_edges)
            data2.to(device)
            loader = data_loader(data2, num_layers=nlayer, num_neighbour_per_layer=-1, separate=False,
                                 input_nodes=last_layer_affected_nodes, persistent_workers=False)
            init_inter_result = inference_with_intermediate_value(model, loader)

            # inference for final graph
            data3 = Data(x=data.x, edge_index=final_edges)
            data3.to(device)
            loader2 = data_loader(data3, num_layers=nlayer, num_neighbour_per_layer=-1, separate=False,
                                  input_nodes=last_layer_affected_nodes, persistent_workers=False)
            final_inter_result = inference_with_intermediate_value(model, loader2)

            # compare the hidden state in each layer. use is_close function.
            last_layer = "conv2"
            node_is_close = torch.all(torch.isclose(final_inter_result[last_layer], init_inter_result[last_layer],
                                               atol=verification_tolerance), dim=1)
            n_real_affected = node_is_close.shape[0] - torch.sum(node_is_close)

            difference_ratio.append(n_real_affected/len(affected_nodes[nlayer-1]))

        np.save(osp.join("examples", "theoretical", f"real_vs_theoretical_{args.dataset}_{batch_size}_{args.stream}_{args.it}.npy"),
                    difference_ratio)


if __name__ == '__main__':
    main()