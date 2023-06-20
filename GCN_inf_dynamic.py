# NOTE: run with baselinePyG conda env!
from tqdm import tqdm

from GCN import *
from utils import *
from theoretical_est import EdgelistToGraph, Mergable

@measure_time
@torch.no_grad()
def inference_with_intermediate_value(model, data: Union[pyg.data.Data, pyg.loader.NeighborLoader], folder:str, prefix:str) :
    # Could be pyg.Data or NeighbourLoader
    model.eval()
    if isinstance(data, pyg.data.Data):
        print("Using Full Data")
        _, result_each_layer = model(data.x, data.edge_index, data.edge_attr)

    elif isinstance(data, pyg.loader.NeighborLoader):
        print("Using Neighbour Loader")
        # inference batches with loader. 不然会因为batch sampling的randomness导致不匹配。
        result_each_layer = {}
        for batch in tqdm(data):
            batch = batch.to(device, 'edge_index')
            batch_size = batch.batch_size
            _, batch_result_each_layer, _ = model(batch.x, batch.edge_index)  # 虽然不是很懂，但是copy-paste来的是这样的
            for layer in batch_result_each_layer:
                if layer not in result_each_layer:
                    result_each_layer[layer] = np.array([])
                # 把batch_result_each_layer 按照batch.input_id  concate在一起。
                result_each_layer[layer] = concate_by_id(result_each_layer[layer],
                                                      batch_result_each_layer[layer][:batch_size].cpu(),
                                                      batch.input_id) # Global node index of each node in batch.

    for layer in result_each_layer:
        np.save(osp.join(root, "dynamic", "examples", folder, prefix+f"_{layer}.npy"), result_each_layer[layer])
    return result_each_layer.keys()

def main():
    out_folder = "out_each_layer"
    clean_command = "rm -rf " + osp.join(root, "dynamic", "examples", out_folder) + "/*.npy"
    os.system(clean_command)

    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    prefix_final = "_".join(["final", args.dataset, args.aggr, args.distribution])
    prefix_initial = "_".join(["initial", args.dataset, args.aggr, args.distribution])


    dataset = load_dataset(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_dataset(dataset)
    data = dataset[0].to(device)
    # add_mask(data)  # inference doesnt need mask
    # print_data(data)
    use_loader = is_large(data) or args.use_loader

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        if not is_large(data) and not args.use_loader:
            print(f"No available model. Please run `python GCN.py --dataset {args.dataset} --aggr {args.aggr}`")
        else:
            print(f"No available model. Please run `python GCN_neighbor_loader.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])

        ## 1. run inference for whole graph (edge added). save the intermediate value for each layer and each node as the final targeted result.
        if not use_loader:
            inference_with_intermediate_value(model, data, out_folder, prefix_final)
        else:
            loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1, separate=False) # load all neighbours
            inference_with_intermediate_value(model, loader, out_folder, prefix_final)

        batch_size = int(args.perbatch / 100 * data.num_edges) # perbatch is [x]%, so divide by 100
        print(f"Batch size for streaming graph: {batch_size}")

        # edge selection according to args.distribution
        sample_edges, initial_edges = edge_remove(data.edge_index.cpu().numpy(), batch_size, args.distribution, data.is_directed())
        initial_edges = torch.from_numpy(initial_edges).to(device)  # from numpy to torch tensor
        sample_nodes = np.unique(sample_edges.reshape((-1,)))

        ## 2. calculate theorectially affected number of nodes
        graph = EdgelistToGraph(np.transpose(data.edge_index.cpu().numpy()))
        fetched_vertex_affected_merged = Mergable(graph, sample_edges, nlayers=2, num_effected_end=2)
        theoretical_affected = {"input" : sample_nodes.size}
        theoretical_affected['conv1'] = fetched_vertex_affected_merged[0]  # affected next layer equals to fetched previous layer
        theoretical_affected['conv2'] = fetched_vertex_affected_merged[1]

        ## 3. run inference for the initial graph (before edge adding). save the intermediate value for each layer and each node as the initial result
        # data2 = dataset[0].__copy__().to(device)
        data2 = data  # the bulky data.x is not changed, directly change edge index on the same varaible. They are not used together.
        data2.edge_index = initial_edges  # todo: 不是很确定这里改变了edge_index后，会不会产生isolated node，进而影响后面的判断（by index)？
        data2.edge_index.to(device)
        if not use_loader :
            layer_names = inference_with_intermediate_value(model, data2, out_folder, prefix_initial)
        else:
            loader = data_loader(data2, num_layers=2, num_neighbour_per_layer=-1, separate=False) # select all
            layer_names = inference_with_intermediate_value(model, loader, out_folder, prefix_initial)

        ## 4. load the two files\arrays and compare. calculate different rate at number level and node level. Different rate indicates the affected area (value changed)
        print(f"dataset: {args.dataset}    batch size: {batch_size}    aggregator: {args.aggr}    directed: {data.is_directed()}")
        print("\tdifferent numbers\ttotal numbers\tratio diff number\tdifferent nodes\ttheoretical affected node\ttotal nodes\tratio different node\tratio(real:theoretical)")
        for layer_name in layer_names:
            final = np.load(osp.join(root, "dynamic", "examples", out_folder, prefix_final + f"_{layer_name}.npy"))
            initial = np.load(osp.join(root, "dynamic", "examples", out_folder, prefix_initial + f"_{layer_name}.npy"))
            comparison = final == initial
            number_level_diff = np.sum(np.where(comparison == False, 1, 0))
            node_level_diff = np.sum(np.where(comparison.all(axis=1) == False, 1,0))
            print(f"[{layer_name}]\t{number_level_diff}\t{comparison.size}\t{number_level_diff/comparison.size}"
                  f"\t{node_level_diff}\t{theoretical_affected[layer_name]}\t{len(comparison)}"
                  f"\t{node_level_diff/len(comparison)}\t{node_level_diff/theoretical_affected[layer_name]}")
            del final, initial
        # remove redundant saved models, only save the one with the highest accuracy
        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()