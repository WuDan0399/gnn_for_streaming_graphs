# import sys
# sys.path.append("..")

from GCN import *
from utils import *

from theoretical_est import EdgelistToGraph, Mergable


@torch.no_grad()
def inference_with_intermediate_value(model, data, folder:str, prefix:str) :
    model.eval()
    out, result_each_layer = model(data.x, data.edge_index, data.edge_attr)
    for layer in result_each_layer:
        np.save(osp.join(root, "dynamic", "examples", folder, prefix+f"_{layer}.npy"), result_each_layer[layer])
    return result_each_layer.keys()

def main():
    out_folder = "out_each_layer"
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print_dataset(dataset)
    data = dataset[0].__copy__().to(device)
    # add_mask(data)  # inference doesnt need mask
    print_data(data)
    use_loader = is_large(data)

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        print(f"No available model. Please run `python GCN.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])

        ## 1. run inference for whole graph (edge added). save the intermediate value for each layer and each node as the final targeted result.
        if not use_loader:
            inference_with_intermediate_value(model, data, out_folder, "final")
        else:
            # todo: Use data loader for the whole graph and test.
            pass

        batch_size = int(args.perbatch / 100 * data.num_edges) # perbatch is [x]%, so divide by 100
        print(f"Batch size: {batch_size}")

        # edge selection according to args.distribution
        sample_edges, initial_edges = edge_remove(data.edge_index.numpy(), batch_size, args.distribution, data.is_directed())
        initial_edges = torch.from_numpy(initial_edges).to(device) # from numpy to torch tensor
        sample_nodes = np.unique(sample_edges.reshape((-1,)))

        ## 2. calculate theorectially affected number of nodes
        graph = EdgelistToGraph(np.transpose(data.edge_index.numpy()))
        fetched_vertex_affected_merged = Mergable(graph, sample_edges, nlayers=2, num_effected_end=2)
        theoretical_affected = {"input" : sample_nodes.size}
        theoretical_affected['conv1'] = fetched_vertex_affected_merged[0]  # affected next layer equals to fetched previous layer
        theoretical_affected['conv2'] = fetched_vertex_affected_merged[1]

        ## 3. run inference for the initial graph (before edge adding). save rthe intermediate value for each layer and each node as the initial result
        data2 = dataset[0].__copy__().to(device)
        data2.edge_index = initial_edges  # todo: 不是很确定这里改变了edge_index后，会不会产生isolated node，进而影响后面的判断（by index)？

        if not use_loader :
            layer_names = inference_with_intermediate_value(model, data2, out_folder, "initial")
        else: # todo:
            pass

        ## 4. load the two files\arrays and compare. calculate different rate at number level and node level. Different rate indicates the affected area (value changed)
        print(f"dataset: {args.dataset}    batch size: {batch_size}    aggregator: {args.aggr}    directed: {data.is_directed()}")
        print("\tdifferent numbers\ttotal numbers\tratio diff number\tdifferent nodes\ttheoretical affected node\ttotal nodes\tratio different node\tratio(real:theoretical)")
        for layer_name in layer_names:
            final = np.load(osp.join(root, "dynamic", "examples", out_folder, f"final_{layer_name}.npy"))
            initial = np.load(osp.join(root, "dynamic", "examples", out_folder, f"initial_{layer_name}.npy"))
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