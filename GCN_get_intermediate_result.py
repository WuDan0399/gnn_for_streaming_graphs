#  NOTE: run with gnnEnv conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#  Data Loader from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py
#################################################################################
import random

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.logging import init_wandb, log

from tqdm import tqdm
from utils import *
from GCN import GCN


@torch.no_grad()
def test(model, loader, folder:str, postfix: str) :
    model.eval()
    print("Using Neighbour Loader for Full Graph Inference")
    intermediate_result_each_layer = {}
    for batch in tqdm(loader):
        batch = batch.to(device, 'edge_index')
        batch_size = batch.batch_size
        _, batch_result_each_layer, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index)

        # save out_per_layer and intermediate result per layer
        for layer in batch_intermediate_result_per_layer :
            if layer not in intermediate_result_each_layer :
                intermediate_result_each_layer[layer] = {}
                intermediate_result_each_layer[layer]['a-'] = torch.empty((0))
                intermediate_result_each_layer[layer]['a'] = torch.empty((0))
            # 把batch_result_each_layer 按照batch.input_id  concate在一起。
            if len(intermediate_result_each_layer[layer]['a-']) != 0:
                intermediate_result_each_layer[layer]['a-'] = torch.concat((intermediate_result_each_layer[layer]["a-"],
                                                     batch_intermediate_result_per_layer[layer]["a-"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a-'] = batch_intermediate_result_per_layer[layer][
                                                                                "a-"][:batch_size].cpu()
            if len(intermediate_result_each_layer[layer]['a']) != 0:
                intermediate_result_each_layer[layer]['a'] = torch.concat((intermediate_result_each_layer[layer]["a"],
                                                     batch_intermediate_result_per_layer[layer]["a-"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a'] = batch_intermediate_result_per_layer[layer][
                                                                                "a"][:batch_size].cpu()

    for layer in intermediate_result_each_layer:
        create_directory(osp.join(folder, layer))
        torch.save(intermediate_result_each_layer[layer]['a-'], osp.join(folder, layer, f"before_aggregation{postfix}.pt"))
        torch.save(intermediate_result_each_layer[layer]['a'], osp.join(folder, layer, f"after_aggregation{postfix}.pt"))

    return


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    out_folder = osp.join("examples", "intermediate", args.dataset, args.aggr)

    print_dataset(dataset)
    data = dataset[0].to(device)

    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)
    model = model.to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)

    if len(available_model) == 0 :  # no available model, train from scratch
        print(f"No available model. Please run `python GCN_neighbor_loader.py --dataset {args.dataset} --aggr {args.aggr}`")

    else :  # choose the model with the highest test acc
        accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
                    len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
        index_best_model = np.argmax(accuracy)
        model = load(model, available_model[index_best_model])
        loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1, separate=False)  # load all neighbours
        _ = test(model, loader, out_folder, "")  # full graph result

        if args.perbatch < 1:
            batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
        else:
            batch_size = int(args.perbatch)
        print(f"Batch size for streaming graph: {batch_size}")

        # edge selection according to args.distribution
        sample_edges, initial_edges = edge_remove(data.edge_index.cpu().numpy(), batch_size, args.distribution,
                                                  data.is_directed())
        initial_edges = torch.from_numpy(initial_edges).to(device)  # from numpy to torch tensor
        # sample_nodes = np.unique(sample_edges.reshape((-1,)))

        ## run inference for the initial graph (before edge adding).
        data2 = data  # the bulky data.x is not changed, directly change edge index on the same varaible. They are not used together.
        data2.edge_index = initial_edges  # todo: 不是很确定这里改变了edge_index后，会不会产生isolated node，进而影响后面的判断（by index)？
        data2.edge_index.to(device)

        loader = data_loader(data2, num_layers=2, num_neighbour_per_layer=-1, separate=False)  # select all
        if len(sample_edges)==1:
            post_fix = f"_({sample_edges[0,0]}, {sample_edges[0,1]})"
        else:
            post_fix = str(random.randint(0, 99))
            np.save(osp.join(out_folder, post_fix+".npy"), sample_edges)

        _ = test(model, loader, out_folder, post_fix)  # full graph result


if __name__ == '__main__':
    main()