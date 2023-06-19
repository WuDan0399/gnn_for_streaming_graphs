# NOTE: run with baselinePyG conda env!
##  GNN has inherent stablility, it is necessary to update for every interaction(edge) change.
##  Therefore, edges can be held and processed later in batch.
##  Larger batch allows the graph to be processed as a whole from the start.
##  Super small batch (k=1) can be processed by recompute the neighbourhood of one vertex only.
##  Meanwhile, large batch could lead to severe accurarcy drop.
##  This script aims to investigate the balance between batch size, accuracy and computation.

import GCN as myGCN
import GCN_neighbor_loader as GCN_w_loader
from utils import *


def main():
    out_folder = "batch_accuracy"
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]  # Only non-mem-costly edge_index changes, keep a copy of edge index and reuse the same variable
    full_edge_index = data.edge_index.detach()

    use_loader = is_large(data) or args.use_loader

    model = myGCN.GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    ## Get available model
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

        ## different content starts from here
        maximum_batch_size = 0.005
        num_steps = 100
        num_edges = dataset[0].num_edges
        num_nodes = dataset[0].num_nodes
        batch_sizes = np.linspace(0, maximum_batch_size*num_edges, num_steps)
        batch_sizes = batch_sizes.astype(int)  # round off to int
        batch_sizes = np.unique(batch_sizes)  # avoid repeated computation in case the graph is too small and batch_size do not increase
        false_per_batch = np.zeros(batch_sizes.shape)
        for i, batch_size in enumerate(batch_sizes):
            # choose edges in batch by args.distribution
            _, initial_edges = edge_remove(full_edge_index.numpy(), batch_size, args.distribution,
                                                      data.is_directed())
            initial_edges = torch.from_numpy(initial_edges).to(device)  # from numpy to torch tensor
            data.edge_index = initial_edges  # todo: 不是很确定这里改变了edge_index后，会不会产生isolated node，进而影响后面的判断（by index)？
            if not use_loader :
                acc = myGCN.test_all(model, data)  # dump the whole graph as input, no masking
            else:  # todo: change to use data loader for testing. No need for test all?
                loader = data_loader(data, num_layers=2, num_neighbour_per_layer=-1,
                                     separate=False)  # load all neighbours
                # todo: 这里应该去看Hub Node 还是看全部的node？
                acc = GCN_w_loader.test(model, loader)

            false_per_batch[i] = acc * num_nodes

        fig, ax = plt.subplots()
        ax.plot(batch_sizes, false_per_batch, 'bo')
        plt.hlines(false_per_batch[0], batch_sizes[0]-5, batch_sizes[-1]+5, colors='tomato', linestyles='dashdot')
        ax.grid()
        ax.set_xlabel('batch_size')  # Add an x-label to the axes.
        ax.set_ylabel('false classification')  # Add a y-label to the axes.
        ax.set_title(f"{args.dataset}, {args.aggr}, {num_steps} steps")  # Add a title to the axes.
        plt.savefig(osp.join("batch_accuracy", f"{args.dataset}_{args.aggr}_{num_steps}_steps.pdf"))
        with open(osp.join("batch_accuracy", f"acc_{args.dataset}_{args.aggr}_{num_steps}_steps.txt"), 'w') as f:
            str_acc = [str(x) for x in false_per_batch]
            line = "\t".join(str_acc)
            f.write(line)

        # remove redundant saved models, only save the one with the highest accuracy
        available_model.pop(index_best_model)
        clean(available_model)


if __name__ == '__main__':
    main()
