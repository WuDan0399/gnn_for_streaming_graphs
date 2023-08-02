#  NOTE: run with gnnEnv conda env!
#################################################################################
#  Original Code from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gcn.py
#  Data Loader from:
#  https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/to_hetero_mag.py
#################################################################################
import random

from tqdm import tqdm
from utils import *
from GCN import GCN
from torch_geometric.data import Data

# todo: 改成 {batch_size}_{id} directory里面存 edge_list before and after, timing_result before and after.
# 对于一个 directory 如果里面已经有了 ground truth files， skip for full graph inference.

@torch.no_grad()
def inference_for_intermediate_result(model, loader, save_dir:str = "", postfix: str = "") :
    model.eval()
    print("Using Neighbour Loader for Full Graph Inference")
    intermediate_result_each_layer = defaultdict(lambda: defaultdict(lambda: torch.empty((0))))

    # save out_per_layer and intermediate timing_result per layer
    if isinstance(loader, pyg.loader.neighbor_loader.NeighborLoader):
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch.batch_size
            _, _, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index)
            # change sage.py and gcn.py to only return the information of target nodes in batch
            for layer in batch_intermediate_result_per_layer :
                if len(intermediate_result_each_layer[layer]['a-']) != 0:
                    intermediate_result_each_layer[layer]['a-'] = torch.concat((intermediate_result_each_layer[layer]["a-"],
                                                        batch_intermediate_result_per_layer[layer]["a-"][:batch_size].cpu()))
                else:
                    intermediate_result_each_layer[layer]['a-'] = batch_intermediate_result_per_layer[layer]["a-"][:batch_size].cpu()

                if len(intermediate_result_each_layer[layer]['a']) != 0:
                    intermediate_result_each_layer[layer]['a'] = torch.concat((intermediate_result_each_layer[layer]["a"],
                                                        batch_intermediate_result_per_layer[layer]["a"][:batch_size].cpu()))
                else:
                    intermediate_result_each_layer[layer]['a'] = batch_intermediate_result_per_layer[layer]["a"][:batch_size].cpu()

    elif isinstance(loader, EgoNetDataLoader):
        for batch in tqdm(loader):
            batch = batch.to(device)
            _, _, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index, batch.batch, batch.ptr)
            for layer in batch_intermediate_result_per_layer :
                if len(intermediate_result_each_layer[layer]['a-']) != 0:
                    intermediate_result_each_layer[layer]['a-'] = torch.concat((intermediate_result_each_layer[layer]["a-"],
                                                        batch_intermediate_result_per_layer[layer]["a-"].cpu()))
                else:
                    intermediate_result_each_layer[layer]['a-'] = batch_intermediate_result_per_layer[layer]["a-"].cpu()

                if len(intermediate_result_each_layer[layer]['a']) != 0:
                    intermediate_result_each_layer[layer]['a'] = torch.concat((intermediate_result_each_layer[layer]["a"],
                                                        batch_intermediate_result_per_layer[layer]["a"].cpu()))
                else:
                    intermediate_result_each_layer[layer]['a'] = batch_intermediate_result_per_layer[layer]["a"].cpu()

    if save_dir != "":
        for layer in intermediate_result_each_layer:
            create_directory(osp.join(save_dir, layer))
            torch.save(intermediate_result_each_layer[layer]['a-'], osp.join(save_dir, layer, f"before_aggregation{postfix}.pt"))
            torch.save(intermediate_result_each_layer[layer]['a'], osp.join(save_dir, layer, f"after_aggregation{postfix}.pt"))

    return intermediate_result_each_layer


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)

    niters = int(1000 // args.perbatch)  # times for small batch deletion (initial state)
    print(f"Iterations: {niters}")
    # args = FakeArgs(dataset="Cora", aggr="min", perbatch=10, stream="mix", interval=50000)
    dataset = load_dataset(args)
    print_dataset(dataset)
    data = dataset[0]
    timing_sampler(data, args)

    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)
    print(f"Batch size for streaming graph: {batch_size}")

    # model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args)
    # model = load_available_model(model, args).to(device)

    for i in tqdm(range(niters)):
        out_folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream, f"batch_size_{batch_size}", str(i))
        create_directory(out_folder)

        # edge selection
        initial_edges, final_edges, inserted_edges, removed_edges = get_graph_dynamics(data.edge_index, batch_size, args.stream)
        torch.save(initial_edges, (osp.join(out_folder, "initial_edges.pt")))
        torch.save(final_edges, (osp.join(out_folder, "final_edges.pt")))
        if inserted_edges.shape[1] :
            torch.save(inserted_edges, (osp.join(out_folder, "inserted_edges.pt")))
        if removed_edges.shape[1] :
            torch.save(removed_edges, (osp.join(out_folder, "removed_edges.pt")))

        # fake input for debuging
        # out_folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
        #                       f"batch_size_{batch_size}", '7')
        # folder = osp.join("examples", "intermediate", "Cora", "min", 'add',"batch_size_1")
        # data_dir = '7'
        # initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt"))
        # final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
        # inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
        # removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))


        # # Final Result
        # data2 = Data(x=data.x, edge_index=final_edges)
        # data2.to(device)
        # loader = data_loader(data2, num_layers=2, num_neighbour_per_layer=-1, separate=False)
        # _ = inference_for_intermediate_result(model, loader, out_folder, "_final")  #
        #
        #
        # ## run inference for the initial graph (before edge adding).
        # data3 = Data(x=data.x, edge_index=initial_edges)
        # data3.to(device)
        # post_fix = "_initial"
        # loader2 = data_loader(data3, num_layers=2, num_neighbour_per_layer=-1, separate=False)
        # _ = inference_for_intermediate_result(model, loader2, out_folder, post_fix)


if __name__ == '__main__':
    main()