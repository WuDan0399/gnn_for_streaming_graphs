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
from load_dataset import load_dataset

@torch.no_grad()
def inference_for_intermediate_result(model, loader, save_dir:str = "", postfix: str = "") :
    model.eval()
    print("Using Neighbour Loader for Full Graph Inference")
    intermediate_result_each_layer = defaultdict(lambda: defaultdict(lambda: torch.empty((0))))

    if isinstance(loader, pyg.loader.neighbor_loader.NeighborLoader):
        for batch in tqdm(loader):
            torch.cuda.empty_cache()
            batch.to(device)
            batch_size = batch.batch_size
            _, _, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index)
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

    niters = 100
    dataset = load_dataset(args)
    print_dataset(dataset)
    data = dataset[0]

    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)
    print(f"Batch size for streaming graph: {batch_size}")

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


if __name__ == '__main__':
    main()

