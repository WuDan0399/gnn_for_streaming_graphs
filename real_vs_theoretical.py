import torch
from tqdm import tqdm
from utils import *
from load_dataset import load_dataset
from GCN import GCN
import pickle


def get_direct_affected_tensor(inserted_edges: torch.tensor, removed_edges: torch.tensor) -> set:
    dest_add = torch.unique(inserted_edges[1, :])
    dest_rm = torch.unique(removed_edges[1, :])
    # Concatenate and find unique values across both tensors
    all_unique = torch.unique(torch.cat((dest_add, dest_rm)))
    # Convert to a set of integers
    direct_affected_nodes = set(all_unique.tolist())
    return direct_affected_nodes


@torch.no_grad()
def inference_for_intermediate_result(model, loader):
    model.eval()
    intermediate_result_each_layer = defaultdict(lambda: defaultdict(lambda: torch.empty((0))))
    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch.batch_size
        _, _, batch_intermediate_result_per_layer = model(batch.x, batch.edge_index)
        # change sage.py and gcn.py to only return the information of target nodes in batch
        for layer in batch_intermediate_result_per_layer:
            if len(intermediate_result_each_layer[layer]['a-']) != 0:
                intermediate_result_each_layer[layer]['a-'] = torch.concat((intermediate_result_each_layer[layer]["a-"],
                                                                            batch_intermediate_result_per_layer[layer][
                                                                                "a-"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a-'] = batch_intermediate_result_per_layer[layer]["a-"][
                                                              :batch_size].cpu()

            if len(intermediate_result_each_layer[layer]['a']) != 0:
                intermediate_result_each_layer[layer]['a'] = torch.concat((intermediate_result_each_layer[layer]["a"],
                                                                           batch_intermediate_result_per_layer[layer][
                                                                               "a"][:batch_size].cpu()))
            else:
                intermediate_result_each_layer[layer]['a'] = batch_intermediate_result_per_layer[layer]["a"][
                                                             :batch_size].cpu()
    return intermediate_result_each_layer


def intm_affected(model, data, edges, nlayer: int = 2, inserted_edges=None, removed_edges=None, init_in_edge_dict=None,
                  final_in_edge_dict=None, init_out_edge_dict=None, final_out_edge_dict=None) -> Tuple[dict, set]:
    if isinstance(inserted_edges, torch.Tensor):  # full graph, only use one dict
        direct_affected_nodes = get_direct_affected_tensor(inserted_edges, removed_edges)
    else:  # sampled graph
        direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
    dicts = [i for i in [init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict] if i != None]
    # total_fetched_nodes = affected_nodes_each_layer(dicts, direct_affected_nodes, depth=nlayer)
    # fetched_nodes = torch.LongTensor(list(total_fetched_nodes[nlayer]))
    total_affected_nodes = affected_nodes_each_layer(dicts, direct_affected_nodes, depth=nlayer - 1)
    affected_nodes = torch.LongTensor(list(total_affected_nodes[nlayer - 1]))
    print(f"total affected nodes: {len(affected_nodes)}")

    # data2 = data.clone()
    # data2.edge_index = edges
    print(f"former edges {data.edge_index.shape[1]}")
    data.edge_index = edges
    print(f"current edges {data.edge_index.shape[1]}")
    loader = data_loader(data, num_layers=nlayer, num_neighbour_per_layer=-1, separate=False,
                         input_nodes=affected_nodes)
    intm_raw = inference_for_intermediate_result(model, loader)
    # rename the keys for alignment
    intm = {
        it_layer: {
            "before": {
                affected_nodes[i].item(): value["a-"][i]
                for i in range(len(affected_nodes))
            }, "after": {
                affected_nodes[i].item(): value["a"][i]
                for i in range(len(affected_nodes))
            }, }
        for it_layer, value in intm_raw.items()
    }
    return intm, total_affected_nodes[nlayer - 1]  # id of affected nodes


if __name__ == '__main__':
    depth = 2
    use_sampled_graph = False

    create_directory(osp.join("examples", "theoretical"))

    # args = FakeArgs(dataset="products", aggr="min", perbatch=100,
    #                 stream="mix", model="GCN", save_int=True)
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    batch_size = int(args.perbatch)
    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None
    # for products and papers, too large
    num_sample = 5

    data = dataset[0]
    # edge_dict_full = to_dict(data.edge_index)  # for full graph
    f = open(osp.join("examples", "theoretical", f"2layerGCN_affected_vs_real_{args.dataset}_{batch_size}.txt"), "w")
    f.write(
        f"batch_size\t#layer\texample_id\tth_affected_nodes(sampled)\treal_affected_nodes(sampled)\tnot_sure_nodes(sampled)\n")

    if args.dataset == 'papers':
        model = GCN(dataset.num_features, 256, dataset.num_classes + 1, args).to(device)
    else:
        model = GCN(dataset.num_features, 256, dataset.num_classes, args).to(device)

    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)
    if len(available_model) == 0:  # no available model, train from scratch
        print("no trained model")
    else:
        model = load(model, available_model[0])

    folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                      f"batch_size_{batch_size}")
    entries = os.listdir(folder)
    data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))][
                   :num_sample]

    pbar = tqdm(data_folders)
    total_all_cases = []
    change_all_cases = []
    notsure_all_cases = []
    case_id = -1
    if use_sampled_graph:
        print("Use saved cases for sampled graphs")
    else:
        print("Use snapshots with randomly removed edges of full graph")
        all_edges = data.edge_index
        # Assuming your defaultdict is named my_defaultdict
        edge_dict_file_name = f'{args.dataset}_full_dict.pickle'
        if os.path.exists(edge_dict_file_name):
            with open(edge_dict_file_name, 'rb') as file:
                edge_dict_full = pickle.load(file)
        else:
            edge_dict_full = to_dict(data.edge_index)
            with open(edge_dict_file_name, 'wb') as file:
                pickle.dump(edge_dict_full, file)
        print("Successfully build/load the edge_index dict")

    # for data_dir in pbar:
    for data_dir in data_folders:
        # failed_cases = ["6", "2",]
        # if data_dir in failed_cases:
        #     continue
        # pbar.set_description('Processing ' + str(batch_size))
        try:
            if use_sampled_graph:
                print(f"Processing dir {data_dir}")
                case_id += 1
                print(f"Iteration {case_id}/{num_sample}")
                initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt"))  # for sampled graph
                final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
                inserted_edges, removed_edges = [], []
                if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")):
                    inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
                    inserted_edges = [(src.item(), dst.item()) for src, dst in
                                      zip(inserted_edges[0], inserted_edges[1])]
                if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")):
                    removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
                    removed_edges = [(src.item(), dst.item()) for src, dst in zip(removed_edges[0], removed_edges[1])]
                # Assuming your defaultdict is named my_defaultdict
                edge_dict_file_name = f'{args.dataset}_{data_dir}_dict.pickle'
                if os.path.exists(edge_dict_file_name):
                    with open(edge_dict_file_name, 'rb') as file:
                        init_out_edge_dict_sampled = pickle.load(file)
                else:
                    init_out_edge_dict_sampled = to_dict(initial_edges)
                    with open(edge_dict_file_name, 'wb') as file:
                        pickle.dump(init_out_edge_dict_sampled, file)
                print(f"Successfully build/load the init_out_edge_dict_sampled dict.")
                final_edge_dict_file_name = f'{args.dataset}_{data_dir}_final_dict.pickle'
                if os.path.exists(final_edge_dict_file_name):
                    with open(final_edge_dict_file_name, 'rb') as file:
                        final_out_edge_dict_sampled = pickle.load(file)
                else:
                    final_out_edge_dict_sampled = to_dict(final_edges)
                    with open(final_edge_dict_file_name, 'wb') as file:
                        pickle.dump(final_out_edge_dict_sampled, file)
                print(f"Successfully build/load the final_out_edge_dict_sampled dict.")
                direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
                intm_before, affected_before = intm_affected(model, data, initial_edges, 2, inserted_edges,
                                                             removed_edges,
                                                             init_out_edge_dict_sampled, final_out_edge_dict_sampled)
                del initial_edges
                intm_after, affected_after = intm_affected(model, data, final_edges, 2, inserted_edges, removed_edges,
                                                           init_out_edge_dict_sampled, final_out_edge_dict_sampled)
                del final_edges
                del init_out_edge_dict_sampled
                torch.cuda.empty_cache()

            else:  # use full graph, randomly add/remove edges
                indices = torch.randperm(all_edges.size(1))
                edges_before = all_edges[:, indices[:-batch_size // 2]]
                inserted_edges = all_edges[:, indices[-batch_size // 2:]]
                removed_edges = all_edges[:, indices[:batch_size // 2]]
                # Find unique values in each tensor
                direct_affected_nodes = get_direct_affected_tensor(inserted_edges, removed_edges)
                # total_affected = affected_nodes_each_layer([edge_dict_full],
                #                                                    direct_affected_nodes, depth=depth - 1, self_loop=True)
                intm_before, affected_before = intm_affected(model, data, edges_before, 2, inserted_edges,
                                                             removed_edges, edge_dict_full)
                del edges_before
                edges_after = all_edges[:, indices[batch_size // 2:]]
                intm_after, affected_after = intm_affected(model, data, edges_after, 2, inserted_edges, removed_edges,
                                                           edge_dict_full)
                del edges_after
        except torch.cuda.OutOfMemoryError:
            print(f"Out of memory error for sample {data_dir}")
            # Clear CUDA memory
            torch.cuda.empty_cache()
            continue

        total_affected = affected_before | affected_after
        total_nodes = len(total_affected)  # use same dict for before and after, so no need to get set union.
        changed = 0
        notsure = 0
        for node in total_affected:
            if node not in affected_before or node not in affected_after:
                notsure += 1  # not sure whether changed
            else:
                if not torch.all(torch.isclose(intm_before[f"layer{depth}"]["after"][node],
                                               intm_after[f"layer{depth}"]["after"][node], atol=1e-6)):
                    changed += 1
        print(f"Total/changed/notsure: {total_nodes}/{changed}/{notsure}")
        total_all_cases.append(total_nodes)
        change_all_cases.append(changed)
        notsure_all_cases.append(notsure)
        f.write(
            f"{batch_size}\t{depth}\t{case_id}\t{total_nodes}\t{changed}\t{notsure}\n")
    total_all_cases = np.array(total_all_cases)
    change_all_cases = np.array(change_all_cases)
    notsure_all_cases = np.array(notsure_all_cases)
    f.write(
        f"average real:theoretical = {np.mean(change_all_cases / total_all_cases)}, "
        f"with nosure nodes= {np.mean((notsure_all_cases + change_all_cases) / total_all_cases)}\n")
