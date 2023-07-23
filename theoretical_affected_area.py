from utils import *

if __name__ == '__main__':
    depths = [2] #[2,3,4,5]
    create_directory(osp.join("examples", "theoretical"))
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]
    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)
    print(f"Batch size for streaming graph: {batch_size}")
    folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                               f"batch_size_{batch_size}")

    entries = os.listdir(folder)
    data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))]
    affected_ratio = [[] for _ in depths]

    # Multiple different data (initial state info and final state info) directory
    for data_dir in data_folders :
        print(osp.join(folder, data_dir))
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
        final_in_edge_dict = to_dict(final_edges[[1, 0], :])

        direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])


        for i, nlayer in enumerate(depths):
            affected_nodes = affected_nodes_each_layer([init_out_edge_dict, final_out_edge_dict],
                                                       direct_affected_nodes, depth=nlayer - 1)
            affected_ratio[i].append(len(affected_nodes[nlayer-1])/data.num_nodes)

    for i, nlayer in enumerate(depths):
        np.save(osp.join("examples", "theoretical", f"affected_{args.dataset}_{batch_size}_{nlayer}.npy"), affected_ratio[i])
