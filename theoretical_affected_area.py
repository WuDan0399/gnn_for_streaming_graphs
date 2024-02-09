from utils import *
from load_dataset import load_dataset
from tqdm import tqdm, trange

if __name__ == '__main__':
    max_layer = 4
    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    create_directory(osp.join("examples", "theoretical"))
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    data = dataset[0]

    num_samples = [10]
    batch_sizes = [1]
    for batch_size, num_sample in zip(batch_sizes, num_samples):
        affected_ratio_sampled = [[] for _ in range(max_layer)] 
        affected_ratio_full = [[] for _ in range(max_layer)]

        folder = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                                   f"batch_size_{batch_size}")
        entries = os.listdir(folder)
        data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))][:num_sample]

        pbar = tqdm(data_folders)
        for data_dir in pbar:
            pbar.set_description('Processing ' + str(batch_size))
            inserted_edges, removed_edges = [], []
            if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")) :
                inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
                inserted_edges = [(src.item(), dst.item()) for src, dst in zip(inserted_edges[0], inserted_edges[1])]
            if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")) :
                removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
                removed_edges = [(src.item(), dst.item()) for src, dst in zip(removed_edges[0], removed_edges[1])]

            final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
            final_out_edge_dict_sampled = to_dict_wiz_cache(final_edges, osp.join(folder,data_dir), f'final_out_edge_dict.pickle')
            del final_edges
            initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt")) 
            init_out_edge_dict_sampled = to_dict_wiz_cache(initial_edges, osp.join(folder,data_dir), f'init_out_edge_dict.pickle')
            del initial_edges

            direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
            affected_nodes_sampled = affected_nodes_each_layer([init_out_edge_dict_sampled,
                                                                final_out_edge_dict_sampled],
                                                               direct_affected_nodes, depth=max_layer, self_loop=True)
            for layer in affected_nodes_sampled.keys():
                print(f"{layer}: {len(affected_nodes_sampled[layer])}")

