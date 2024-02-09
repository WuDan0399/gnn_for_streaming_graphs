from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric.datasets import Reddit, Planetoid, CitationFull, Yelp, AmazonProducts
from utils import *

class RandFeatDataset(InMemoryDataset):
    def __init__(self, dir, file_name, dataset_name, feature_length=1024,
                 transform=None, pre_transform=None, skiprows=2, has_time=True
                 ):
        self.dir = dir
        self.file_name = file_name
        self.dataset_name = dataset_name
        self.feature_length = feature_length
        self.skiprows = skiprows
        self.has_time = has_time
        super(RandFeatDataset, self).__init__(dir, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.file_name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []

        if self.file_name.endswith('.csv'):
            import pandas as pd
            df = pd.read_table(osp.join(self.dir, self.file_name), delim_whitespace=True)
            if '/' in str(df['time'].iloc[0]):
                df['time'] = pd.to_datetime(df['time']).astype(int) // 10 ** 9  
            data = df.to_numpy()
        else:
            data = np.loadtxt(fname=osp.join(self.dir, self.file_name), skiprows=self.skiprows)

        edge_index = torch.tensor(data[:, :2], dtype=torch.long).t().contiguous()
        unique_nodes = torch.unique(edge_index)
        mapping = {node.item(): i for i, node in enumerate(unique_nodes)}
        edge_index = torch.tensor([[mapping[node.item()] for node in edge] for edge in edge_index],
                                            dtype=torch.long)
        num_nodes = len(unique_nodes)
        embedding = torch.nn.Embedding(num_nodes, self.feature_length) 
        x = embedding(torch.range(start=0, end=num_nodes-1, dtype=torch.long))

        if self.has_time:
            edge_weight = torch.tensor(data[:, 2], dtype=torch.float)
            edge_time = torch.tensor(data[:, 3], dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight, time=edge_time)
        else:
            data = Data(x=x, edge_index=edge_index)
        data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



def generate_snapshots(data:Data, dataset_name:str, stream:str, args:argparse.Namespace):
    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    timing_sampler(data, args)

    if hasattr(data, "time"):
        exist = True
        for batch_size in batch_sizes:
            if not osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", "add",
                                  f"batch_size_{batch_size}", str(0))):
                exist = False
                break
        if exist:
            return
        idx_sorted_by_time = torch.argsort(data.time)
        sorted_edge_index = data.edge_index[:, idx_sorted_by_time]
        for batch_size, num_sample in zip(batch_sizes, num_samples):
            if osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", "add",
                                       f"batch_size_{batch_size}", str(0))):
                continue
            for it_sample in range(num_sample):
                folder = osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", "add",
                                  f"batch_size_{batch_size}", str(it_sample))
                create_directory(folder)
                final_edegs = sorted_edge_index[:, :-batch_size*(it_sample+1)]
                initial_edges = sorted_edge_index[:, :-batch_size*(it_sample)]
                inserted_edges = sorted_edge_index[:, -batch_size*(it_sample+1):-batch_size*(it_sample)]
                torch.save(final_edegs, osp.join(folder, "final_edges.pt"))
                torch.save(initial_edges, osp.join(folder, "initial_edges.pt"))
                torch.save(inserted_edges, osp.join(folder, "inserted_edges.pt"))
    else:
        exist = True
        for batch_size in batch_sizes:
            if not osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", "add",
                                       f"batch_size_{batch_size}", str(0))):
                exist = False
                break
        if exist:
            return

        for batch_size, num_sample in zip(batch_sizes, num_samples):
            if osp.exists(osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", stream,
                                       f"batch_size_{batch_size}", str(0))):
                continue

            print(f"Generating graph topology snapshots for batch size {batch_size}, sample {num_sample}.")
            for i in range(num_sample):
                out_folder = osp.join(root, "dynamic", "examples", "intermediate", dataset_name, "min", stream,
                                  f"batch_size_{batch_size}", str(i))
                create_directory(out_folder)
                # edge selection
                initial_edges, final_edges, inserted_edges, removed_edges = get_graph_dynamics(data.edge_index,
                                                                                               batch_size,
                                                                                               stream)
                torch.save(initial_edges, (osp.join(out_folder, "initial_edges.pt")))
                torch.save(final_edges, (osp.join(out_folder, "final_edges.pt")))
                if inserted_edges.shape[1]:
                    torch.save(inserted_edges, (osp.join(out_folder, "inserted_edges.pt")))
                if removed_edges.shape[1]:
                    torch.save(removed_edges, (osp.join(out_folder, "removed_edges.pt")))
    print("generate snapshots, end.")

def one_hot_fake_label(dataset: pyg.data.Data, max_degree: int = 1000) -> list:
    new_dataset = []
    for data in dataset:
        node_degree = degree(data.edge_index[0]) 
        real_max_degree = torch.max(node_degree).item()
        node_degree = torch.clamp(
            node_degree, max=max_degree - 1
        ) 
        unique_degree, indices = torch.unique(
            node_degree, return_inverse=True
        )
        one_hot_size = max(
            len(unique_degree), min(int(real_max_degree), max_degree)
        )
        new_x = torch.zeros(
            (len(node_degree), one_hot_size), dtype=torch.float
        )
        new_x[torch.arange(len(node_degree)), indices] = 1
        new_y = torch.randint(0, 2, data.y.shape[:1], dtype=torch.long)
        data.x = new_x
        data.y = new_y
        new_dataset.append(data)
    return new_dataset


def load_dataset(args: argparse.Namespace, transform: Optional[Callable] = None):
    from ogb.nodeproppred import PygNodePropPredDataset
    if args.dataset == "Cora":
        if transform is not None:
            print("[Error] Cora dataset cannot support transform")
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "Cora")
    elif args.dataset == "PubMed":
        if transform is not None:
            print("[Error] PubMed dataset cannot support transform")
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "PubMed")
    elif args.dataset == "reddit": 
        if transform is not None:
            print("[Error] Reddit dataset cannot support transform")
        dataset = Reddit(osp.join(root, "datasets", "Reddit"))
    elif args.dataset == "cora":
        if transform is not None:
            print("[Error] Cora dataset cannot support transform")
        dataset = CitationFull(
            osp.join(root, "datasets", "CitationFull"), "Cora")
    elif args.dataset == "yelp": 
        if transform is not None:
            print("[Error] Yelp dataset cannot support transform")
        dataset = Yelp(osp.join(root, "datasets", "Yelp"))
    elif args.dataset == "amazon":
        if transform is not None:
            print("[Error] Amazon dataset cannot support transform")
        dataset = AmazonProducts(osp.join(root, "datasets", "Amazon"))
        for data in dataset: 
            data.y = data.y.argmax(dim=-1).float()
    elif args.dataset == "products":
        dataset = PygNodePropPredDataset(name="ogbn-products", root=osp.join(root, "datasets"))
    elif args.dataset == "papers":
        dataset = PygNodePropPredDataset(name="ogbn-papers100M", root=osp.join(root, "datasets"))
    elif args.dataset == "wiki": 
        dataset = RandFeatDataset(dir=osp.join(root, "datasets", "wikipedia_link_en"),
                                  file_name="out.wikipedia_link_en", dataset_name="wiki", transform=transform,
                                  skiprows=1, has_time=False)
    else:
        if args.dataset == 'uci':
            dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "opsahl-ucsocial"),
                                      file_name="out.opsahl-ucsocial", dataset_name="uci", transform = transform)
        elif args.dataset == "dnc":
            dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "dnc-temporalGraph"),
                                      file_name="out.dnc-temporalGraph", dataset_name="dnc", transform = transform)
        elif args.dataset == "epi":
            dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "epinions"),
                                      file_name="user_rating.csv", dataset_name="epi", transform = transform)
        else:
            print("No such dataset. Available: Cora/cora/PubMed/reddit/yelp/uci/dnc/epi/products/papers")

    if args.binary:
        dataset = one_hot_fake_label(dataset, max_degree=500)

    generate_snapshots(dataset[0], args.dataset, args.stream, args)  # generate snapshots for the first(only) graph

    return dataset

def load_dataset_dgl(args: argparse.Namespace, transform: Optional[Callable] = None):
    from ogb.nodeproppred import DglNodePropPredDataset
    from dgl import AddSelfLoop
    from dgl.data import CoraGraphDataset, CoraFullDataset, PubmedGraphDataset, RedditDataset, YelpDataset, \
        WikiCSDataset, RedditDataset
    from dgl.data import AsNodePredDataset
    transform = (
        AddSelfLoop()
   ) 
    if args.dataset == "Cora": 
        dataset = CoraGraphDataset(raw_dir=osp.join(root, "datasets"), transform=transform)
    elif args.dataset == "PubMed":
        dataset = PubmedGraphDataset(raw_dir=osp.join(root, "datasets"), transform=transform)
    elif args.dataset == "reddit":
        dataset = RedditDataset(raw_dir=osp.join(root, "datasets"), transform=transform)
    elif args.dataset == "cora": 
        dataset = CoraFullDataset(raw_dir=osp.join(root, "datasets"), transform=transform)
    elif args.dataset == "yelp":
        dataset = YelpDataset(raw_dir=osp.join(root, "datasets"), transform=transform)
    elif args.dataset == "amazon":
       pass
    elif args.dataset == "products": 
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=osp.join(root, "datasets")), transform=transform)
    elif args.dataset == "papers":
        dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M", root=osp.join(root, "datasets")), transform=transform)
    return dataset



if __name__ == '__main__':
    dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "opsahl-ucsocial"),
                              file_name="out.opsahl-ucsocial", dataset_name="uci")
    dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "dnc-temporalGraph"),
                              file_name="out.dnc-temporalGraph", dataset_name="dnc")
    dataset = RandFeatDataset(dir=osp.join(root, "datasets", "dynamic_datasets", "epinions"),
                              file_name="user_rating.csv", dataset_name="epi")
