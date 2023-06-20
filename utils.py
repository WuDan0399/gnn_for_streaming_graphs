import argparse
import os
import os.path as osp
import re
from typing import Tuple, Union
import multiprocessing

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from functools import wraps
import time
import torch_geometric as pyg
from torch_geometric.datasets import Reddit, Planetoid, CitationFull, Yelp, AmazonProducts
from torch_geometric.loader import NeighborLoader
from torch_geometric.typing import EdgeType, InputNodes, OptTensor


np.random.seed(0)
torch.manual_seed(0)

root = os.getenv("DYNAMIC_GNN_ROOT")
# root = "/Users/wooden/PycharmProjects/GNNAccEst"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_arrays_with_shape_and_type(**kwargs):
    """
    For meature_time decorator, in case one of the argument is a tensor\array,
    the measure_time function will print the content of the tensor\array, which is messy.
    :param kwargs:  any function arguments that using measure_time decorator
    :return:  a modified argument dict with tensor\arrays replaced with a descriptive string
    """
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            kwargs[key] = f"numpy.ndarray, shape: {value.shape}, dtype: {value.dtype}"
        elif isinstance(value, torch.Tensor):
            kwargs[key] = f"torch.Tensor, shape: {value.shape}, dtype: {value.dtype}"
    return kwargs

# decorator
def measure_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) :
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} {args} {replace_arrays_with_shape_and_type(**kwargs)} Took {total_time:.4f} seconds')

        return result

    return timeit_wrapper


def save(model_state_dict, epoch, acc, name) -> None :
    torch.save(model_state_dict, osp.join("examples/trained_model", f"{name}_{epoch}_{acc:.3f}.pt"))


def load(model, file_name: str) :
    print(f"Loading model {file_name} ...")
    model_path = osp.join(root, "dynamic", "examples", "trained_model", file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def clean(files: list) -> None :
    if len(files) == 0 :
        print("Clean directory, no useless models.")
    else :
        print(f"Useless models found, cleaning {files}.")
        for file in files :
            os.remove(osp.join(root, "dynamic", "examples", "trained_model", file))


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None) -> None :
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h) :
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None :
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {accuracy["train"] * 100:.2f}% \n'
                        f' Validation Accuracy: {accuracy["val"] * 100:.2f}%'),
                       fontsize=16)
    else :
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


def add_mask(data: pyg.data.Data) -> None :
    if not hasattr(data, "train_mask") :  # add train\test\validation masks if not any
        train_ratio, test_ratio, validation_ratio = 0.75, 0.1, 0.15
        rand_choice = np.random.rand(len(data.y))
        data.train_mask = torch.tensor(
            [True if x < train_ratio else False for x in rand_choice], dtype=torch.bool)
        data.test_mask = torch.tensor([True if x >= train_ratio and x <
                                               train_ratio + test_ratio else False for x in rand_choice],
                                      dtype=torch.bool)
        data.val_mask = torch.tensor([True if x >= train_ratio +
                                              test_ratio else False for x in rand_choice], dtype=torch.bool)


def data_loader(data, input_nodes:OptTensor = None,
                num_layers:int = 2,
                num_neighbour_per_layer:int = -1,
                separate: bool = True,
                persistent_workers: bool = True) :
    if multiprocessing.cpu_count() <10:
        kwargs = {'batch_size': 4, 'num_workers': 2, 'persistent_workers': persistent_workers}
    else:
        kwargs = {'batch_size': 16, 'num_workers': 4, 'persistent_workers': persistent_workers}
    # print(kwargs)
    if separate :
        train_loader = NeighborLoader(data, num_neighbors=[num_neighbour_per_layer] * num_layers, shuffle=True,
                                      input_nodes=data.train_mask, **kwargs)
        val_loader = NeighborLoader(data, num_neighbors=[num_neighbour_per_layer] * num_layers,
                                    input_nodes=data.val_mask, **kwargs)
        test_loader = NeighborLoader(data, num_neighbors=[num_neighbour_per_layer] * num_layers,
                                     input_nodes=data.test_mask, **kwargs)
        return train_loader, val_loader, test_loader
    else :
        return NeighborLoader(data, input_nodes= input_nodes, num_neighbors=[num_neighbour_per_layer] * num_layers,
                              **kwargs)  # by default use all nodes as input_nodes


def print_dataset(dataset) -> None :
    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')


def print_data(data: pyg.data.Data) -> None :
    print(data)
    print('==============================================================')
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {(2 * data.num_edges) / data.num_nodes:.2f}')
    if hasattr(data, "train_mask") :
        print(f'Number of training nodes: {data.train_mask.sum()}')
        print(f'Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
    print(f'Contains self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def general_parser(parser: argparse.ArgumentParser) -> argparse.Namespace :
    parser.add_argument("-d", '--dataset', type=str, default='yelp')

    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--aggr', type=str, default="min")
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
    parser.add_argument('--wandb', action='store_true', help='Track experiment')
    parser.add_argument("--distribution", default="random",
                        type=str, help="distribution of edges in one batch (random/burst/combine)")
    parser.add_argument("-l", "--nlayers", default=5,
                        type=int, help="number of layers")
    parser.add_argument("-i", "--initial", default=70.0, type=float,
                        help="percentage of edges loaded at the begining. [0.0, 100.0]")
    parser.add_argument("-pb", "--perbatch", default=0.1, type=float,
                        help="percentage of edges loaded per batch. [0.0, 100.0]")
    parser.add_argument("-nb", "--numbatch",
                        default=50, type=int, help="number of batches")
    parser.add_argument("--range", default="full", type=str, help="range of inference [full/affected/mono]")
    parser.add_argument("--use_loader", action='store_true', help="whether to use data loader")
    parser.add_argument("--save_int", action='store_true', help="whether to save intermediate result during inference")
    args = parser.parse_args()
    return args


def load_dataset(args: argparse.Namespace) :
    if args.dataset == "Cora" :  # class
        # 2,708,  10,556,  1,433 , 7
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "Cora")
    elif args.dataset == "PubMed" :  # class
        dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "PubMed")
    elif args.dataset == 'reddit' :  # class
        dataset = Reddit(osp.join(root, "datasets", "Reddit"))
    elif args.dataset == "cora" :  # class
        dataset = CitationFull(osp.join(root, "datasets", "CitationFull"), "Cora")
    elif args.dataset == "yelp" :  # tasks Non one-hot
        dataset = Yelp(osp.join(root, "datasets", "Yelp"))
    elif args.dataset == "amazon" :  # class, but on-hot representation
        dataset = AmazonProducts(osp.join(root, "datasets", "Amazon"))
        # for data in dataset : # Actually only one data
        #     data.y = data.y.type(torch.float32)
        #     data.y = data.y.argmax(dim=-1)
    else:
        print("No such dataset. Available: Cora/cora/PubMed/reddit/yelp/amazon")
    return dataset


def load_available_model(args: argparse.Namespace) :
    available_model = []
    name_prefix = f"{args.dataset}_GCN_{args.aggr}_0.5dropout"
    for file in os.listdir("examples/trained_model") :
        if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
            available_model.append(file)


def affected_nodes_by_layer(graph: nx.DiGraph, srcs, depth) :
    affected = []
    for src in srcs :
        bfs_tree = nx.bfs_tree(graph, source=src, depth_limit=depth)
        nodes = list(bfs_tree.nodes)
        affected = np.union1d(affected, nodes)
    return affected


def unique_and_location(array:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # generated by chatgpt
    # dictionary to store indices
    indices_dict = {}

    for idx, element in np.ndenumerate(array) :
        if element not in indices_dict :
            indices_dict[element] = [idx[0]]
        else :
            indices_dict[element].append(idx[0])

    # get unique elements and their locations
    unique_elements = np.array(list(indices_dict.keys()))
    element_locations = list(indices_dict.values())

    return unique_elements, element_locations


def burst_sampler(full_edges: np.ndarray, num_edge_changed) -> np.ndarray :
    # Return the indexes for the topest hub node.
    # Sort hub nodes and in-edge indexes by degree and return the top `num_edge_changed` indexes.

    _, index_in_edges = unique_and_location(full_edges[1])  # no need for id, only need degree and location
    # sort index_in_edges the in-degree (len of index_in_edges), in descending order
    index_in_edges.sort(key=lambda x : len(x), reverse=True)  # list of list
    flatten_sorted_index_in_edges = [item for sublist in index_in_edges for item in sublist]

    return flatten_sorted_index_in_edges[:num_edge_changed]


def edge_remove(full_edges: np.ndarray, batch_size:int, distribution: str, directed: bool = False) -> Tuple[
    np.ndarray, np.ndarray] :
    # ATTENTION: RETURN VALUE HAVE DIFFERENT SHAPE!!!!
    # batch size refers to number of undirected edges. If directed edges are removed, remove double the size
    num_edges = full_edges.shape[1]
    num_edge_changed = 2 * batch_size if directed else batch_size  # if undirected, sample a half and add back the other direction.

    if distribution == "random" :
        index_added_edges = np.random.choice(num_edges, num_edge_changed, replace=False)
    elif distribution == "burst" :
        index_added_edges = burst_sampler(full_edges, num_edge_changed)
    elif distribution == "combine" :
        print("Combined sampling is not implemented yet. Use random instead")
        index_added_edges = np.random.choice(num_edges, num_edge_changed, replace=False)

    if not directed :  # if directed, the other direction is added.
        index_augmented = []
        for index in index_added_edges :
            the_other_direction = np.flip(full_edges[:, index])
            for x in range(full_edges.shape[1]) :
                if np.all(full_edges[:, x] == the_other_direction) :
                    index_augmented.append(x)
        index_augmented = np.sort(np.concatenate((index_added_edges, np.array(index_augmented))))

    sample_edges = np.transpose(full_edges[:, index_added_edges])  # TRANSPOSED!!!!!!
    edges_one_batch_missing = full_edges[:, np.setdiff1d(np.arange(num_edges), index_augmented)]

    return sample_edges, edges_one_batch_missing


def is_large(data) :
    return True if data.num_nodes > 50000 else False


def concate_by_id(full: np.ndarray, batch: np.ndarray, position: np.ndarray) -> np.ndarray :
    batch_size = len(batch)
    full_size = len(full)

    if np.array_equal(position, np.arange(full_size, full_size + batch_size)) :
        # directly append to the tail
        full = np.concatenate((full, batch), axis=0) if len(full) != 0 else batch

    else :  # extend `full` if necessary and insert `batch`
        if np.max(position) >= full_size :
            extended = np.zeros((np.max(position) + 1, full.shape[1]))
            extended[:full.shape[0]] = full
            full = extended
        # insert 'batch' into 'full' at the positions specified by 'position'
        full[position] = batch

    return full

def create_directory(path):  # by chatgpt
    try:
        os.makedirs(path)
        print(f"Directory '{path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{path}' already exists.")

def to_dict(edges: torch.Tensor):
    edge_dict = {}
    for i in range(edges.size(1)) :
        source_node = edges[0, i].item()
        dest_node = edges[1, i].item()

        if source_node not in edge_dict :
            edge_dict[source_node] = []

        edge_dict[source_node].append(dest_node)
    return edge_dict