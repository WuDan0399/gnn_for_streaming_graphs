import subprocess
import argparse
import os
import os.path as osp
import re
import time
from functools import wraps
from itertools import groupby
from operator import attrgetter
from collections import defaultdict
from typing import Tuple, Union, List, Dict, Optional, Callable

import numpy as np
import torch
import torch_geometric as pyg
from torch_geometric.loader import NeighborLoader, NeighborSampler
from torch_geometric.typing import OptTensor
from torch_geometric.utils import degree
from torch.utils.data import DataLoader
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data, Batch
from configs import *
import pickle

np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=10)

print_data_loader = True


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class FakeArgs:
    def __init__(
        self,
        dataset="Cora",
        model="GCN",
        use_gdc=False,
        patience=2,
        aggr="min",
        save_int=True,
        hidden_channels=256,
        epochs=10,
        perbatch=1,
        stream="add",
        interval=5000000,
        it=0,
        binary=False,
        mt=0,
        range="affected",
        loader="default",
    ):
        self.dataset = dataset
        self.model = model
        self.use_gdc = use_gdc
        self.aggr = aggr
        self.save_int = save_int
        self.hidden_channels = hidden_channels
        self.perbatch = perbatch
        self.stream = stream
        self.interval = interval
        self.it = it
        self.binary = binary
        self.patience = patience
        self.epochs = epochs
        self.mt = mt
        self.range = range
        self.loader = loader

class EgoNetDataLoader(DataLoader):
    def __init__(
        self,
        data,
        node_indices,
        batch_size,
        k=5,
        num_workers=0,
        persistent_workers=False,
    ):
        self.data = data
        self.node_indices = node_indices
        self.batch_size = batch_size
        self.k = k
        super().__init__(
            node_indices,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    def collate_fn(self, batch_node_indices):
        batch_data_list = []
        for node_idx in batch_node_indices:
            subset, edge_index, _, _ = k_hop_subgraph(
                node_idx.item(),
                self.k,
                self.data.edge_index,
                relabel_nodes=True,
                num_nodes=None,
                flow="target_to_source",
                directed=False,
            )
            x = self.data.x[subset]
            y = self.data.y[node_idx]
            batch_data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return Batch.from_data_list(batch_data_list)


def group_task_queue(task_q: list) -> dict:
    # groupby returns consecutive keys, so need to sort first
    result = {}
    task_q_sorted = sorted(
        task_q, key=attrgetter("dst", "op")
    )  # sorting by 'dst' first, then by 'op'
    for dst, tasks_in_dst in groupby(task_q_sorted, key=attrgetter("dst")):
        result[dst] = {
            op: [(task.src, task.msg) for task in tasks]
            for op, tasks in groupby(tasks_in_dst, key=attrgetter("op"))
        }
    return result


def print_task_queue(task_q):
    for task in task_q:
        print(f"<{task.op}, {task.src}, {task.dst}, {task.msg.shape}>")


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


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()  # wait for warm-up to finish


# decorator
def measure_time(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(
            f"Function {func.__name__} {args} {replace_arrays_with_shape_and_type(**kwargs)} Took {total_time:.4f} seconds"
        )

        return result

    return timeit_wrapper


def save(model_state_dict, epoch, acc, name) -> None:
    torch.save(
        model_state_dict,
        osp.join("examples/trained_model", f"{name}_{epoch}_{acc:.3f}.pt"),
    )


def load(model, file_name: str):
    print(f"Loading model {file_name} ...")
    model_path = osp.join(root, "dynamic", "examples",
                          "trained_model", file_name)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def clean(files: list) -> None:
    if len(files) == 0:
        print("Clean directory, no useless models.")
    else:
        print(f"Useless models found, cleaning {files}.")
        for file in files:
            os.remove(osp.join(root, "dynamic",
                      "examples", "trained_model", file))


def load_tensors_to_dict(root: str, skip: int = 5, postfix: str = "_initial.pt"):
    result = {}
    for path, dirs, files in os.walk(root):
        if files:  # check if there are any files in the current directory
            path_parts = path.split(os.sep)  # get parts of the path
            sub_dict = result
            for part in path_parts[skip:]:  # skip unnecessary directory
                if part not in sub_dict:
                    sub_dict[part] = {}
                sub_dict = sub_dict[part]
            for file in files:
                if file.endswith(
                    postfix
                ):  # check if the file is a .pt file and the correpsonding postfix
                    name_parts = file[:-3].split(
                        "_"
                    )  # we remove the extension and split the filename
                    sub_dict[name_parts[0]] = torch.load(
                        os.path.join(path, file))
    return result


def add_mask(data: pyg.data.Data) -> None:
    if not hasattr(data, "train_mask"):  # add train\test\validation masks if not any
        print("Add mask to the dataset.")
        if data.num_nodes < 1000000:  # 5M
            train_ratio, test_ratio, validation_ratio = 0.75, 0.1, 0.15  # fully split the dataset
        else:
            shrink_ratio = 1000000 / data.num_nodes  # capped at 5M nodes
            print(f"Data set too large, to save training time, use {shrink_ratio} of the nodes for training, roughly 5M nodes.")
            train_ratio, test_ratio, validation_ratio = 0.75 * shrink_ratio, 0.1 * shrink_ratio, 0.15 * shrink_ratio  # fully split the dataset
        rand_choice = np.random.rand(len(data.y))
        data.train_mask = torch.tensor(
            [True if x < train_ratio else False for x in rand_choice], dtype=torch.bool
        )
        data.test_mask = torch.tensor(
            [
                True if x >= train_ratio and x < train_ratio + test_ratio else False
                for x in rand_choice
            ],
            dtype=torch.bool,
        )
        data.val_mask = torch.tensor(
            [True if x >= train_ratio + test_ratio and x < train_ratio + test_ratio + validation_ratio
             else False for x in rand_choice],
            dtype=torch.bool,
        )


def data_loader(
    data,
    input_nodes=None,
    num_layers: int = 2,
    num_neighbour_per_layer: int = -1,
    separate: bool = True,
    persistent_workers: bool = False,
    loader_type: str = "default"
):
    # kwargs = {'batch_size' : 8, 'num_workers' : 1, 'persistent_workers' : persistent_workers}
    kwargs = loader_configs
    kwargs["persistent_workers"] = persistent_workers
    global print_data_loader
    if loader_type == "default":
        if separate:
            if print_data_loader:
                print("Using default data loader: pyg.NeighborLoader, Train Mode")
                print_data_loader = False
            train_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                shuffle=True,
                input_nodes=data.train_mask,
                **kwargs,
            )
            val_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                input_nodes=data.val_mask,
                **kwargs,
            )
            test_loader = NeighborLoader(
                data,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                input_nodes=data.test_mask,
                **kwargs,
            )
            return train_loader, val_loader, test_loader
        else:
            if print_data_loader:
                print("Using default data loader: pyg.NeighborLoader, Full Graph Test Mode")
                print_data_loader = False
            return NeighborLoader(
                data,
                input_nodes=input_nodes,
                num_neighbors=[num_neighbour_per_layer] * num_layers,
                **kwargs,
            )  # by default use all nodes as input_nodes
    elif loader_type == "quiver":
        if separate:
           raise NotImplementedError
        else:
            if print_data_loader:
                print("Using quiver data loader: quiver.pyg.GraphSageSampler, Full Graph Test Mode")
                print_data_loader = False
            seed_loader = torch.utils.data.DataLoader(torch.arange(0, data.num_nodes-1).long(),
                                                       batch_size=loader_configs["batch_size"])
            return seed_loader


def print_dataset(dataset) -> None:
    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")


def print_data(data: pyg.data.Data) -> None:
    print(data)
    print("==============================================================")
    # Gather some statistics about the graph.
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {(2 * data.num_edges) / data.num_nodes:.2f}")
    if hasattr(data, "train_mask"):
        print(f"Number of training nodes: {data.train_mask.sum()}")
        print(
            f"Training node label rate: {int(data.train_mask.sum()) / data.num_nodes:.2f}"
        )
    print(f"Contains isolated nodes: {data.has_isolated_nodes()}")
    print(f"Contains self-loops: {data.has_self_loops()}")
    print(f"Is undirected: {data.is_undirected()}")


def general_parser(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument("-d", "--dataset", type=str, default="yelp")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--aggr", type=str, default="min")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--use_gdc", action="store_true", help="Use GDC")
    parser.add_argument("--mode",type=str, default="mixed", help="for dgl train mode")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use one-hot encoding of node degree as node attribute, fake labels of ego networks are generated",
    )
    parser.add_argument(
        "--distribution",
        default="random",
        type=str,
        help="distribution of edges in one batch (random/burst/combine)",
    )
    parser.add_argument("-l", "--nlayers", default=5,
                        type=int, help="number of layers")
    parser.add_argument(
        "-i",
        "--initial",
        default=70.0,
        type=float,
        help="percentage of edges loaded at the begining. [0.0, 100.0]",
    )
    parser.add_argument(
        "-pb",
        "--perbatch",
        default=100,
        type=float,
        help="percentage of edges loaded per batch. [0.0, 100.0]",
    )
    parser.add_argument(
        "--interval",
        default=5000000,
        type=float,
        help="percentage of the latest proportion/number of edges for inference. [0.0, 1.0] or int. Default: 500k edges",
    )
    parser.add_argument(
        "-nb", "--numbatch", default=50, type=int, help="number of batches"
    )
    parser.add_argument(
        "--range",
        default="full",
        type=str,
        help="range of inference [full/affected/mono]",
    )
    parser.add_argument(
        "--use_loader", action="store_true", help="whether to use data loader"
    )
    parser.add_argument(
        "--save_int",
        action="store_true",
        help="whether to save intermediate timing_result during inference",
    )
    parser.add_argument(
        "--stream",
        default="mix",
        type=str,
        help="how edges changes, insertion or deletion or both [add/delete/mix]",
    )
    parser.add_argument(
        "--loader",
        default="default",
        type=str,
        help="loader to use, pyg.Neighborloader by default [default/quiver]",
    )
    parser.add_argument(
        "--mt",
        default=0,
        type=int,
        help="threads in multi-threading, 0 for single thread (default)",
    )
    parser.add_argument(
        "--id",
        default=-1,
        type=int,
        help="id of the sample to be inferred, only used in multi-threading",
    )
    args = parser.parse_args()
    return args


def timing_sampler(data: pyg.data.Data, args):
    # todo: check wther runable on GPU -> do complex operation, should i offload then upload to cuda?
    """
    Simulate the streaming graphs. Generate random time information, get the lastest 'interval' edges of data for gnn.
    :param data: PyG.data
    :param args: command arguments
    :param interval: float or int. in (0,1) -> ratio. >1 -> exact size of num_edges
    :return: the lastest 'interval' edges for data.
    """
    if isinstance(data, pyg.data.Data):
        num_edges = data.num_edges
        edge_index = data.edge_index
    else: # assume dgl graph
        num_edges = data.num_edges()
        edge_index = torch.stack(data.edges())
    print(f"Sampling {args.interval} edges for {args.dataset}")
    folder_path = os.path.join("examples", "creation_time")
    create_directory(folder_path)

    if args.interval > 1:
        # (opt1) roughly interval edges
        interval = args.interval / num_edges
        threshold = 1 - interval
        # (opt2) Exact interval edges, but time-consuming
        # threshold = torch.min(mask.topk(interval).values)
    else:
        interval = args.interval
        threshold = 1 - args.interval

    file_path = os.path.join(
        "examples", "creation_time", f"{args.dataset}_{interval}_{threshold}.pt"
    )

    if os.path.exists(file_path):
        mask = torch.load(file_path)
    else:
        print(f"Generate random time information for dataset {args.dataset}")
        mask = torch.rand(num_edges)
        torch.save(mask, file_path)

    # Filter data.edge_index based on the mask
    filtered_edge_index = edge_index[:, mask > threshold]
    if isinstance(data, pyg.data.Data):
        data.edge_index = filtered_edge_index
    else: # assume dgl graph
        data.remove_edges(torch.arange(data.number_of_edges()))
        # Add new edges
        data.add_edges(filtered_edge_index[0], filtered_edge_index[1])
    return


def load_available_model(model, args: argparse.Namespace):
    available_model = []
    if args.dataset in ["cora", "yelp", "reddit", "PubMed", "amazon", "products", "papers"]:
        name_prefix = f"{args.dataset}_{args.model}_{args.aggr}"
    else:
        name_prefix = f"{args.dataset}_{args.model}_{args.aggr}_lp"
    for file in os.listdir("examples/trained_model"):
        if re.match(name_prefix + "_[0-9]+_[0-9]\.[0-9]+\.pt", file):
            available_model.append(file)
    if len(available_model) == 0:  # no available model, train from scratch
        print(
            f"No available model. Please run `python {args.model}.py --dataset {args.dataset} --aggr {args.aggr}`"
        )
        return None

    # choose the model with the highest test acc
    accuracy = [
        float(re.findall("[0-9]\.[0-9]+", model_name)[0])
        for model_name in available_model
        if len(re.findall("[0-9]\.[0-9]+", model_name)) != 0
    ]
    index_best_model = np.argmax(accuracy)
    model = load(model, available_model[index_best_model])

    # remove redundant saved models, only save the one with the highest accuracy
    available_model.pop(index_best_model)
    clean(available_model)
    return model


def count_layers(model):
    # Count number of conv layers.
    count = 0
    for child in model.children():
        if any(
            [
                isinstance(child, cls)
                for cls in vars(pyg.nn.conv).values()
                if isinstance(cls, type)
            ]
        ):
            count += 1
        elif len(list(child.children())) > 0:
            count += count_layers(child)
    # print(f"Number of layers: {count}")
    return count


def affected_nodes_each_layer(
    edge_dicts: List[Dict[int, torch.Tensor]],
    srcs: Union[list, set],
    depth: int,
    self_loop: bool = False,
) -> defaultdict:
    affected = defaultdict(set)
    affected[0] = set(srcs)
    for i in range(depth):
        affected[i + 1].update(srcs)  # srcs are affected each layer
        # for gin and sage, changes also propagate along self-loop.
        if self_loop:
            affected[i + 1].update(affected[i])
        for node in affected[i]:
            for edge_dict in edge_dicts:
                affected[i + 1].update(edge_dict[node])
    return affected

# def affected_nodes_each_layer(
#     edge_dicts: List[Dict[int, torch.Tensor]],
#     srcs: Union[list, set],
#     depth: int,
#     self_loop: bool = False,
# ) -> defaultdict:
#     affected = defaultdict(list)
#     affected[0] = list(srcs)
#     for i in range(depth):
#         for src in srcs:
#             affected[i + 1] = list(srcs)
#         for node in affected[i]:
#             # for gin and sage, changes also propagate along self-loop.
#             if self_loop and (node not in affected[i + 1]):
#                 affected[i + 1].append(node)
#             for edge_dict in edge_dicts:
#                 for nghbr in edge_dict[node]:
#                     if nghbr not in affected[i + 1]:
#                         affected[i + 1].append(nghbr)
#     return affected


def unique_and_location(
    array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:  # generated by chatgpt
    # dictionary to store indices
    indices_dict = {}

    for idx, element in np.ndenumerate(array):
        if element not in indices_dict:
            indices_dict[element] = [idx[0]]
        else:
            indices_dict[element].append(idx[0])

    # get unique elements and their locations
    unique_elements = np.array(list(indices_dict.keys()))
    element_locations = list(indices_dict.values())

    return unique_elements, element_locations


def partition_by_aggregation_phase(
    model_configs: List[str], aggr: str
) -> List[List[str]]:
    sublists = []
    sublist = []
    for idx, elem in enumerate(model_configs):
        if elem == aggr:
            if sublist:  # if the sublist is not empty, add it to the sublists
                sublists.append(sublist)
            sublist = [elem]  # start a new sublist starting with the pattern
        else:
            sublist.append(elem)
        # at the end of the list, add the remaining sublist
        if idx == len(model_configs) - 1:
            sublists.append(sublist)
    if (
        sublists[0][0] != aggr
    ):  # if the first sublist doesn't contain the aggr, remove it
        sublists.pop(0)
    return sublists


def burst_sampler(full_edges: np.ndarray, num_edge_changed) -> np.ndarray:
    # Return the indexes for the topest hub node.
    # Sort hub nodes and in-edge indexes by degree and return the top `num_edge_changed` indexes.

    _, index_in_edges = unique_and_location(
        full_edges[1]
    )  # no need for id, only need degree and location
    # sort index_in_edges the in-degree (len of index_in_edges), in descending order
    index_in_edges.sort(key=lambda x: len(x), reverse=True)  # list of list
    flatten_sorted_index_in_edges = [
        item for sublist in index_in_edges for item in sublist
    ]

    return flatten_sorted_index_in_edges[:num_edge_changed]


def edge_remove(
    full_edges: np.ndarray, batch_size: int, distribution: str, directed: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    # ATTENTION: RETURN VALUE HAVE DIFFERENT SHAPE!!!!
    # batch size refers to number of undirected edges. If directed edges are removed, remove double the size
    num_edges = full_edges.shape[1]
    num_edge_changed = (
        2 * batch_size if directed else batch_size
    )  # if undirected, sample a half and add back the other direction.

    if distribution == "random":
        index_added_edges = np.random.choice(
            num_edges, num_edge_changed, replace=False)
    elif distribution == "burst":
        index_added_edges = burst_sampler(full_edges, num_edge_changed)
    elif distribution == "combine":
        print("Combined sampling is not implemented yet. Use random instead")
        index_added_edges = np.random.choice(
            num_edges, num_edge_changed, replace=False)

    if not directed:  # if directed, the other direction is added.
        index_augmented = []
        for index in index_added_edges:
            the_other_direction = np.flip(full_edges[:, index])
            for x in range(full_edges.shape[1]):
                if np.all(full_edges[:, x] == the_other_direction):
                    index_augmented.append(x)
        index_augmented = np.sort(
            np.concatenate((index_added_edges, np.array(index_augmented)))
        )

    sample_edges = np.transpose(
        full_edges[:, index_added_edges])  # TRANSPOSED!!!!!!
    edges_one_batch_missing = full_edges[
        :, np.setdiff1d(np.arange(num_edges), index_augmented)
    ]

    return sample_edges, edges_one_batch_missing

def get_graph_dynamics(
    tensor: torch.Tensor, batch_size: int, stream: str = "mix"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :param tensor: original edge_index
    :param batch_size: number of streaming edges in a time interval
    :param mode: operations of edges. insertion, deletion or both.
    :return: initial_edges, final_edges, inserted_edges, removed_edges
    """
    # Ensure batch_size is smaller than tensor's size
    assert batch_size < tensor.size(
        1
    ), "batch_size should be smaller than the number of columns in the tensor"

    # Generate random permutation of indices
    indices = torch.randperm(tensor.size(1))

    if stream == "add":
        # Select columns excluding batch_size number of random columns
        new_tensor = tensor[:, indices[:-batch_size]]
        removed_tensor = tensor[:, indices[-batch_size:]]
        return new_tensor, tensor, removed_tensor, torch.empty((2, 0))

    elif stream == "delete":
        new_tensor = tensor[:, indices[:-batch_size]]
        removed_tensor = tensor[:, indices[-batch_size:]]
        return tensor, new_tensor, torch.empty((2, 0)), removed_tensor

    elif stream == "mix":
        set_A_indices = indices[: batch_size // 2]
        set_B_indices = indices[batch_size // 2: batch_size]
        tensor_A = tensor[:, indices[batch_size // 2:]]
        tensor_B = tensor[
            :, torch.cat((indices[: batch_size // 2], indices[batch_size:]))
        ]
        set_A = tensor[:, set_A_indices]
        set_B = tensor[:, set_B_indices]
        # for corner case [batch_size = 1], randomly swap set_A&set_B, tensor_A&tensor_B to avoid all cases are "remove"
        if batch_size == 1 and torch.rand(1) > 0.5:
            set_A, set_B = set_B, set_A
            tensor_A, tensor_B = tensor_B, tensor_A
        return tensor_A, tensor_B, set_A, set_B


def is_large(data):
    return True if data.num_nodes > 50000 else False


def concate_by_id(
    full: np.ndarray, batch: np.ndarray, position: np.ndarray
) -> np.ndarray:
    batch_size = len(batch)
    full_size = len(full)

    if np.array_equal(position, np.arange(full_size, full_size + batch_size)):
        # directly append to the tail
        full = np.concatenate((full, batch), axis=0) if len(
            full) != 0 else batch

    else:  # extend `full` if necessary and insert `batch`
        if np.max(position) >= full_size:
            extended = np.zeros((np.max(position) + 1, full.shape[1]))
            extended[: full.shape[0]] = full
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

def to_dict_wiz_cache(edges:torch.Tensor, dir, filename):
    edge_dict_file_name = os.path.join(dir, filename)
    if os.path.exists(edge_dict_file_name):
        with open(edge_dict_file_name, 'rb') as file:
            edge_dict = pickle.load(file)
    else:
        edge_dict = to_dict(edges)
        with open(edge_dict_file_name, 'wb') as file:
            pickle.dump(edge_dict, file)
    print(f"Successfully build/load the {filename[:-7]} dict")
    return edge_dict

# @measure_time
def to_dict(edges: torch.Tensor): # faster
    #from tqdm import tqdm
    # print("start building edge dict")
    edge_dict = defaultdict(list)
    sources, destinations = edges[0].tolist(), edges[1].tolist()
    #pbar = tqdm(total=len(sources))
    for i, (source, dest) in enumerate(zip(sources, destinations)):
        edge_dict[source].append(dest)
    #    if i % 10000000 == 0:
    #        pbar.update(10000000)
    #pbar.close()
    return edge_dict

# def to_dict(edges: torch.Tensor):  # slower
#     edge_dict = defaultdict(list)
#     for i in range(edges.size(1)):
#         source_node = edges[0, i].item()
#         dest_node = edges[1, i].item()
#         edge_dict[source_node].append(dest_node)
#     return edge_dict


def get_stacked_tensors_from_dict(dictionary: dict, ids):
    return torch.stack([dictionary[id] for id in ids])


# def get_open_fds():
#     pid = os.getpid()
#     fds = []
#     try:
#         fds = os.listdir(f'/proc/{pid}/fd')
#     except FileNotFoundError:
#         # Handle the case where the /proc filesystem isn't available, e.g., on non-Linux systems
#         pass
#     return len(fds)


def get_open_fds():
    pid = os.getpid()
    output = subprocess.check_output(['lsof', '-p', str(pid)])
    lines = output.decode().strip().split('\n')
    return len(lines) - 1
