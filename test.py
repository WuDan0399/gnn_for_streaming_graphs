import torch.nn.functional as F
from torch import Tensor, nn
from torch_sparse import matmul
import torch
import time

from utils import *
from GCN import GCN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
get_time = time.time
unit = "seconds"
niter = 1000

print_style = "table"  # "readable"   "table"


def matmul_test():
    print("[Test] matrix multiplication (Linear)..")
    percentage = 0.1  # viewed as for all numbers within this threshold, redo the computation
    # Generating random tensor with shape (300, 256)
    for l in [200, 500, 800, 1000, 5000]:
        weight = torch.randn(256,l).to(device)
        bias = torch.randn(256).to(device)
        x = torch.randn(l).to(device)
        num_elements = int(x.numel() * percentage)  # Calculate the number of elements to make zero
        indices = torch.randperm(x.numel())[:num_elements]  # Randomly choose indices to make zero
        x2 = x.clone()
        x2[indices] = 0.01  # Change selected elements
        old_y = F.linear(x, weight, bias)  # old value

        if print_style == "table":
            print(f"[MatMul Runtime: {weight.shape} and {x.shape}, {niter} iters, {unit}, {device}] ")
        # Ground Truth Value
        start = get_time()
        for _ in range(niter):
            new_y = F.linear(x2, weight, bias)  # new value
        end = get_time()
        total_time = end - start
        if print_style == "readable":
            print(f'MatMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'GroundTruth\t {total_time:.4f}')

        #  opt 1: y' = Wx + Wδx + b , given y = Wx + b, and x' = x + δx. Directly use matrix multiplication with zeros
        start = get_time()
        for _ in range(niter) :
            # 模拟一个compare的步骤 对 x， 比如 前面改掉几个numbers，然后 相减，计算增量，增量的话，bias应该是0?
            delta_x = x2-x
            delta_y = F.linear(delta_x, weight)  # δy = Wδx
            new_y_inc = delta_y + old_y
        end = get_time()
        total_time = end - start
        if print_style == "readable":
            print(f'Incremental computation with Sparse MatMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'Incremental \t {total_time:.4f}')

        # opt 2: take those vectors out, then add it back.
        start = get_time()

        # 模拟一个compare的步骤 对 x， 比如 前面改掉几个numbers，然后 相减，计算增量，直接取出来NZ的部分，来做向量乘法，最后加起来
        for _ in range(niter) :
            diff_indices = torch.nonzero(x2 != x).flatten() # this output a 2-d array, so flatten for vector x
            delta_x2 = x2[diff_indices]-x[diff_indices]
            delta_y2 = F.linear(delta_x2, weight[:, diff_indices])
            new_y_vec = old_y + delta_y2

        end = get_time()
        total_time = end - start
        if print_style == "readable" :
            print(f'Incremental computation with VecMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'Vector\t {total_time:.4f}')
        # Correctness test
        threshold = 1e-6

        # Compare the tensors within the threshold
        # print(new_y == new_y_inc)
        # print(new_y == new_y_vec)
        # print(torch.isclose(new_y, new_y_inc, atol=threshold))
        # print(torch.isclose(new_y, new_y_vec, atol=threshold))



def load_intermediate_result(folder_intermediate, src, dest):
    root_folder = folder_intermediate
    intm = {}
    label0 = ["truth", "base"]
    label1 = ["layer1", "layer2"]
    label2 = ["before", "after"]
    for l0 in label0 :
        if l0 not in intm :
            intm[l0] = {}
        for l1 in label1 :
            if l1 not in intm[l0] :
                intm[l0][l1] = {}
            for l2 in label2 :
                if l0 == "truth" :
                    postfix = "_aggregation.pt"
                else :
                    postfix = f"_aggregation_({src}, {dest}).pt"
                intm[l0][l1][l2] = torch.load(osp.join(root_folder, l1, l2 + postfix))
                print(f"Load {osp.join(root_folder, l1, l2 + postfix)} to intm[{l0}][{l1}][{l2}]")
    return intm

def compare_graph_states(intm:dict, node):
    is_same = []
    is_same.append(torch.all(intm["truth"]["layer1"]['before'][node] == intm["base"]["layer1"]['before'][node]))
    is_same.append(torch.all(intm["truth"]["layer1"]['after'][node] == intm["base"]["layer1"]['after'][node]))
    is_same.append(torch.all(intm["truth"]["layer2"]['before'][node] == intm["base"]["layer2"]['before'][node]))
    is_same.append(torch.all(intm["truth"]["layer2"]['after'][node] == intm["base"]["layer2"]['after'][node]))
    print(f"Node {node}:")
    print(is_same)
    # if torch.tensor(False) in is_same:
    #     print(node)


def initialization(model_path, folder_intermediate, src, dest):
    fake_args = FakeArgs()
    root = "/Users/wooden/PycharmProjects/GNNAccEst"
    dataset = Planetoid(osp.join(root, "datasets", "Planetoid"), "Cora")
    data = dataset[0]
    model = GCN(dataset.num_features, 256, 7, fake_args)
    model = model.to(device)
    model = model.load_state_dict(
        torch.load(osp.join(root, model_path), map_location=device))

    #  load intermediate data for verification
    intm = load_intermediate_result(folder_intermediate, src, dest)
    edgelist = to_dict(data.edge_index)

    # check change before\after streaming edges
    compare_graph_states(intm, src)
    compare_graph_states(intm, dest)

    # for 2nd layer: 1-hop neighbour of dest
    print(f"For 1-hop neighbour of {dest}")
    for neighbour in edgelist[dest]:
        compare_graph_states(intm, neighbour)

    # for node in range(data.num_nodes):
    #     compare_graph_states(intm, node)

    return model, intm, data, edgelist

def single_task_test(model_path, folder_intermediate, src, dest):
    model, intm, _ = initialization(model_path, folder_intermediate, src, dest)

    # todo: change to neighbours
    print("[Test: Should contain False] Aggregation in Initial graph v.s. Updated Graph")
    print(intm["base"]["layer1"]['after'][0] == intm["truth"]["layer1"]['after'][0])
    print("[Test] Aggregation in Initial graph \ Recalculate in Initial Graph")
    print(intm["base"]["layer1"]['after'][0] == torch.min(intm["base"]["layer1"]['before'][[1862, 2582]], dim=0).values)
    print("[Test] Aggregation in Updated graph \ Recalculate in Updated Graph")
    print(intm["truth"]["layer1"]['after'][0] == torch.min(intm["truth"]["layer1"]['before'][[633, 1862, 2582]],
                                                           dim=0).values)
    print("[Test] Incremental Computation from Initial Graph to Updated Graph")
    print(torch.minimum(intm["base"]["layer1"]['after'][0], intm["base"]["layer1"]['before'][633]) ==
          intm["truth"]["layer1"]['after'][0])

def workflow_test(model_path, folder_intermediate, src, dst):
    model, intm, data, edgelist = initialization(model_path, folder_intermediate, src, dst)

    # Frontier Method test : Especially 2nd Layer
    task_q = [Task('add', src, dst, intm["base"]["layer1"]['before'][src])]
    task_q_bkp = []  # backup frontier for next iteration

    # first layer
    for task in task_q :  # assume op is 'add'
        updated_dst = torch.minimum(intm["base"]["layer1"]['after'][task.dst], task.msg)
        assert torch.all(updated_dst == intm["truth"]["layer1"]['after'][task.dst])
        if not torch.all(updated_dst == intm["base"]["layer1"]['after'][task.dst]) :
            # Update current layer output
            intm["base"]["layer1"]['after'][task.dst] = updated_dst

            # update the task queue
            for next_hop_neighbor in edgelist[task.dst] :
                task_q_bkp.append(
                    Task('delete', task.dst, next_hop_neighbor, intm["base"]["layer2"]['before'][task.dst]))
                task_q_bkp.append(Task('add', task.dst, next_hop_neighbor, intm["truth"]["layer2"]['before'][task.dst]))

    print_task_queue(task_q_bkp)
    # task merging or sorting
    task_dict = group_task_queue(task_q_bkp)

    for destination in task_dict :
        aggred_dst = intm["base"]["layer2"]['after'][destination]  # old aggregated timing_result
        # change_mask = torch.zeros(message.shape, dtype=torch.bool)

        old_message_list = torch.stack([msg for _, msg in task_dict[destination]["delete"]])
        aggregated_old_message = torch.min(old_message_list, dim=0).values
        new_message_list = torch.stack([msg for _, msg in task_dict[destination]["add"]])
        aggregated_new_message = torch.min(new_message_list, dim=0).values

        delete_mask = aggred_dst == aggregated_old_message
        if torch.sum(delete_mask) != 0 :
            # 下面这一句需要检查一下对不对
            masked_new_old_message = torch.stack(
                (aggregated_new_message[delete_mask], aggregated_old_message[delete_mask]))
            aggregated_new_old_message = torch.min(masked_new_old_message, dim=0)
            if torch.sum(aggregated_new_old_message.indices) != 0 :
                # 有old values dominates的值， 无法恢复，重新计算
                print("[skip] recomputed")
            else :
                print("target: incremental get it")
                aggred_dst = torch.min(aggred_dst, aggregated_new_message).values
                assert torch.all(aggred_dst == intm["truth"]["layer2"]['after'][destination])
        else :
            print("No change after deletion")
            aggred_dst = torch.min(aggred_dst, aggregated_new_message).values
            assert torch.all(aggred_dst == intm["truth"]["layer2"]['after'][destination])

if __name__ == '__main__':
    # matmul_test()
    # single_task_test(model_path='dynamic/examples/trained_model/Cora_GCN_min_69_0.721.pt',
    #                  folder_intermediate = "examples/intermediate/Cora/min/",
    #                  src = 633,
    #                  dest = 0)

    workflow_test(model_path='dynamic/examples/trained_model/Cora_GCN_min_69_0.721.pt',
                     folder_intermediate = "examples/intermediate/Cora/min/",
                     src = 633,
                     dst = 0)

