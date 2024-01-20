# NOTE: run with baselinePyG conda env!
import os

import torch
from tqdm import tqdm
from GCN import GCN
from EventQueue import *
from TaskQueue import *
from utils import *
from torch_geometric.data import Data
from get_intermediate_result import inference_for_intermediate_result

def inter_layer_calculate(it_layer:int, model_config:list, model, destination:int, changed_aggred_dst:torch.Tensor):
    # TODO: [DEBUG] the interlayer calculation does not give exactly the same result as in the model.
    # TODO: [DEBUG] divergence begins at the x=self.lin(x) process.
    # TODO: [DEBUG] linear(x[index]) != linear(x)[index]   My method v.s. original   Skipped for fast debugging.
    loc_aggr = [i for i in range(len(model_config)) if model_config[i] in ["min", "max"]] + [len(model_config)]
    start_layer = loc_aggr[it_layer-1]+1
    end_layer = loc_aggr[it_layer]
    x = changed_aggred_dst.to(device)
    for operation in model_config[start_layer: end_layer]:
        # operations from the targeted layer's aggregator to the next layer's aggregator (excluded)
        x = eval(operation)
    return x.to("cpu")


@torch.no_grad()
def incremental_inference(model, inititial_out_edge_dict: dict, current_out_edge_dict: dict, current_in_edge_dict: dict,
                          intm_initial: dict, inserted_edges: list, removed_edges: list):
    """
    Problems:
    1. very result after lin(x) could be slightly different in two versions.
    """
    memory_access_mine = 0
    model.eval()
    model_config_path = "model_configs/GCN.txt"
    with open(model_config_path, 'r') as f:
        model_config = f.read().splitlines()

    task_q, task_q_bkp = TaskQueue(), TaskQueue()

    start = time.perf_counter()

    # Initial Tasks
    for src, dest in inserted_edges:
        task_q.push_task('add',  dest, intm_initial["layer1"]['before'][src])
        # TODO: prob 1
        # task_q.push_task('add', src, dest, intm_final["layer1"]['before'][src])

    for src, dest in removed_edges:
        task_q.push_task('delete', dest, intm_initial["layer1"]['before'][src])

    memory_access_mine += (len(inserted_edges) +len(removed_edges))*len(intm_initial["layer1"]['before'][inserted_edges[0][0]])

    # message value after the corresponding message is consumed,e.g., after all messages in this layer is consumed.
    task_dict = task_q.reduce("min")

    nlayer = 2

    cnt_dict = defaultdict(lambda: defaultdict(int))
    for it_layer in range(1, nlayer+1):  # todo: decide number of layers by model or intermediate value
        out = dict()  # changed input for next layer or model output for all changed nodes

        for destination in task_dict:
            cnt_dict[it_layer]["computed"] = cnt_dict[it_layer]["computed"] + 1
            aggred_dst = intm_initial[f"layer{it_layer}"]['after'][destination]  # old aggregated timing_result
            memory_access_mine += len(aggred_dst)
            aggregated_new_message = task_dict[destination]["add"]
            if "delete" not in task_dict[destination]:
                # add-only, aggregated_new_message cannot be []
                cnt_dict[it_layer]["add_only"] = cnt_dict[it_layer]["add_only"] + 1
                changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
                comp_inter_layer_and_propagate = True if not torch.all(changed_aggred_dst == aggred_dst) else False

            else:
                aggregated_old_message = task_dict[destination]["delete"]
                delete_mask = aggred_dst == aggregated_old_message
                if torch.sum(delete_mask) != 0 :
                    if aggregated_new_message==[]:
                        # delete-only, aggregated_new_message is [], recompute
                        # print(f"recomputed {destination}")
                        cnt_dict[it_layer]["recompute"] = cnt_dict[it_layer]["recompute"] + 1
                        neighbours = current_in_edge_dict[destination]  #
                        if neighbours != []:
                            if type(intm_initial[f"layer{it_layer}"]['before']) == torch.Tensor :
                                message_list = intm_initial[f"layer{it_layer}"]['before'][neighbours]
                            else :
                                message_list = get_stacked_tensors_from_dict(intm_initial[f"layer{it_layer}"]['before'],
                                                                             neighbours)
                                for m in message_list:
                                    memory_access_mine += len(m)
                            changed_aggred_dst = torch.min(message_list, dim=0).values
                        else :
                            changed_aggred_dst = torch.zeros(aggred_dst.shape) # if no message, get 0s
                        comp_inter_layer_and_propagate = True

                    else:
                        masked_new_old_message = torch.stack(
                            (aggregated_new_message[delete_mask], aggregated_old_message[delete_mask]))
                        aggregated_new_old_message = torch.min(masked_new_old_message, dim=0)
                        if torch.sum(aggregated_new_old_message.indices) != 0 :
                            # print(f"recomputed {destination}")
                            cnt_dict[it_layer]["recompute"] = cnt_dict[it_layer]["recompute"] + 1
                            neighbours = current_in_edge_dict[destination]
                            if type(intm_initial[f"layer{it_layer}"]['before']) == torch.Tensor:
                                message_list = intm_initial[f"layer{it_layer}"]['before'][neighbours]
                            else:
                                message_list = get_stacked_tensors_from_dict(intm_initial[f"layer{it_layer}"]['before'],
                                                                             neighbours)
                                for m in message_list:
                                    memory_access_mine += len(m)
                            changed_aggred_dst = torch.min(message_list, dim=0).values
                            comp_inter_layer_and_propagate = True
                        else :
                            # print(f"[covered] incremental compute {destination}")
                            cnt_dict[it_layer]["covered"] = cnt_dict[it_layer]["covered"] + 1
                            changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
                            comp_inter_layer_and_propagate = True

                else :
                    cnt_dict[it_layer]["del_no_change"] = cnt_dict[it_layer]["del_no_change"] + 1
                    if not aggregated_new_message==[]:
                        # print(f"[no change for delete] incremental compute {destination}")
                        changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
                        comp_inter_layer_and_propagate = True if not torch.all(changed_aggred_dst == aggred_dst) else False
                    else:
                        comp_inter_layer_and_propagate = False
                        # print(f"[no change for delete and no add] {destination}")

            if comp_inter_layer_and_propagate:
                intm_initial[f"layer{it_layer}"]['after'][destination] = changed_aggred_dst
                next_layer_before_aggregation = inter_layer_calculate(it_layer, model_config, model, destination,
                                                                      changed_aggred_dst)
                if it_layer < nlayer :
                    # TODO: prob 1
                    # next_layer_before_aggregation = intm_final[f"layer{it_layer + 1}"]['before'][destination]
                    task_q_bkp.bulky_push(inititial_out_edge_dict[destination], current_out_edge_dict[destination],
                                          intm_initial[f"layer{it_layer + 1}"]['before'][destination],
                                          next_layer_before_aggregation)
                    memory_access_mine += len(intm_initial[f"layer{it_layer + 1}"]['before'][destination])
                    # propagated_nodes.add(destination)
                out[destination] = next_layer_before_aggregation # either input of next layer, or output of whole model

            # else:
            #     print("no change, propagation stops") # could be add\delete no effect

        """ 
         end of layer processing: 1.update result in intm_initial for verification 2. add task for changed edges 
         3. task queue update.
        """
        if it_layer<nlayer:  # end of layer warp up.
            # add task for removed edges each layer, delete the old message before the message changes
            for src, dest in removed_edges :
                task_q_bkp.push_task('delete', dest, intm_initial[f"layer{it_layer + 1}"]['before'][src])
                memory_access_mine += len(intm_initial[f"layer{it_layer + 1}"]['before'][src])

            # update the next layer input
            for node in out:
                intm_initial[f"layer{it_layer + 1}"]['before'][node] = out[node]

            # add task for changed edges each layer, add the new message after the message is updated
            for src, dest in inserted_edges :
                task_q_bkp.push_task('add', dest, intm_initial[f"layer{it_layer + 1}"]['before'][src])
                memory_access_mine += len(intm_initial[f"layer{it_layer + 1}"]['before'][src])


            # update the task queue
            task_q = task_q_bkp
            task_q_bkp = TaskQueue()
            task_dict = task_q.reduce('min')
    
    print(f"Memory Accessed in my method: {memory_access_mine*4/1024} KB")
    end = time.perf_counter()
    return cnt_dict, end-start

def batch_incremental_inference(model, data, folder:str, verify:bool = False, data_it:int=0) :
    verification_tolerance = 1e-05
    t_distribution = []
    # nlayers = len(list(model.children()))
    nlayers = count_layers(model)
    condition_distribution = defaultdict(list)
    entries = os.listdir(folder)
    data_folders = [entry for entry in entries if entry.isdigit() and os.path.isdir(os.path.join(folder, entry))]

    # Multiple different data (initial state info and final state info) directory
    for data_dir in tqdm(data_folders[:5]):
        # print(osp.join(folder, data_dir))
        initial_edges = torch.load(osp.join(folder, data_dir, "initial_edges.pt"))
        final_edges = torch.load(osp.join(folder, data_dir, "final_edges.pt"))
        inserted_edges, removed_edges = [], []
        if osp.exists(osp.join(folder, data_dir, "inserted_edges.pt")):
            inserted_edges = torch.load(osp.join(folder, data_dir, "inserted_edges.pt"))
            inserted_edges = [(src.item(), dst.item()) for src, dst in zip(inserted_edges[0], inserted_edges[1])]
        if osp.exists(osp.join(folder, data_dir, "removed_edges.pt")):
            removed_edges = torch.load(osp.join(folder, data_dir, "removed_edges.pt"))
            removed_edges = [(src.item(), dst.item()) for src, dst in zip(removed_edges[0], removed_edges[1])]
        init_out_edge_dict = to_dict(initial_edges)
        init_in_edge_dict = to_dict(initial_edges[[1, 0], :])
        final_out_edge_dict = to_dict(final_edges)
        final_in_edge_dict = to_dict(final_edges[[1,0], :])

        # get Initial Result: either load from file, or run with full model inference.
        intm_initial = load_tensors_to_dict(osp.join(folder, data_dir), skip=7, postfix="_initial.pt")
        direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
        fetched_nodes = affected_nodes_each_layer([init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict],
                                                    direct_affected_nodes, depth=nlayers)
        # last_layer_fetched_nodes = torch.LongTensor(list(fetched_nodes[nlayers].union(src_nodes)))
        last_layer_fetched_nodes = torch.LongTensor(list(fetched_nodes[nlayers]))
        print(last_layer_fetched_nodes)
        memory_access_baseline = len(fetched_nodes) * data.x.shape[1] * 4 / 1024  # in KB
        print(f"Memory Accessed in baseline: {memory_access_baseline} KB")
        
        if intm_initial == {} :
            # todo: change to affected part inference to save time for evaluation
            print("Running Inference for Theoretical Affected Area to Get Initial Result")
            # get neighbourhood information for all theoretical affected nodes, and the srcs, in case of recompute.
            # src_nodes = set([src for src, _ in inserted_edges + removed_edges])
            direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
            fetched_nodes = affected_nodes_each_layer([init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict],
                                                      direct_affected_nodes, depth=nlayers)
            # last_layer_fetched_nodes = torch.LongTensor(list(fetched_nodes[nlayers].union(src_nodes)))
            last_layer_fetched_nodes = torch.LongTensor(list(fetched_nodes[nlayers]))

            data2 = Data(x=data.x, edge_index=initial_edges).to(device)
            loader = data_loader(data2, num_layers=nlayers, num_neighbour_per_layer=-1, separate=False,
                                 input_nodes=last_layer_fetched_nodes)
            intm_initial_raw = inference_for_intermediate_result(model, loader)
            # rename the keys for alignment
            intm_initial = {
                it_layer : {
                    'before' : {last_layer_fetched_nodes[i].item() : value['a-'][i] for i in
                                range(len(last_layer_fetched_nodes))},
                    'after' : {last_layer_fetched_nodes[i].item() : value['a'][i] for i in
                               range(len(last_layer_fetched_nodes))}
                }
                for it_layer, value in intm_initial_raw.items()
            }

        cnt_dict, t_inc = incremental_inference(model, init_out_edge_dict, final_out_edge_dict, final_in_edge_dict,
                                                  intm_initial, inserted_edges, removed_edges)
        t_distribution.append(t_inc)

        if verify:
            # verification
            intm_final = load_tensors_to_dict(osp.join(folder, data_dir), skip=7, postfix="_final.pt")
            direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
            affected_nodes = affected_nodes_each_layer([init_out_edge_dict, final_out_edge_dict], direct_affected_nodes, depth=nlayers-1)
            for it_layer in range(nlayers):
                # print(f"[Layer {it_layer}] REAL:THEO={cn_tcomputed[it_layer+1]}:{len(affected_nodes[it_layer])}")
                print(f"[Layer {it_layer}] REAL:THEO={cnt_dict[it_layer + 1]['computed']}:{len(affected_nodes[it_layer])}")
                for node in affected_nodes[it_layer]:
                    for it_phase in intm_initial[f"layer{it_layer+1}"]: # current layer result of affected node in current layer
                        # TODO: prob 1
                        # if not torch.all(intm_initial[f"layer{it_layer+1}"][it_phase][node]
                        #                  == intm_final[f"layer{it_layer+1}"][it_phase][node]):
                        if not torch.all(torch.isclose(intm_initial[f"layer{it_layer+1}"][it_phase][node],
                                                       intm_final[f"layer{it_layer+1}"][it_phase][node], atol=verification_tolerance)):
                            print(f"{bcolors.FAIL}[Failed]{bcolors.ENDC} {it_layer+1}, {it_phase}, {node}")
                        else:
                            print(f"{bcolors.OKGREEN}[Matched]{bcolors.ENDC} {it_layer+1}, {it_phase}, {node}")
                    if it_layer < nlayers - 1:
                        for it_phase in intm_initial[f"layer{it_layer+2}"]: # next layer result of affected node in current layer
                            # TODO: prob 1
                            # if not torch.all(intm_initial[f"layer{it_layer+2}"][it_phase][node]
                            #                  == intm_final[f"layer{it_layer+2}"][it_phase][node]):
                            if not torch.all(torch.isclose(intm_initial[f"layer{it_layer+2}"][it_phase][node],
                                                           intm_final[f"layer{it_layer+2}"][it_phase][node], atol=verification_tolerance)):
                                print(f"{bcolors.FAIL}[Failed result in next layer]{bcolors.ENDC} {it_layer + 2}, {it_phase}, {node}")
                            else:
                                print(f"{bcolors.OKGREEN}[Matched result in next layer]{bcolors.ENDC} {it_layer + 2}, {it_phase}, {node}")

        for it_layer in cnt_dict.keys():
            condition_distribution[it_layer].append([ cnt_dict[it_layer]["computed"],
                                                cnt_dict[it_layer]["add_only"], cnt_dict[it_layer]["del_no_change"],
                                                cnt_dict[it_layer]["covered"], cnt_dict[it_layer]["recompute"] ])

    return condition_distribution, t_distribution


def main():
    # device = 'cpu'
    # args = FakeArgs(dataset="Cora", aggr="min", perbatch=1, stream="delete")
    parser = argparse.ArgumentParser()
    args = general_parser(parser)    
    dataset = load_dataset(args)
    # print_dataset(dataset)
    data = dataset[0].to(device)
    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)

    model = load_available_model(model, args)

    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)
    # print(f"Batch size for streaming graph: {batch_size}")

    intr_result_dir = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,  f"batch_size_{batch_size}")

    condition_distribution, exec_time_dist = batch_incremental_inference(model, data, intr_result_dir, verify=False, data_it=args.it)

    conditions_dir = osp.join("examples", "condition_distribution")
    create_directory(conditions_dir)
    for it_layer in condition_distribution.keys():
        np.save(osp.join(conditions_dir, f"[tot_add_delno_cov_rec]{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_{args.it}.npy"),
            condition_distribution[it_layer])
    time_dir = osp.join("examples","timing_result", "incremental")
    create_directory(time_dir)
    np.save(osp.join(time_dir,  "_".join(intr_result_dir.split("/")[2:]) + f"_{args.it}.npy"), exec_time_dist)


if __name__ == '__main__':
    main()