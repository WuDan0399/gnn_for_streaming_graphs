# NOTE: run with baselinePyG conda env!
import os

import torch
from tqdm import tqdm
from EventQueue import *
from utils import *
from torch_geometric.data import Data
from get_intermediate_result import inference_for_intermediate_result


# import multiprocessing as mp
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import concurrent.futures

class inkstream:
    def __init__(self, model, folder: str = "", aggregator: str = "min", verify: bool = False, verification_tolerance: float = 1e-6, out_channels:int = 1, ego_net: bool = False, multi_thread: int = 0):
        self.model = model
        self.folder = folder
        self.aggregator = aggregator
        self.is_monotonic = True if aggregator in ["min", "max"] else False
        self.verify = verify
        self.verification_tolerance = verification_tolerance
        self.out_channels = out_channels
        self.ego_net = ego_net
        self.multi_thread = multi_thread

        self.event_dict = {} 
        self.fetched_nodes = None  

        self.nlayer = count_layers(self.model)

        if self.multi_thread > 0:
            print(f"Use {self.multi_thread} threads.")
            mp.set_start_method('spawn')
            self.model.share_memory()
        else:
            print("Use single thread.")

    @torch.no_grad()
    def user_apply(self, events: dict, base_value: torch.Tensor, intm_initial: dict = None, it_layer: int = 0, node: int = -1):
        raise NotImplementedError

    @torch.no_grad()
    def transformation(self, model_operation: str, x: torch.Tensor):
        x = eval(model_operation)
        return x

    def inc_aggregator_pair(self, message_a, message_b):  # for min/max
        return torch.minimum(message_a, message_b)

    def inc_aggregator(self, message_list: torch.Tensor):
        return torch.min(message_list, dim=0)

    def monotonic_aggregator(self, messages: list):
        # Applicable for min/max as aggregator
        return messages[0] if len(messages) == 1 else torch.min(torch.stack(messages), dim=0)[0]

    def accumulative_aggregator(self, messages: list):
        # Applicable for add/mean as aggregator
        return messages[0] if len(messages) == 1 else torch.sum(torch.stack(messages), dim=0)[0]

    def user_reducer(self, messages: list):
        raise NotImplementedError

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        return

    def create_events_for_changed_edges(self, event_q, inserted_edges, removed_edges, message_list, updated_message_dict=None):
        if updated_message_dict is None:
            updated_message_dict = {}
        if self.is_monotonic:
            for src, dest in removed_edges:
                event_q.push_monotonic_event("remove", dest, message_list[src])
            for src, dest in inserted_edges:
                if src in updated_message_dict:
                    event_q.push_monotonic_event("insert", dest, updated_message_dict[src]
                                                 )
                else:
                    event_q.push_monotonic_event(
                        "insert", dest, message_list[src])
        else:
            for src, dest in removed_edges:
                event_q.push_accumulative_event(
                    "update", dest, -message_list[src])
            for src, dest in inserted_edges:
                if src in updated_message_dict:
                    event_q.push_accumulative_event("update", dest, updated_message_dict[src])
                else:
                    event_q.push_accumulative_event("update", dest, message_list[src])

    def load_context(self, data_dir: str, data: Data):
        inserted_edges, removed_edges = [], []
        if osp.exists(osp.join(data_dir, "inserted_edges.pt")):
            inserted_edges = torch.load(
                osp.join(data_dir, "inserted_edges.pt"))
            inserted_edges = [(src.item(), dst.item()) for src, dst in zip(inserted_edges[0], inserted_edges[1])]
        if osp.exists(osp.join(data_dir, "removed_edges.pt")):
            removed_edges = torch.load(
                osp.join(data_dir, "removed_edges.pt"))
            removed_edges = [
                (src.item(), dst.item())
                for src, dst in zip(removed_edges[0], removed_edges[1])
            ]
        if inserted_edges == [] and removed_edges == []:
            raise Exception("Problematic Data: no inserted or removed edges", data_dir)

        final_edges = torch.load(osp.join(data_dir, "final_edges.pt"))
        final_out_edge_dict = to_dict_wiz_cache(final_edges, data_dir, f'final_out_edge_dict.pickle')
        final_in_edge_dict = to_dict_wiz_cache(final_edges[[1, 0], :], data_dir, f"final_in_edge_dict.pickle")
        del final_edges
        

        initial_edges = torch.load(osp.join(data_dir, "initial_edges.pt"))
        init_out_edge_dict = to_dict_wiz_cache(initial_edges, data_dir, f'init_out_edge_dict.pickle')
        init_in_edge_dict = to_dict_wiz_cache(initial_edges[[1, 0], :], data_dir, f"init_in_edge_dict.pickle")

        intm_initial = load_tensors_to_dict(
            osp.join(data_dir), skip=7, postfix="_initial.pt")
        if intm_initial == {}:
            intm_initial = self.intm_fetched(data, initial_edges, False, inserted_edges, removed_edges,
                                             init_in_edge_dict, final_in_edge_dict, init_out_edge_dict, final_out_edge_dict)
        return final_edges, inserted_edges, removed_edges, init_in_edge_dict, init_out_edge_dict, final_in_edge_dict, final_out_edge_dict, intm_initial

    def intm_fetched(self, data, edges, reuse: bool = True, inserted_edges=None, removed_edges=None, init_in_edge_dict=None, final_in_edge_dict=None, init_out_edge_dict=None, final_out_edge_dict=None):
        if not reuse:
            direct_affected_nodes = set([dst for _, dst in inserted_edges + removed_edges])
            total_fetched_nodes = affected_nodes_each_layer([
                init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict, ], direct_affected_nodes, depth=self.nlayer)
            self.fetched_nodes = torch.LongTensor(
                list(total_fetched_nodes[self.nlayer]))

        data.edge_index = edges
        loader = data_loader(data, num_layers=self.nlayer, num_neighbour_per_layer=-
                             1, separate=False, input_nodes=self.fetched_nodes)
        intm_raw = inference_for_intermediate_result(self.model, loader)
        intm = {
            it_layer: {
                "before": {
                    self.fetched_nodes[i].item(): value["a-"][i]
                    for i in range(len(self.fetched_nodes))
                }, "after": {
                    self.fetched_nodes[i].item(): value["a"][i]
                    for i in range(len(self.fetched_nodes))
                }, }
            for it_layer, value in intm_raw.items()
        }
        return intm

    def verification(self, data, data_dir: str, final_edges, inserted_edges, removed_edges, init_out_edge_dict, final_out_edge_dict, intm_initial, cnt_dict):
        intm_final = load_tensors_to_dict(osp.join(self.folder, data_dir), skip=7, postfix="_final.pt"
                                          )
        if intm_final == {}:
            intm_final = self.intm_fetched(data, final_edges)

        direct_affected_nodes = set(
            [dst for _, dst in inserted_edges + removed_edges])
        affected_nodes = affected_nodes_each_layer(
            [init_out_edge_dict, final_out_edge_dict], direct_affected_nodes, depth=self.nlayer - 1)
        for it_layer in range(self.nlayer):
            for node in affected_nodes[it_layer]:
                # current layer result of affected node in current layer
                for it_phase in intm_initial[f"layer{it_layer + 1}"]:
                    if not torch.all(torch.isclose(intm_initial[f"layer{it_layer + 1}"][it_phase][node], intm_final[f"layer{it_layer + 1}"][it_phase][node], atol=self.verification_tolerance)):
                        print(
                            f"{bcolors.FAIL}[Failed]{bcolors.ENDC} {it_layer + 1}, {it_phase}, {node}")
                    else:
                        print(
                            f"{bcolors.OKGREEN}[Matched]{bcolors.ENDC} {it_layer + 1}, {it_phase}, {node}")
                if it_layer < self.nlayer - 1:
                    # test result for transformation and user-defined functions with next layer result
                    if not torch.all(torch.isclose(intm_initial[f"layer{it_layer + 2}"]["before"][node], intm_final[f"layer{it_layer + 2}"]["before"][node], atol=self.verification_tolerance)):
                        print(
                            f"{bcolors.FAIL}[Failed result in next layer]{bcolors.ENDC} {it_layer + 2}, before, {node}")
                    else:
                        print(
                            f"{bcolors.OKGREEN}[Matched result in next layer]{bcolors.ENDC} {it_layer + 2}, before, {node}")

    def incremental_aggregation_user(self, events:dict=None, it_layer:int=-1, destination:int=-1, previous_in_edge_dict:dict={},
                                     current_in_edge_dict:dict={}, intm_initial:dict={}) -> Tuple[bool, torch.Tensor, str]:
        raise NotImplementedError

    def incremental_aggregation_add(self, events:dict=None, it_layer:int=-1, destination:int=-1, intm_initial:dict={}) -> Tuple[bool, torch.Tensor, str]:
        aggred_dst = intm_initial[f"layer{it_layer}"]["after"][destination]
        changed_aggred_dst = aggred_dst + events["update"]
        return True, changed_aggred_dst, "recompute"

    def incremental_aggregation_mean(self, events:dict=None, it_layer:int=-1, destination:int=-1, prev_degree:int=1, curr_degree:int=1, intm_initial:dict={}) -> Tuple[bool, torch.Tensor, str]:
        aggred_dst = intm_initial[f"layer{it_layer}"]["after"][destination]
        changed_aggred_dst = (aggred_dst *prev_degree + events["update"])/curr_degree
        return True, changed_aggred_dst, "recompute"

    def incremental_aggregation_mono(self, events:dict=None, it_layer:int=-1, destination:int=-1, current_in_edge_dict:dict={}, intm_initial:dict={}) -> Tuple[bool, torch.Tensor, str]:
        # old aggregated timing_result
        aggred_dst = intm_initial[f"layer{it_layer}"]["after"][destination]
        no_new_message = "insert" not in events
        if not no_new_message:
            aggregated_new_message = events["insert"]
        if "remove" not in events:
            condition = "add_only"
            changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
            changed = not torch.equal(changed_aggred_dst, aggred_dst)
        else:
            aggregated_old_message = events["remove"]
            remove_mask = (aggred_dst == aggregated_old_message)
            if remove_mask.any():
                if no_new_message:
                    condition = "recompute"
                    neighbours = current_in_edge_dict[destination] 
                    if neighbours != []:
                        message_list = get_stacked_tensors_from_dict(intm_initial[f"layer{it_layer}"]["before"], neighbours)
                        changed_aggred_dst = self.inc_aggregator(message_list).values
                    else:
                        changed_aggred_dst = torch.zeros(aggred_dst.shape)  
                    changed = True

                else:
                    all_less = torch.le(aggregated_new_message[remove_mask], aggregated_old_message[remove_mask]).all()
                    if all_less:
                        condition = "recompute"
                        neighbours = current_in_edge_dict[destination]
                        if neighbours != []:
                            message_list = get_stacked_tensors_from_dict(intm_initial[f"layer{it_layer}"]["before"], neighbours)
                            changed_aggred_dst = self.inc_aggregator(message_list).values
                        else:
                            changed_aggred_dst = torch.zeros(aggred_dst.shape)
                        changed = True
                    else:
                        condition = "covered"
                        changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
                        changed = True

            else:
                condition = "del_no_change"
                if no_new_message:
                    changed = False
                    changed_aggred_dst = None
                else:
                    changed_aggred_dst = torch.minimum(aggred_dst, aggregated_new_message)
                    changed = not torch.equal(changed_aggred_dst, aggred_dst)

        return changed, changed_aggred_dst, condition

    @torch.no_grad()
    def incremental_inference_st(self, initial_out_edge_dict: dict, initial_in_edge_dict: dict, current_out_edge_dict: dict, current_in_edge_dict: dict, intm_initial: dict, inserted_edges: list, removed_edges: list):
        self.model.eval()
        event_q, event_q_bkp = EventQueue(), EventQueue()

        start = time.perf_counter()
        self.create_events_for_changed_edges(event_q, inserted_edges, removed_edges, intm_initial["layer1"]["before"])
        self.event_dict = event_q.reduce(
            self.monotonic_aggregator, self.accumulative_aggregator, self.user_reducer)

        cnt_dict = defaultdict(lambda: defaultdict(int))
        for it_layer, operations_per_layer in enumerate(self.model_config):
            out = dict()
            degree_dict = defaultdict(dict)
            for destination in self.event_dict: 
                if operations_per_layer[0] in ["min", "max"]:
                    (aggr_changed, changed_aggred_dst, condition) = self.incremental_aggregation_mono(
                        self.event_dict[destination], it_layer + 1, destination, current_in_edge_dict, intm_initial)
                elif operations_per_layer[0] == "add":
                    (aggr_changed, changed_aggred_dst, condition) = self.incremental_aggregation_add(
                        self.event_dict[destination], it_layer + 1, destination, intm_initial)
                elif operations_per_layer[0] == "mean":
                    if destination not in degree_dict:
                        degree_dict[destination]["current"] = len(current_in_edge_dict[destination])
                        degree_dict[destination]["initial"] = len(initial_in_edge_dict[destination])
                    (aggr_changed, changed_aggred_dst, condition) = self.incremental_aggregation_mean(
                        self.event_dict[destination], it_layer + 1, destination, degree_dict[destination]["initial"], degree_dict[destination]["current"], intm_initial)
                else:
                    (aggr_changed, changed_aggred_dst, condition) = self.incremental_aggregation_user(
                    self.event_dict[destination], it_layer + 1, destination, current_in_edge_dict, intm_initial)

                cnt_dict[it_layer + 1][condition] += 1
                cnt_dict[it_layer + 1]["computed"] += 1
                if not aggr_changed and "user" not in self.event_dict[destination]:
                    continue
                else:
                    if not aggr_changed:
                        changed_aggred_dst = intm_initial[f"layer{it_layer+1}"]["after"][destination]
                    else:
                        intm_initial[f"layer{it_layer+1}"]["after"][destination] = changed_aggred_dst
                    next_layer_before_aggregation = changed_aggred_dst.unsqueeze(0).to(device)
                    for model_operation in operations_per_layer[1:]:
                        if model_operation == "user_apply":
                            next_layer_before_aggregation = (
                                next_layer_before_aggregation.squeeze())
                            next_layer_before_aggregation = self.user_apply(
                                self.event_dict[destination], next_layer_before_aggregation, intm_initial, it_layer, destination)
                            next_layer_before_aggregation = (
                                next_layer_before_aggregation.unsqueeze(0))
                        elif isinstance(model_operation, Callable):
                            next_layer_before_aggregation = model_operation(next_layer_before_aggregation
                                                                            )
                        else:
                            print("Unrecognized operation: ", model_operation)
                    next_layer_before_aggregation = (
                        next_layer_before_aggregation.squeeze().to("cpu"))

                    if it_layer + 1 < self.nlayer:
                        event_q_bkp.bulky_push(initial_out_edge_dict[destination], current_out_edge_dict[destination],
                                               intm_initial[f"layer{it_layer + 2}"]["before"][destination],
                                               next_layer_before_aggregation, operations_per_layer[0])
                    out[destination] = next_layer_before_aggregation

            """ 
             end of layer processing: 1.update result in intm_initial for verification 2. insert event for changed edges 
             3. event queue update.
            """
            if it_layer + 1 < self.nlayer: 
                self.create_events_for_changed_edges(
                    event_q_bkp, inserted_edges, removed_edges, intm_initial[f"layer{it_layer + 2}"]["before"], out)

                    intm_initial[f"layer{it_layer + 2}"]["before"][node] = out[node]
                    self.user_propagate(node, out[node], event_q_bkp)

                event_q = event_q_bkp
                event_q_bkp = EventQueue()
                self.event_dict = event_q.reduce(
                    self.monotonic_aggregator, self.accumulative_aggregator, self.user_reducer)

        end = time.perf_counter()
        return cnt_dict, end - start

    @torch.no_grad()
    def batch_incremental_inference(self, data, niters:int=10):
        t_distribution = []
        condition_distribution = defaultdict(list)
        entries = os.listdir(self.folder)
        data_folders = [
            entry
            for entry in entries
            if entry.isdigit() and os.path.isdir(os.path.join(self.folder, entry))
        ]
        
        for data_dir in tqdm(data_folders[:niters]):
            _, inserted_edges, removed_edges, init_in_edge_dict, init_out_edge_dict, final_in_edge_dict, final_out_edge_dict, intm_initial = self.load_context(
                osp.join(self.folder, data_dir), data)
            cnt_dict, t_inc = self.incremental_inference_st(
                init_out_edge_dict, init_in_edge_dict, final_out_edge_dict, final_in_edge_dict, intm_initial, inserted_edges, removed_edges)
            t_distribution.append(t_inc)

            for it_layer in cnt_dict.keys():
                condition_distribution[it_layer].append([
                    cnt_dict[it_layer]["computed"], cnt_dict[it_layer]["add_only"], cnt_dict[it_layer]["del_no_change"], cnt_dict[it_layer]["covered"], cnt_dict[it_layer]["recompute"]])

            for it_layer in condition_distribution.keys():
                np.save(f"tmp_GIN_layer{it_layer}.npy",condition_distribution[it_layer])

        return condition_distribution, t_distribution

    @torch.no_grad()
    def incremental_layer(self, operations_in_layer, it_layer, destination, current_in_edge_dict, intm_initial):
        try:
            aggr_changed, changed_aggred_dst, condition = self.incremental_aggregation_mono(
                self.event_dict[destination], it_layer + 1, destination, current_in_edge_dict, intm_initial)
        except Exception as e:
            print(e)

        if not aggr_changed and "user" not in self.event_dict[destination]:
            return False, False, None, None, condition
        else:
            changed_aggred_dst_copy = None
            if not aggr_changed:
                changed_aggred_dst = intm_initial[f"layer{it_layer+1}"]["after"][
                    destination
                ]
            else:
                changed_aggred_dst_copy = changed_aggred_dst.clone()
            next_layer_before_aggregation = changed_aggred_dst.unsqueeze(
                0).to(device)
            for model_operation in operations_in_layer[1:]:
                if model_operation == "user_apply":
                    next_layer_before_aggregation = (
                        next_layer_before_aggregation.squeeze())
                    try:
                        next_layer_before_aggregation = self.user_apply(
                            self.event_dict[destination], next_layer_before_aggregation, intm_initial, it_layer, destination)
                    except Exception as e:
                        print(e)

                    next_layer_before_aggregation = (
                        next_layer_before_aggregation.unsqueeze(0))
                elif isinstance(model_operation, Callable):
                    next_layer_before_aggregation = model_operation(
                        next_layer_before_aggregation)
                else:
                    print("Unrecognized operation: ", model_operation)
            efore_aggregation = next_layer_before_aggregation.squeeze().to("cpu")
            return True, aggr_changed, next_layer_before_aggregation, changed_aggred_dst_copy, condition
