from ignite import ignite
from utils import *
from GIN import GIN
from EventQueue import *


class ignite_gin(ignite):
    def user_apply(self, events: dict, base_value: torch.Tensor, intm_initial: dict = None, it_layer: int = 0, node: int = -1):

        if "user" not in events.keys():  # left side (aggregated value) changes, right side (h_u) remains the same
            right_side = intm_initial[f"layer{it_layer+1}"]['before'][node]
        else:
            assert (len(events["user"]) == 1)
            right_side = events['user'][0]
        return base_value + right_side

    def user_reducer(self, messages: list):
        return messages

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        # chane to (\eps+1)*value if \eps not equal to 0
        event_queue.push_user_event("user", node, value)

    def inc_aggregator_pair(self, message_a, message_b):  # for min/max
        return torch.maximum(message_a, message_b)

    def inc_aggregator(self, message_list: torch.Tensor):
        return torch.max(message_list, dim=0)
    
    def monotonic_aggregator(self, messages: list):
        if len(messages) == 2:
            return torch.maximum(messages[0], messages[1])
        else:
            return torch.max(torch.stack(messages), dim=0).values




def main():
    # args = FakeArgs(dataset="cora", aggr="min", perbatch=1,
    #                 stream="add", model="GIN", binary=True, save_int=True, it=1)
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    model = GIN(data.x.shape[1], 2, args).to(device)
    model = load_available_model(model, args)

    if args.perbatch < 1:
        # perbatch is [x]%, so divide by 100
        batch_size = int(args.perbatch / 100 * data.num_edges)
    else:
        batch_size = int(args.perbatch)

    intr_result_dir = osp.join("examples", "intermediate", args.dataset, "min", args.stream,
                               f"batch_size_{batch_size}")

    starter = ignite_gin(model, "model_configs/GIN.txt",
                         intr_result_dir, ego_net=True)

    condition_distribution, exec_time_dist = starter.batch_incremental_inference(data, data_it=args.it)

    conditions_dir = osp.join("examples", "condition_distribution", "GIN")
    create_directory(conditions_dir)
    for it_layer in condition_distribution.keys():
        np.save(osp.join(conditions_dir,
                         f"[tot_add_delno_cov_rec]{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_{args.it}.npy"),
                condition_distribution[it_layer])
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)
    np.save(osp.join(time_dir, "_".join(intr_result_dir.split(
        "/")[2:]) + f"_{args.it}.npy"), exec_time_dist)


if __name__ == '__main__':
    main()
