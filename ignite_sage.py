from ignite import ignite
from utils import *
from SAGE import SAGE
from EventQueue import *

class ignite_sage(ignite):

    def user_apply(self, events: dict, base_value: torch.Tensor, intm_initial: dict = None, it_layer: int=0, node:int=-1):
        if "user" not in events.keys() :  # left side (aggregated value) changes, right side (h_u) remains the same
            right_side = intm_initial[f"layer{it_layer+1}"]['before'][node]
        else:
            assert(len(events["user"]) == 1)
            right_side = events['user'][0]
        return eval(f"base_value + self.model.convs[{it_layer}].lin_r(right_side)")

    def user_reducer(self, messages: list) :
        return messages

    def user_propagate(self, node: int, value:torch.Tensor, event_queue: EventQueue) :
        event_queue.push_user_event("user", node, value)

def main() :
    # args = FakeArgs(dataset="cora", model="SAGE", aggr="min", perbatch=1, stream="add")
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    model = SAGE(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)
    model = load_available_model(model, args)

    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)

    intr_result_dir = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                               f"batch_size_{batch_size}")

    starter = ignite_sage(model, "model_configs/SAGE.txt", intr_result_dir, verify=True)

    condition_distribution, exec_time_dist = starter.batch_incremental_inference(data, data_it=args.it)

    conditions_dir = osp.join("examples", "condition_distribution", "SAGE")
    create_directory(conditions_dir)
    for it_layer in condition_distribution.keys() :
        np.save(osp.join(conditions_dir,
                         f"[tot_add_delno_cov_rec]{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_{args.it}.npy"),
                condition_distribution[it_layer])
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)
    np.save(osp.join(time_dir, "_".join(intr_result_dir.split("/")[2 :]) + f"_{args.it}.npy"), exec_time_dist)


if __name__ == '__main__' :
    main()