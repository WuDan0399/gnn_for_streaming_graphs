from inkstream import inkstream
from utils import *
from SAGE import SAGE
from EventQueue import *
from load_dataset import load_dataset

class inkstream_sage(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "min",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        ego_net: bool = False,
        multi_thread: int = 0
    ):
        super().__init__(
            model,
            intr_result_dir,
            aggregator,
            verify,
            verification_tolerance,
            ego_net,
            multi_thread,
        )
        # manually change the model_config, remove any operation before 1st aggregation function.
        self.model_config = [
            [
                self.aggregator,
                lambda x: self.model.convs[0].lin_l(x),
                "user_apply",
                lambda x: x.relu(),
            ],
            [
                self.aggregator,
                lambda x: self.model.convs[1].lin_l(x),
                "user_apply",
                lambda x: x.relu(),
            ],
        ]

    def layer1(self, x):
        return self.model.convs[0].lin_l(x)

    def layer2(self, x):
        return self.model.convs[1].lin_l(x)

    def user_apply(
        self,
        events: dict,
        base_value: torch.Tensor,
        intm_initial: dict = None,
        it_layer: int = 0,
        node: int = -1,
    ):
        # print("user_apply")
        if (
            "user" not in events.keys()
        ):  # left side (aggregated value) changes, right side (h_u) remains the same
            right_side = intm_initial[f"layer{it_layer+1}"]["before"][node].to(
                device)
        else:
            assert len(events["user"]) == 1
            right_side = events["user"][0].to(device)
        return eval(f"base_value + self.model.convs[{it_layer}].lin_r(right_side)")

    def user_reducer(self, messages: list):
        return messages

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        event_queue.push_user_event("user", node, value)


def main():
    # args = FakeArgs(dataset="cora", model="SAGE", aggr="min", perbatch=1, stream="add")
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    if args.dataset in ["uci", "dnc", "epi"]:
        model = SAGE(dataset.num_features, 256,256, args).to(device)
    else:
        if args.dataset == 'papers':
            model = SAGE(dataset.num_features, 256, dataset.num_classes + 1, args).to(device)
        else:
            model = SAGE(dataset.num_features, 256, dataset.num_classes, args).to(device)
    model = load_available_model(model, args)

    if args.perbatch < 1:
        batch_size = int(
            args.perbatch / 100 * data.num_edges
        )  # perbatch is [x]%, so divide by 100
    else:
        batch_size = int(args.perbatch)

    intr_result_dir = osp.join(
        "examples",
        "intermediate",
        args.dataset,
        "min",
        args.stream,
        f"batch_size_{batch_size}",
    )

    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)
    conditions_dir = osp.join("examples", "condition_distribution", "SAGE")
    create_directory(conditions_dir)

    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None

    if args.mt == 0:
        starter = inkstream_sage(model, intr_result_dir,
                                 multi_thread=args.mt, aggregator=args.aggr, verify=False)
        condition_distribution, exec_time_dist = starter.batch_incremental_inference(data, niters=num_sample)
        if args.aggr in ["min", 'max']:
            for it_layer in condition_distribution.keys():
                np.save(osp.join(conditions_dir,
                                 f"[tot_add_delno_cov_rec]SAGE_{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}.npy"),
                        condition_distribution[it_layer])
        np.save(osp.join(time_dir, f"SAGE_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}.npy"), exec_time_dist)
    else:
        if not isinstance(args.id, int) and args.id >= 0:
            raise ValueError("missing argument --id when --mt is used.")

        starter = inkstream_sage(model, osp.join(intr_result_dir, str(args.id)),
                                 multi_thread=args.mt, aggregator=args.aggr, verify=False)
        condition, exec_time = starter.incremental_inference_mt(data)
        if condition is not None and exec_time is not None:
            for it_layer in condition.keys():
                with open(
                        osp.join(
                            conditions_dir,
                            f"[tot_add_delno_cov_rec]SAGE_{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_mt{args.mt}.txt",
                        ), "a") as f:
                    f.write(f'{condition[it_layer]["computed"]}\t{condition[it_layer]["add_only"]}\t{condition[it_layer]["del_no_change"]}\t{condition[it_layer]["covered"]}\t{condition[it_layer]["recompute"]}\n')
            with open(
                osp.join(time_dir, f"SAGE_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_mt{args.mt}.txt"), "a") as f:
                f.write(f"{exec_time}\n")


if __name__ == "__main__":
    main()
