from inkstream import inkstream
from utils import *
from GIN import GIN
from EventQueue import *

from load_dataset import load_dataset

class inkstream_gin(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "max",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        out_channels: int = 1,
        ego_net: bool = False,
        multi_thread: int = 0,
    ):
        super().__init__(
            model,
            intr_result_dir,
            aggregator,
            verify,
            verification_tolerance,
            out_channels,
            ego_net,
            multi_thread,
        )
        self.model_config = [
            [self.aggregator, "user_apply", lambda x: self.model.convs[0].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[1].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[2].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[3].nn(x).relu()],
            [self.aggregator, "user_apply", lambda x: self.model.convs[4].nn(x).relu()],
        ]

    def user_apply(
        self,
        events: dict,
        base_value: torch.Tensor,
        intm_initial: dict = None,
        it_layer: int = 0,
        node: int = -1,
    ):
        if (
            "user" not in events.keys()
        ): 
            right_side = intm_initial[f"layer{it_layer+1}"]["before"][node].to(
                device)
        else:
            assert len(events["user"]) == 1
            right_side = events["user"][0].to(device)
        return base_value + right_side

    def user_reducer(self, messages: list):
        return messages

    def user_propagate(self, node: int, value: torch.Tensor, event_queue: EventQueue):
        event_queue.push_user_event("user", node, value)

    def inc_aggregator_pair(self, message_a, message_b): 
        return torch.maximum(message_a, message_b)

    def inc_aggregator(self, message_list: torch.Tensor):
        return torch.max(message_list, dim=0)

    def monotonic_aggregator(self, messages: list):
        if len(messages) == 2:
            return torch.maximum(messages[0], messages[1])
        else:
            return torch.max(torch.stack(messages), dim=0).values


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    if args.dataset in ["uci", "dnc", "epi"]:
        model = GIN(dataset.num_features, 256, 256, args).to(device)
        out_channels = 256
    else:
        if args.dataset == 'papers':
            model = GIN(dataset.num_features, dataset.num_classes + 1, args).to(device)
            out_channels = dataset.num_classes+1
        else:
            model = GIN(dataset.num_features, dataset.num_classes, args).to(device)
            out_channels = dataset.num_classes
    model = load_available_model(model, args)

    if args.perbatch < 1:
        batch_size = int(args.perbatch / 100 * data.num_edges)
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

    conditions_dir = osp.join("examples", "condition_distribution", "GIN")
    create_directory(conditions_dir)
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)

    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None

    if args.dataset in ["papers", "products"]:
        num_sample = 1
    else: 
        num_sample = min(num_sample, 1)

    if args.mt == 0:
        starter = inkstream_gin(
            model,
            intr_result_dir,
            aggregator=args.aggr,
            verify=False,
            out_channels=out_channels
        )
        condition_distribution, exec_time_dist = starter.batch_incremental_inference(data, niters=num_sample)
        unique_id = 0
        while osp.exists(osp.join(time_dir,f"GIN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_{unique_id}.npy")):
            unique_id += 1
        for it_layer in condition_distribution.keys():
            np.save(
                osp.join(
                    conditions_dir,
                    f"[tot_add_delno_cov_rec]GIN_{args.dataset}_{args.aggr}_{args.stream}_{batch_size}_layer{it_layer}_{unique_id}.npy",
                ),
                condition_distribution[it_layer])
        np.save(
            osp.join(time_dir, f"GIN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_{unique_id}.npy"),
            exec_time_dist,
        )
    else:
        if not isinstance(args.id, int) and args.id >= 0:
            raise ValueError("missing argument --id when --mt is used.")
        starter = inkstream_gin(
            model,
            osp.join(intr_result_dir, str(args.id)),
            aggregator="max",
            ego_net=True,
            multi_thread=args.mt,
            verify=False,
        )
        condition, exec_time = starter.incremental_inference_mt(data)
        if condition is not None and exec_time is not None:
            for it_layer in condition.keys():
                with open(
                        osp.join(
                            conditions_dir,
                            f"[tot_add_delno_cov_rec]GIN_{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_mt{args.mt}.txt",
                        ), "a") as f:
                    f.write(str(condition[it_layer]) + "\n")
            with open(osp.join(time_dir, f"GIN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_mt{args.mt}.txt"), "a") as f:
                f.write(f"{exec_time}\n")


if __name__ == "__main__":
    main()
