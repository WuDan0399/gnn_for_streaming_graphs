from inkstream import inkstream
from utils import *
from GCN import GCN
from EventQueue import *

from load_dataset import load_dataset

class inkstream_gcn(inkstream):
    def __init__(
        self,
        model,
        intr_result_dir,
        aggregator: str = "min",
        verify: bool = False,
        verification_tolerance: float = 1e-5,
        out_channels:int = 1,
        ego_net: bool = False,
        multi_thread: int = 0
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
        # manually change the model_config, remove any operation before 1st aggregation function.
        self.model_config = [
            [self.aggregator, self.conv1_bias, self.conv2],
            [self.aggregator, self.conv2_bias],
        ]

    def conv1(self, x):
        return self.model.conv1.lin(x)

    def conv2(self, x):
        return self.model.conv2.lin(x)

    def conv1_bias(self, x):
        return (x + self.model.conv1.bias).relu()

    def conv2_bias(self, x):
        return x + self.model.conv2.bias


def main():
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0]
    if args.dataset in ["uci", "dnc", "epi"]:
        model = GCN(
            dataset.num_features, 256, 256, args
        ).to(device)
        out_channels = 256
    elif args.dataset == "papers":
        model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes+1, args).to(device)
        out_channels = dataset.num_classes+1
    else:
        model = GCN(
            dataset.num_features, args.hidden_channels, dataset.num_classes, args
        ).to(device)
        out_channels = dataset.num_classes
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
        "min",  # args.aggr,
        args.stream,
        f"batch_size_{batch_size}",
    )


    conditions_dir = osp.join("examples", "condition_distribution", "GCN")
    create_directory(conditions_dir)
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)

    batch_sizes = defaultConfigs.batch_sizes
    num_samples = defaultConfigs.num_samples
    num_sample = num_samples[batch_sizes.index(batch_size)] if batch_size in batch_sizes else None
    if args.dataset == "papers":
        num_sample = 1
    elif args.dataset == "products":
        num_sample = min(num_sample, 10)
    if args.mt == 0:
        starter = inkstream_gcn(model, intr_result_dir, aggregator=args.aggr, verify=False, out_channels=out_channels)
        condition_distribution, exec_time_dist = starter.batch_incremental_inference(data, niters=num_sample)
        unique_id = 0
        while osp.exists(osp.join(time_dir,f"GCN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_{unique_id}.npy")):
            unique_id += 1
        for it_layer in condition_distribution.keys():
            np.save(
                osp.join(
                    conditions_dir,
                    f"[tot_add_delno_cov_rec]GCN_{args.dataset}_{args.aggr}_{args.stream}_{batch_size}_layer{it_layer}_{unique_id}.npy",
                ),
                condition_distribution[it_layer],
            )
        np.save(
            osp.join(time_dir, f"GCN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_{unique_id}.npy"),
            exec_time_dist,
        )
    else:
        if not isinstance(args.id, int) and args.id >= 0:
            raise ValueError("missing argument --id when --mt is used.")
        starter = inkstream_gcn(model, osp.join(intr_result_dir, str(args.id)),
                                multi_thread=args.mt, aggregator=args.aggr, verify=False)
        condition, exec_time = starter.incremental_inference_mt(data)
        if condition is not None and exec_time is not None:
            for it_layer in condition.keys():
                with open(
                        osp.join(
                            conditions_dir,
                            f"[tot_add_delno_cov_rec]GCN_{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}_mt{args.mt}.txt",
                        ), "a") as f:
                    f.write(f'{condition[it_layer]["computed"]}\t{condition[it_layer]["add_only"]}\t{condition[it_layer]["del_no_change"]}\t{condition[it_layer]["covered"]}\t{condition[it_layer]["recompute"]}\n')
            with open(
                osp.join(time_dir, f"GCN_{args.dataset}_{args.aggr}_{args.stream}_batch_size_{batch_size}_mt{args.mt}.txt"), "a") as f:
                f.write(f"{exec_time}\n")


if __name__ == "__main__":
    main()
