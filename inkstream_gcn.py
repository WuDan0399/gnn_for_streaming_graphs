from inkstream import inkstream
from utils import *
from GCN import GCN
from EventQueue import *

# class ignite_gcn(inkstream):
#     pass

def main() :
    # device = 'cpu'
    # args = FakeArgs(dataset="Cora", aggr="min", perbatch=1, stream="add", it=0)
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)
    data = dataset[0].to(device)
    model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes, args).to(device)
    model = load_available_model(model, args)

    if args.perbatch < 1 :
        batch_size = int(args.perbatch / 100 * data.num_edges)  # perbatch is [x]%, so divide by 100
    else :
        batch_size = int(args.perbatch)

    intr_result_dir = osp.join("examples", "intermediate", args.dataset, args.aggr, args.stream,
                               f"batch_size_{batch_size}")

    starter = inkstream(model, "model_configs/GCN.txt", intr_result_dir, verify=True)

    condition_distribution, exec_time_dist = starter.batch_incremental_inference(data)

    conditions_dir = osp.join("examples", "condition_distribution", "GCN")
    create_directory(conditions_dir)
    for it_layer in condition_distribution.keys() :
        np.save(osp.join(conditions_dir,
                         f"[tot_add_delno_cov_rec]GCN_{args.dataset}_{args.stream}_{batch_size}_layer{it_layer}.npy"),
                condition_distribution[it_layer])
    time_dir = osp.join("examples", "timing_result", "incremental")
    create_directory(time_dir)
    np.save(osp.join(time_dir, "GCN_"+"_".join(intr_result_dir.split("/")[2 :]) + f"_{args.it}.npy"), exec_time_dist)


if __name__ == '__main__' :
    main()