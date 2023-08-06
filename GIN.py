# From https://github.com/pyg-team/pytorch_geometric/blob/master/examples/compile/gin.py

import os.path as osp
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool

from utils import *


class GIN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, args):
        super().__init__()
        self.save_int = args.save_int

        self.convs = torch.nn.ModuleList()
        for _ in range(5):
            mlp = MLP([in_channels, 64, 64], norm=None)  # original 32
            self.convs.append(GINConv(mlp, train_eps=False,
                              save_intermediate=self.save_int))
            in_channels = 64

        self.mlp = MLP([64, 64, out_channels], dropout=0.5)

    def forward(self, x, edge_index, batch, ptr=None):
        out_per_layer = {}
        intermediate_result_per_layer = defaultdict(
            lambda: defaultdict(lambda: torch.empty((0))))

        for i, conv in enumerate(self.convs):
            x, intermediate_result = conv(x, edge_index)

            if self.save_int:
                # must contains time info, could contain intermediate
                for key in intermediate_result:
                    intermediate_result_per_layer[f"layer{i+1}"][key] = intermediate_result[key][ptr[:-1]]
                # 这里处理一下，通过读取batch.ptr来获取只保存中心结点的信息
                # out_per_layer[f"conv{i+1}"] = x.detach()

            x = x.relu()
            # since out_per_layer is rarely used, I simply remove it.

        x = global_add_pool(x, batch)
        return self.mlp(x), out_per_layer, intermediate_result_per_layer


# def train(model, train_loader, optimizer):
#     model.train()

#     total_loss = 0
#     for data in tqdm(train_loader):
#         data = data.to(device)
#         optimizer.zero_grad()
#         out = model(data.x, data.edge_index, data.batch)
#         loss = F.cross_entropy(out, data.y)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * data.num_graphs
#     return total_loss / len(train_loader.dataset)


# @torch.no_grad()
# def test(model, loader):
#     model.eval()

#     total_correct = 0
#     for data in loader:
#         data = data.to(device)
#         pred = model(data.x, data.edge_index, data.batch).argmax(dim=-1)
#         total_correct += int((pred == data.y).sum())
#     return total_correct / len(loader.dataset)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     args = general_parser(parser)
#     dataset = load_dataset(args)

#     data = dataset[0]
#     add_mask(data)
#     # print_data(data)
#     timing_sampler(data, args)


#     model = pureGIN(data.x.shape[1], 2).to(device)

#     batch_size = 8 if multiprocessing.cpu_count() < 10 else 64

#     # Compile the model into an optimized version:
#     # Note that `compile(model, dynamic=True)` does not work yet in PyTorch 2.0, so
#     # we use `transforms.Pad` and static compilation as a current workaround.
#     # See: https://github.com/pytorch/pytorch/issues/94640
#     # model = torch_geometric.compile(model)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     available_model = []
#     name_prefix = f"{args.dataset}_GIN_{args.aggr}"
#     for file in os.listdir("examples/trained_model") :
#         if re.match(name_prefix + "_[0-9]+_[0-1]\.[0-9]+\.pt", file) :
#             available_model.append(file)

#     if len(available_model) == 0 :  # no available model, train from scratch
#         best_test_acc = 0
#         best_model_state_dict = None
#         patience = args.patience
#         it_patience = 0

#         node_indices = torch.arange(data.num_nodes)
#         train_indices = node_indices[:int(data.num_nodes * 0.8)]
#         test_indices = node_indices[int(data.num_nodes * 0.8):]
#         train_loader = EgoNetDataLoader(data, train_indices, batch_size=batch_size)
#         test_loader = EgoNetDataLoader(data, test_indices, batch_size=batch_size)

#         for epoch in range(1, args.epochs + 1):
#             loss = train(model, train_loader, optimizer)
#             train_acc = test(model, train_loader)
#             test_acc = test(model, test_loader)
#             print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
#                   f'Test: {test_acc:.4f}')
#             if best_test_acc < test_acc :
#                 best_test_acc = test_acc
#                 best_model_state_dict = model.state_dict()
#                 it_patience = 0
#             else :
#                 it_patience = it_patience + 1
#                 if it_patience >= patience :
#                     print(f"No accuracy improvement {best_test_acc} in {patience} epochs. Early stopping.")
#                     break
#         save(best_model_state_dict, epoch, best_test_acc, name_prefix)
#     else:
#         accuracy = [float(re.findall("[0-1]\.[0-9]+", model_name)[0]) for model_name in available_model if
#                     len(re.findall("[0-1]\.[0-9]+", model_name)) != 0]
#         index_best_model = np.argmax(accuracy)
#         model = load(model, available_model[index_best_model])
#         node_indices = torch.arange(data.num_nodes)
#         loader = EgoNetDataLoader(data, node_indices, batch_size=batch_size)
#         start = time.perf_counter()
#         test_acc = test(model, loader)
#         end = time.perf_counter()
#         print(f'Full Graph. Inference time: {end - start:.4f} seconds')
#         print(f'Test: {test_acc:.4f}')

#         available_model.pop(index_best_model)
#         clean(available_model)
