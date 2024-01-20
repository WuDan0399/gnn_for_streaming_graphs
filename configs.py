import torch
import os


class defaultConfigs:
    batch_sizes = [1, 10, 100, 1000, 10000]
    num_samples = [100, 100, 10, 4, 1]

loader_configs = {
    "batch_size": 16, # 16, #for inf with full nghbr, # 2048 for training with neighbor sampler
    "num_workers": 4,
}
# root = os.getenv("DYNAMIC_GNN_ROOT")
root = "/home/dan/GNN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'  # for debugging
torch.autograd.set_detect_anomaly(True)
