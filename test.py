import torch.nn.functional as F
from torch import Tensor, nn
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
get_time = time.time
unit = "seconds"
niter = 1000

print_style = "table"  # "readable"   "table"


def matmul_test():
    print("[Test] matrix multiplication (Linear)..")
    percentage = 0.1  # viewed as for all numbers within this threshold, redo the computation
    # Generating random tensor with shape (300, 256)
    for l in [200, 500, 800, 1000, 5000]:
        weight = torch.randn(256,l).to(device)
        bias = torch.randn(256).to(device)
        x = torch.randn(l).to(device)
        num_elements = int(x.numel() * percentage)  # Calculate the number of elements to make zero
        indices = torch.randperm(x.numel())[:num_elements]  # Randomly choose indices to make zero
        x2 = x.clone()
        x2[indices] = 0.01  # Change selected elements
        old_y = F.linear(x, weight, bias)  # old value

        if print_style == "table":
            print(f"[MatMul Runtime: {weight.shape} and {x.shape}, {niter} iters, {unit}, {device}] ")
        # Ground Truth Value
        start = get_time()
        for _ in range(niter):
            new_y = F.linear(x2, weight, bias)  # new value
        end = get_time()
        total_time = end - start
        if print_style == "readable":
            print(f'MatMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'GroundTruth\t {total_time:.4f}')

        #  opt 1: y' = Wx + Wδx + b , given y = Wx + b, and x' = x + δx. Directly use matrix multiplication with zeros
        start = get_time()
        for _ in range(niter) :
            # 模拟一个compare的步骤 对 x， 比如 前面改掉几个numbers，然后 相减，计算增量，增量的话，bias应该是0?
            delta_x = x2-x
            delta_y = F.linear(delta_x, weight)  # δy = Wδx
            new_y_inc = delta_y + old_y
        end = get_time()
        total_time = end - start
        if print_style == "readable":
            print(f'Incremental computation with Sparse MatMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'Incremental \t {total_time:.4f}')

        # opt 2: take those vectors out, then add it back.
        start = get_time()

        # 模拟一个compare的步骤 对 x， 比如 前面改掉几个numbers，然后 相减，计算增量，直接取出来NZ的部分，来做向量乘法，最后加起来
        for _ in range(niter) :
            diff_indices = torch.nonzero(x2 != x).flatten() # this output a 2-d array, so flatten for vector x
            delta_x2 = x2[diff_indices]-x[diff_indices]
            delta_y2 = F.linear(delta_x2, weight[:, diff_indices])
            new_y_vec = old_y + delta_y2

        end = get_time()
        total_time = end - start
        if print_style == "readable" :
            print(f'Incremental computation with VecMul of shape {weight.shape} and {x.shape} took {total_time:.4f} {unit} for {niter} rounds on {device}')
        else:
            print(f'Vector\t {total_time:.4f}')
        # Correctness test
        threshold = 1e-6

        # Compare the tensors within the threshold
        # print(new_y == new_y_inc)
        # print(new_y == new_y_vec)
        # print(torch.isclose(new_y, new_y_inc, atol=threshold))
        # print(torch.isclose(new_y, new_y_vec, atol=threshold))

if __name__ == '__main__':
    matmul_test()


