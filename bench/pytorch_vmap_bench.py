"""Head-to-head PyTorch benchmark for vmap(value_and_grad) on the exact same MLP (16 -> 64 -> 4)."""
import time
import torch
import torch.nn.functional as F
from torch.func import vmap, grad_and_value

torch.set_default_dtype(torch.float64)

IN_DIM = 16
HIDDEN_DIM = 64
OUT_DIM = 4


def make_params():
    l1_w_count = HIDDEN_DIM * IN_DIM
    l1_b_count = HIDDEN_DIM
    l2_w_count = OUT_DIM * HIDDEN_DIM
    l2_b_count = OUT_DIM
    total = l1_w_count + l1_b_count + l2_w_count + l2_b_count
    return torch.randn(total) * 0.5


def single_sample_loss(params: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l1_w_end = HIDDEN_DIM * IN_DIM
    l1_b_end = l1_w_end + HIDDEN_DIM
    l2_w_end = l1_b_end + OUT_DIM * HIDDEN_DIM

    w1 = params[:l1_w_end].view(HIDDEN_DIM, IN_DIM)
    b1 = params[l1_w_end:l1_b_end]
    w2 = params[l1_b_end:l2_w_end].view(OUT_DIM, HIDDEN_DIM)
    b2 = params[l2_w_end:]

    h = F.relu(w1 @ x + b1)
    pred = torch.sigmoid(w2 @ h + b2)
    return torch.mean((pred - y) ** 2)


vmap_vg = vmap(grad_and_value(single_sample_loss), in_dims=(None, 0, 0))


def bench(fn, params, xs, ys, warmup=5, iters=25):
    for _ in range(warmup):
        grads, losses = fn(params, xs, ys)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        grads, losses = fn(params, xs, ys)
        _ = losses.sum().item()
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    times.sort()
    return times[0], times[len(times) // 2], times[min(len(times) - 1, (len(times) * 95) // 100)]


def main():
    params = make_params()
    print(f"PyTorch {torch.__version__} | threads={torch.get_num_threads()} | dtype=float64")
    print(f"{'Batch':<10} | {'Backend':<22} | {'Min (us)':>11} | {'Med (us)':>11} | {'p95 (us)':>11} | {'Samples/sec':>12}")
    print("-" * 90)

    for b in [64, 256, 1024, 4096]:
        xs = torch.randn(b, IN_DIM)
        ys = torch.rand(b, OUT_DIM)
        mn, md, p95 = bench(vmap_vg, params, xs, ys)
        sps = b / (md / 1e6)
        print(f"{b:<10} | {'PyTorch torch.func':<22} | {mn:11.1f} | {md:11.1f} | {p95:11.1f} | {sps:12.0f}")


if __name__ == "__main__":
    main()
