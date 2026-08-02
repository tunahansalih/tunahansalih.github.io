"""Fixed-input attention benchmark for W7-T4.

Run once unchanged for the baseline. Then change only candidate_attention()
using exactly one technique: scaled_dot_product_attention, torch.compile,
bf16 (on supported hardware), or a batching change.
"""

import statistics
import time

import torch


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def baseline_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def candidate_attention(q, k, v):
    # Change only this function for the "after" measurement.
    return baseline_attention(q, k, v)


def measure(fn, q, k, v, warmup=10, repeats=30):
    for _ in range(warmup):
        fn(q, k, v)
    synchronize(q.device)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(q, k, v)
        synchronize(q.device)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main():
    torch.manual_seed(7)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    shape = (2, 8, 512, 64)  # batch, heads, tokens, head dimension
    q = torch.randn(shape, device=device)
    k = torch.randn(shape, device=device)
    v = torch.randn(shape, device=device)

    expected = baseline_attention(q, k, v)
    actual = candidate_attention(q, k, v)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-4)

    baseline_ms = measure(baseline_attention, q, k, v)
    candidate_ms = measure(candidate_attention, q, k, v)
    print(f"device={device} shape={shape}")
    print(f"baseline median:  {baseline_ms:.3f} ms")
    print(f"candidate median: {candidate_ms:.3f} ms")
    print(f"ratio: {baseline_ms / candidate_ms:.3f}x")


if __name__ == "__main__":
    main()
