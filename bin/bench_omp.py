#!/usr/bin/env python3
"""OpenBLAS threading benchmark for Anggira's matrix sizes.

Tests whether OMP_NUM_THREADS=2 helps at dim=144 on ARM64 big cores.
Run with taskset to pin to big cores:

  OMP_NUM_THREADS=1 taskset -c 6,7 python3 bin/bench_omp.py
  OMP_NUM_THREADS=2 taskset -c 6,7 python3 bin/bench_omp.py
"""

import os
import time
import numpy as np

def bench_matmul(T, dim, n_trials=50, n_warmup=5):
    """Benchmark a single matmul of shape (T, dim) @ (dim, dim)."""
    A = np.random.randn(T, dim).astype(np.float32)
    B = np.random.randn(dim, dim).astype(np.float32)
    # warmup
    for _ in range(n_warmup):
        A @ B
    t0 = time.perf_counter()
    for _ in range(n_trials):
        A @ B
    return (time.perf_counter() - t0) / n_trials * 1000  # ms

def bench_attention(dim, heads, T, n_trials=20, n_warmup=3):
    """Benchmark attention score computation: Q @ K^T / sqrt(d_k)."""
    d_k = dim // heads
    Q = np.random.randn(T, heads, d_k).astype(np.float32)
    K = np.random.randn(T, heads, d_k).astype(np.float32)
    scale = d_k ** -0.5
    # warmup
    for _ in range(n_warmup):
        scores = np.einsum('thd, Thd -> hTt', Q, K) * scale
    t0 = time.perf_counter()
    for _ in range(n_trials):
        scores = np.einsum('thd, Thd -> hTt', Q, K) * scale
    return (time.perf_counter() - t0) / n_trials * 1000  # ms

def main():
    threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    print(f"🔧  OMP_NUM_THREADS={threads}")
    print(f"    Core pin: taskset -c 6,7 (ARM big cores)")
    print()

    dim = 144
    heads = 6
    context_lengths = [64, 128, 256, 512]

    print(f"{'Context':>8} | {'Matmul (ms)':>11} | {'Attention (ms)':>14} | {'Total (ms)':>10}")
    print("-" * 55)

    results = {}
    for T in context_lengths:
        mm_time = bench_matmul(T, dim)
        attn_time = bench_attention(dim, heads, T)
        total = mm_time + attn_time
        results[T] = {'matmul': mm_time, 'attention': attn_time, 'total': total}
        print(f"{T:>8} | {mm_time:>11.2f} | {attn_time:>14.2f} | {total:>10.2f}")

    print()

    # Compare with T=512 baseline
    if 512 in results:
        baseline = results[512]['total']
        print(f"{'Context':>8} | {'Speedup vs T=512':>17}")
        print("-" * 30)
        for T in context_lengths:
            speedup = baseline / results[T]['total']
            print(f"{T:>8} | {speedup:>16.1f}x")

    # Per-step estimate
    # Forward pass: ~10 matmuls per block, 5 blocks + attention = ~50 matmuls + 5 attn
    print(f"\n📊  Estimated per-step forward pass (5 blocks, 8 heads equivalent):")
    for T in context_lengths:
        mm_per_step = results[T]['matmul'] * 50  # ~50 matmuls
        attn_per_step = results[T]['attention'] * 5  # ~5 attention layers
        total_per_step = mm_per_step + attn_per_step
        print(f"    T={T:>3}: {total_per_step:>8.1f}ms forward (matmul: {mm_per_step:.1f}, attn: {attn_per_step:.1f})")

if __name__ == '__main__':
    main()
