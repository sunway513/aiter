#!/usr/bin/env python3
"""
FlyDSL RoPE Benchmark — outputs table format compatible with aiter-forge scoring.

Usage:
  python bench_flydsl_rope.py --tokens 128 --qheads 64 --kvheads 8 --dim 128 \
      --block-size 16 --layout flash --dtype bf16

Output format (aiter-forge compatible):
  TOKENS  QHEADS  KVHEADS  DIM  LAYOUT    LATENCY_US  GB/s
  128     64      8        128  flash     26.3        360.64
"""
import argparse
import os
import sys
import time
import torch

# Add FlyDSL and AITER to path
flydsl_root = os.environ.get("FLYDSL_ROOT", "/home/pensun/FlyDSL")
aiter_root = os.environ.get("AITER_ROOT", "/home/pensun/aiter_research")
sys.path.insert(0, flydsl_root)
sys.path.insert(0, aiter_root)

from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module


def bench_gpu_us(fn, warmup=10, iters=100):
    """Benchmark a GPU function, return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e3)  # ns → us

    times.sort()
    return times[len(times) // 2]


def compute_bandwidth(T, QH, KH, D, latency_us, elem_bytes=2):
    """Compute effective HBM bandwidth in GB/s."""
    # Read: Q + K + V + cos + sin per token
    # Write: Q_out + K_out + KeyCache + ValueCache
    read_bytes = T * (QH + KH + KH) * D * elem_bytes  # Q, K, V
    read_bytes += T * D * elem_bytes  # cos + sin (D/2 each, combined = D)
    write_bytes = T * (QH + KH) * D * elem_bytes  # Q_out, K_out
    write_bytes += T * KH * D * elem_bytes * 2  # KeyCache + ValueCache
    total_bytes = read_bytes + write_bytes
    return total_bytes / (latency_us * 1e-6) / 1e9


def main():
    parser = argparse.ArgumentParser(description="FlyDSL RoPE Benchmark")
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--qheads", type=int, required=True)
    parser.add_argument("--kvheads", type=int, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--layout", choices=["flash", "nonflash"], default="flash")
    parser.add_argument("--dtype", choices=["bf16", "f16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pos-int64", action="store_true", help="Use int64 positions (AITER compat)")
    parser.add_argument("--compare-aiter", action="store_true", help="Also benchmark AITER Triton")
    args = parser.parse_args()

    T = args.tokens
    QH, KH, D = args.qheads, args.kvheads, args.dim
    block_size = args.block_size
    flash_layout = args.layout == "flash"
    dtype_str = args.dtype
    torch_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    device = "cuda:0"
    max_pos = 8192

    # Build kernel
    launch_fn = build_fused_rope_cache_module(
        head_dim=D, num_q_heads=QH, num_kv_heads=KH,
        block_size=block_size, is_neox=True, flash_layout=flash_layout,
        dtype_str=dtype_str, pos_int64=args.pos_int64,
    )

    # Create tensors
    q = torch.randn(T, QH, D, dtype=torch_dtype, device=device)
    k = torch.randn(T, KH, D, dtype=torch_dtype, device=device)
    v = torch.randn(T, KH, D, dtype=torch_dtype, device=device)

    pos_dtype = torch.int64 if args.pos_int64 else torch.int32
    positions = torch.randint(0, min(max_pos, 4096), (T,), dtype=pos_dtype, device=device)
    slot_mapping = torch.arange(T, dtype=pos_dtype, device=device)

    half_dim = D // 2
    cos_cache = torch.randn(max_pos, half_dim, dtype=torch_dtype, device=device)
    sin_cache = torch.randn(max_pos, half_dim, dtype=torch_dtype, device=device)

    num_blocks = max(32, (T + block_size - 1) // block_size + 1)
    if flash_layout:
        key_cache = torch.zeros(num_blocks, block_size, KH, D, dtype=torch_dtype, device=device)
        value_cache = torch.zeros(num_blocks, block_size, KH, D, dtype=torch_dtype, device=device)
    else:
        x_pack = 16
        key_cache = torch.zeros(num_blocks, KH, D // x_pack, block_size, x_pack, dtype=torch_dtype, device=device)
        value_cache = torch.zeros(num_blocks, KH, D, block_size, dtype=torch_dtype, device=device)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    stream = torch.cuda.current_stream()

    def run_flydsl():
        launch_fn(q, k, v, positions, cos_cache, sin_cache, slot_mapping,
                  key_cache, value_cache, q_out, k_out, T, stream=stream)

    # Benchmark FlyDSL
    us = bench_gpu_us(run_flydsl, warmup=args.warmup, iters=args.iters)
    bw = compute_bandwidth(T, QH, KH, D, us)

    # Print aiter-forge compatible table
    print(f"{'TOKENS':>8}  {'QHEADS':>6}  {'KVHEADS':>7}  {'DIM':>4}  {'LAYOUT':>8}  {'LATENCY_US':>10}  {'GB/s':>10}")
    print(f"{T:>8}  {QH:>6}  {KH:>7}  {D:>4}  {args.layout:>8}  {us:>10.1f}  {bw:>10.2f}")

    # Optional AITER comparison
    if args.compare_aiter:
        try:
            from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache

            pos_i64 = positions.to(torch.int64)
            slot_i64 = slot_mapping.to(torch.int64)
            cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)
            sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)
            ks = torch.tensor(1.0, device=device)
            vs = torch.tensor(1.0, device=device)
            qa = q.clone()
            ka = k.clone()
            kca = key_cache.clone()
            vca = value_cache.clone()
            qoa = torch.empty_like(q)
            koa = torch.empty_like(k)

            def run_aiter():
                fused_qk_rope_reshape_and_cache(
                    qa, ka, v, kca, vca, slot_i64, pos_i64,
                    cos_4d, sin_4d, ks, vs,
                    is_neox=True, flash_layout=flash_layout,
                    apply_scale=False, q_out=qoa, k_out=koa, output_zeros=False,
                )

            us_aiter = bench_gpu_us(run_aiter, warmup=args.warmup, iters=args.iters)
            bw_aiter = compute_bandwidth(T, QH, KH, D, us_aiter)
            speedup = us_aiter / us

            print(f"\n{'':>8}  {'':>6}  {'':>7}  {'':>4}  {'AITER':>8}  {us_aiter:>10.1f}  {bw_aiter:>10.2f}")
            print(f"\nFlyDSL/AITER speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"\nAITER comparison failed: {e}")


if __name__ == "__main__":
    main()
