#!/usr/bin/env python3
"""Round 2 BF16 GEMM sweep: constraint-safe configs based on round 1 learnings.

Round 1 constraints discovered:
  - b_to_lds=True requires b_preshuffle=False
  - tile_m=tile_n=256 with (block_m_warps=4, block_n_warps=2) = 512 threads (over AMDGPU default limit)
  - split_k capacity: counter_count = M*N/(tile_n*split_k) must be <= 4096

Round 2 strategy:
  - Explore b_preshuffle=False combined with b_to_lds at medium shapes (+tile_m=256 big tiles)
  - Try smaller block_n_warps=1 with tile_256x256 (stays under 512 threads)
  - split_k with smaller tile_n to stay in counter budget
  - Add more stages proxies (async_copy + b_to_lds combos)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def bench(fn, warmup: int = 5, iters: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        _ = fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2.0 * M * N * K / ms / 1e9


def run_flydsl(M, N, K, params):
    from aiter.ops.flydsl import flydsl_hgemm
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ms = bench(lambda: flydsl_hgemm(a, b, auto_shuffle_b=True, **params))
    return tflops(M, N, K, ms), ms


def sweep_shape(M, N, K):
    # b_to_lds path requires b_preshuffle=False.
    b_lds_flags = {"b_to_lds": True, "b_preshuffle": False, "auto_shuffle_b": False}

    if M <= 1024:
        grid = [
            dict(tile_m=64,  tile_n=64,  tile_k=64,  block_m_warps=1, block_n_warps=4),
            dict(tile_m=64,  tile_n=128, tile_k=128, block_m_warps=1, block_n_warps=4),
            dict(tile_m=128, tile_n=64,  tile_k=128, block_m_warps=1, block_n_warps=4),
            dict(tile_m=128, tile_n=128, tile_k=64,  block_m_warps=2, block_n_warps=2, **b_lds_flags),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, **b_lds_flags),
            dict(tile_m=256, tile_n=128, tile_k=64,  block_m_warps=4, block_n_warps=1),
        ]
    elif M <= 4096:
        grid = [
            dict(tile_m=128, tile_n=256, tile_k=64,  block_m_warps=1, block_n_warps=4),
            dict(tile_m=128, tile_n=256, tile_k=128, block_m_warps=1, block_n_warps=4),
            dict(tile_m=256, tile_n=128, tile_k=64,  block_m_warps=4, block_n_warps=1, **b_lds_flags),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, **b_lds_flags),
            dict(tile_m=256, tile_n=256, tile_k=64,  block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=1),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, **b_lds_flags, async_copy=True),
            dict(tile_m=128, tile_n=256, tile_k=64,  block_m_warps=1, block_n_warps=4, async_copy=True),
        ]
    else:  # >= 8192
        grid = [
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=256, tile_k=64,  block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, **b_lds_flags),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=1, **b_lds_flags),
            dict(tile_m=256, tile_n=256, tile_k=64,  block_m_warps=4, block_n_warps=1, async_copy=True),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=1, async_copy=True),
            # split_k with small tile_n to stay in counter budget (max ~4096 counters).
            # At M=8192,N=8192,tile_n=256,split_k=2: counters = 8192*8192/256/2 = 131072 — still too big.
            # Need very small split_k / shape range. Skip split_k for now.
        ]

    results = []
    for params in grid:
        try:
            tf, ms = run_flydsl(M, N, K, params)
            results.append({"M": M, "N": N, "K": K, "variant": "flydsl_round2",
                            "params": params, "tflops": tf, "ms": ms})
        except Exception as exc:
            results.append({"M": M, "N": N, "K": K, "variant": "flydsl_round2",
                            "params": params, "error": repr(exc)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1024,2048,4096,8192,16384")
    ap.add_argument("--output", default="/tmp/gemm_bf16_round2.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shapes = [int(s) for s in args.shapes.split(",")]
    with out_path.open("w") as f:
        for shape in shapes:
            print(f"=== M=N=K={shape} ===", flush=True)
            for r in sweep_shape(shape, shape, shape):
                f.write(json.dumps(r) + "\n")
                f.flush()
                p = r["params"]
                cfg = f'{p.get("tile_m")}x{p.get("tile_n")}x{p.get("tile_k")} w{p.get("block_m_warps")}x{p.get("block_n_warps")}'
                if p.get("b_to_lds"): cfg += " b_lds"
                if p.get("async_copy"): cfg += " async"
                if "error" in r:
                    print(f"  {cfg:<40} ERROR: {r['error'][:60]}", flush=True)
                else:
                    print(f"  {cfg:<40} {r['tflops']:7.2f} TFLOPS", flush=True)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
