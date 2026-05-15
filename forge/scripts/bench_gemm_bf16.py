#!/usr/bin/env python3
"""Replicate ARGUS paper Table 2 BF16 Square GEMM on MI355X.

Baselines:
  torch_matmul  — hipBLASLt via torch (production quality, what most apps use)
  aiter_asm     — gemm_a16w16_asm (hand-tuned ASM path)
  flydsl_default— flydsl_hgemm with default params (untuned FlyDSL)

Tuning: flydsl_hgemm with swept params. Predict-verify hypothesis per shape.

Output: one JSON line per (shape, variant, params) run.
Run inside container `gemm-tune-1` so FlyDSL + AITER + hipBLASLt are in scope.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch


def bench(fn, warmup: int = 5, iters: int = 30) -> float:
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def tflops(M: int, N: int, K: int, ms: float) -> float:
    return 2.0 * M * N * K / ms / 1e9


def run_one(M: int, N: int, K: int, variant: str, params: dict | None = None) -> dict:
    params = params or {}
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

    if variant == "torch_matmul":
        ms = bench(lambda: torch.matmul(a, b.T))
    elif variant == "aiter_asm":
        from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        ms = bench(lambda: gemm_a16w16_asm(a, b, out))
    elif variant == "flydsl_default":
        from aiter.ops.flydsl import flydsl_hgemm
        ms = bench(lambda: flydsl_hgemm(a, b, auto_shuffle_b=True))
    elif variant == "flydsl_tuned":
        from aiter.ops.flydsl import flydsl_hgemm
        ms = bench(lambda: flydsl_hgemm(a, b, auto_shuffle_b=True, **params))
    else:
        raise ValueError(f"unknown variant {variant}")

    return {
        "M": M, "N": N, "K": K,
        "variant": variant,
        "params": params,
        "ms": ms,
        "tflops": tflops(M, N, K, ms),
    }


def sweep_shape(M: int, N: int, K: int) -> list[dict]:
    """Predict-verify hypothesis + candidate grid per shape.

    Hypothesis: small shapes benefit from small tiles (more WGs, higher
    occupancy); large shapes benefit from large tiles + deeper pipeline.
    """
    out: list[dict] = []

    # Three baselines (AITER stock + hipBLASLt).
    for variant in ("torch_matmul", "aiter_asm", "flydsl_default"):
        try:
            out.append(run_one(M, N, K, variant))
        except Exception as exc:
            out.append({"M": M, "N": N, "K": K, "variant": variant,
                        "error": repr(exc)})

    # Candidate tile grid. Keep focused: 8-12 configs per shape based on size.
    if M <= 1024:
        grid = [
            dict(tile_m=64,  tile_n=64,  tile_k=64,  block_m_warps=1, block_n_warps=4),
            dict(tile_m=64,  tile_n=128, tile_k=64,  block_m_warps=1, block_n_warps=4),
            dict(tile_m=128, tile_n=64,  tile_k=64,  block_m_warps=2, block_n_warps=2),
            dict(tile_m=128, tile_n=128, tile_k=64,  block_m_warps=2, block_n_warps=2),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2),
            dict(tile_m=64,  tile_n=128, tile_k=128, block_m_warps=1, block_n_warps=4),
            dict(tile_m=128, tile_n=128, tile_k=64,  block_m_warps=2, block_n_warps=2, b_to_lds=True),
            dict(tile_m=128, tile_n=128, tile_k=64,  block_m_warps=2, block_n_warps=2, async_copy=True),
        ]
    elif M <= 4096:
        grid = [
            dict(tile_m=128, tile_n=128, tile_k=64,  block_m_warps=2, block_n_warps=2),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2),
            dict(tile_m=128, tile_n=256, tile_k=64,  block_m_warps=1, block_n_warps=4),
            dict(tile_m=256, tile_n=128, tile_k=64,  block_m_warps=4, block_n_warps=1),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, b_to_lds=True),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=256, tile_k=64,  block_m_warps=4, block_n_warps=2),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, async_copy=True),
            dict(tile_m=256, tile_n=128, tile_k=64,  block_m_warps=4, block_n_warps=1, b_to_lds=True),
        ]
    else:  # >= 8192
        grid = [
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2),
            dict(tile_m=256, tile_n=128, tile_k=64,  block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1),
            dict(tile_m=256, tile_n=256, tile_k=64,  block_m_warps=4, block_n_warps=2),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=2),
            dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, split_k=2),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, split_k=2),
            dict(tile_m=256, tile_n=256, tile_k=128, block_m_warps=4, block_n_warps=2, b_to_lds=True),
            dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, async_copy=True),
        ]

    for params in grid:
        try:
            out.append(run_one(M, N, K, "flydsl_tuned", params))
        except Exception as exc:
            out.append({"M": M, "N": N, "K": K, "variant": "flydsl_tuned",
                        "params": params, "error": repr(exc)})

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1024,2048,4096,8192,16384")
    ap.add_argument("--output", default="results/gemm_bf16/results.jsonl")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shapes = [int(s) for s in args.shapes.split(",")]
    with out_path.open("w") as f:
        for shape in shapes:
            print(f"=== M=N=K={shape} ===", flush=True)
            for result in sweep_shape(shape, shape, shape):
                f.write(json.dumps(result) + "\n")
                f.flush()
                if "error" in result:
                    print(f"  {result['variant']:15s} {result.get('params', {})}  ERROR: {result['error'][:80]}", flush=True)
                else:
                    p = result.get("params", {})
                    tag = " ".join(f"{k}={v}" for k, v in p.items()) if p else "(default)"
                    print(f"  {result['variant']:15s} {result['tflops']:7.2f} TFLOPS  {tag}", flush=True)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
