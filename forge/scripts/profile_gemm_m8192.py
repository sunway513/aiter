#!/usr/bin/env python3
"""Profile flydsl vs aiter_asm vs torch.matmul at M=8192 with rocprofv3.

Captures counters: SQ_WAVES, MFMA busy, LDS bank conflict, GMEM reads.
Outputs per-kernel summary so we can see why aiter_asm wins.
"""
import argparse
import json
import subprocess
import sys

import torch

KERNEL_CANDIDATES = [
    # (label, variant, params)
    ("flydsl_tuned",
     dict(tile_m=128, tile_n=128, tile_k=64, block_m_warps=2, block_n_warps=2,
          b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False,
          waves_per_eu=1, n_tile_repeat=2)),
    ("flydsl_default", dict(auto_shuffle_b=True)),
]


def run_once(label, cfg, M=8192):
    from aiter.ops.flydsl import flydsl_hgemm
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
    a = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)

    # warmup
    if label.startswith("flydsl"):
        for _ in range(3):
            _ = flydsl_hgemm(a, b, **cfg)
    elif label == "aiter_asm":
        out = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            gemm_a16w16_asm(a, b, out, **cfg)
    elif label == "torch_matmul":
        for _ in range(3):
            _ = torch.matmul(a, b.T)
    torch.cuda.synchronize()

    # Timed runs for profiling.
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    if label.startswith("flydsl"):
        for _ in range(20):
            _ = flydsl_hgemm(a, b, **cfg)
    elif label == "aiter_asm":
        out = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
        for _ in range(20):
            gemm_a16w16_asm(a, b, out, **cfg)
    elif label == "torch_matmul":
        for _ in range(20):
            _ = torch.matmul(a, b.T)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 20
    tf = 2 * M * M * M / ms / 1e9
    return tf, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    if args.label == "flydsl_tuned":
        tf, ms = run_once("flydsl_tuned", KERNEL_CANDIDATES[0][1])
    elif args.label == "flydsl_default":
        tf, ms = run_once("flydsl_default", KERNEL_CANDIDATES[1][1])
    elif args.label == "aiter_asm":
        tf, ms = run_once("aiter_asm", {"kernelName": "bf16gemm_bf16_tn_256x256", "splitK": 1})
    elif args.label == "torch_matmul":
        tf, ms = run_once("torch_matmul", {})
    else:
        raise ValueError(f"unknown label {args.label}")

    print(f"{args.label}: {tf:.2f} TFLOPS ({ms:.3f} ms)")


if __name__ == "__main__":
    sys.exit(main() or 0)
