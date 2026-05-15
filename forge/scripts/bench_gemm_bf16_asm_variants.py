#!/usr/bin/env python3
"""Sweep over AITER's pre-compiled asm BF16 GEMM variants.

Discovery: `aiter.ops.gemm_op_a16w16.gemm_a16w16_asm` accepts a `kernelName`
override. 24 pre-compiled kernels exist for gfx950 with tile sizes from
32x64 to 256x256, variants with bshuffle / pf3 (depth-3 prefetch) / splitk.

This sweep benchmarks every asm variant × split_K factor on the 5 ARGUS shapes
to find the actual-best asm config per shape. The "default" baseline in my
earlier benches used only the 256x256 kernel, which is bad for small M.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ASM_KERNELS = [
    "bf16gemm_bf16_tn_256x256",
    "bf16gemm_bf16_tn_256x256_bpreshuffle",
    "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_32x64_pf3_splitk",
    "bf16gemm_fp32bf16_tn_32x64_splitk_clean",
    "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_48x64_pf3_splitk",
    "bf16gemm_fp32bf16_tn_48x64_splitk_clean",
    "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_64x64_pf3_splitk",
    "bf16gemm_fp32bf16_tn_64x64_splitk_clean",
    "bf16gemm_fp32bf16_tn_80x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_80x64_splitk_clean",
    "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_96x64_pf3_splitk",
    "bf16gemm_fp32bf16_tn_96x64_splitk_clean",
    "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_clean",
    "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk",
    "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk_clean",
]


def bench(fn, w=5, n=25):
    torch.cuda.synchronize()
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="1024,2048,4096,8192,16384")
    ap.add_argument("--output", default="/tmp/gemm_bf16_asm_variants.jsonl")
    args = ap.parse_args()

    shapes = [int(s) for s in args.shapes.split(",")]
    out = Path(args.output)
    with out.open("w") as f:
        for M in shapes:
            N = K = M
            a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
            out_tensor = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
            print(f"\n=== M=N=K={M} ===", flush=True)
            best_tf = 0.0
            best_spec = None
            for kname in ASM_KERNELS:
                # Enumerate split_K factors — small tiles benefit more
                for splitK in (1, 2, 4, 8):
                    if K % splitK:
                        continue
                    try:
                        ms = bench(lambda: gemm_a16w16_asm(a, b, out_tensor, splitK=splitK, kernelName=kname))
                        tf = 2 * M * N * K / ms / 1e9
                        rec = {"M": M, "kernelName": kname, "splitK": splitK, "tflops": tf, "ms": ms}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
                        if tf > best_tf:
                            best_tf = tf
                            best_spec = (kname, splitK)
                            print(f"  NEW BEST {tf:7.2f}  {kname} splitK={splitK}", flush=True)
                    except Exception as exc:
                        rec = {"M": M, "kernelName": kname, "splitK": splitK, "error": repr(exc)[:120]}
                        f.write(json.dumps(rec) + "\n")
                        f.flush()
            print(f"  => best {best_tf:7.2f}  {best_spec}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
