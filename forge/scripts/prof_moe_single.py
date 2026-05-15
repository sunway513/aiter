#!/usr/bin/env python3
"""Minimal MiniMax MoE runner for rocprof."""
import argparse
import sys

import torch


def main():
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk

    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    args = ap.parse_args()

    M, E, K, D, I = args.m, 256, 8, 384, 1536
    torch.manual_seed(0)
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, I * 2, D, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    topk_w, topk_ids = fused_topk(x, gating, K, renormalize=True)

    # Warmup (no prints)
    for _ in range(3):
        _ = fused_moe(x, w1, w2, topk_w, topk_ids,
                      activation=ActivationType.Silu, quant_type=QuantType.No)
    torch.cuda.synchronize()

    # Timed
    for _ in range(10):
        _ = fused_moe(x, w1, w2, topk_w, topk_ids,
                      activation=ActivationType.Silu, quant_type=QuantType.No)
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main() or 0)
