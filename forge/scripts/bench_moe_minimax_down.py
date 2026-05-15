#!/usr/bin/env python3
"""Reproduce the dashboard's MiniMax-M2.5 fused_moe 'down' 0.02 ratio.

Config: E=256, top_k=8, model_dim=N=384, inter_dim=K=1536, BF16.
Dashboard says MI355X ~0.2-1.4 TFLOPS, B300 ~10-75 TFLOPS.

Objective: confirm which aiter code path handles this shape and where the
overhead comes from.
"""
import argparse
import json
import sys
import time

import torch


def bench(fn, w=3, n=10):
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
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk

    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--e", type=int, default=256)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model-dim", type=int, default=384)
    ap.add_argument("--inter-dim", type=int, default=1536)
    args = ap.parse_args()

    M = args.m
    E = args.e
    K = args.topk
    D = args.model_dim
    I = args.inter_dim
    dev = "cuda"
    dt = torch.bfloat16

    print(f"M={M} E={E} topk={K} model_dim={D} inter_dim={I}", flush=True)
    torch.manual_seed(0)
    x = torch.randn(M, D, device=dev, dtype=dt)
    w1 = torch.randn(E, I * 2, D, device=dev, dtype=dt) * 0.02
    w2 = torch.randn(E, D, I, device=dev, dtype=dt) * 0.02
    gating_logits = torch.randn(M, E, device=dev, dtype=torch.float32)
    topk_weights, topk_ids = fused_topk(x, gating_logits, K, renormalize=True)

    # Warm up + bench
    def fn():
        return fused_moe(x, w1, w2, topk_weights, topk_ids,
                         activation=ActivationType.Silu, quant_type=QuantType.No)

    # One call to see any dispatch print
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        out = fn()
    torch.cuda.synchronize()
    print("first-call stderr/stdout capture:", repr(buf.getvalue())[:500], flush=True)

    # Timed
    ms = bench(fn, w=3, n=10)
    # FLOPS estimate: each token routed to top_k experts; per expert: w1 (M, 2I, D) + act + w2 (M, D, I)
    # Total: M * topk * (2 * D * 2I + 2 * I * D) = M * topk * D * I * 6
    flops = M * K * D * I * 6
    tflops = flops / ms / 1e9
    print(f"\nResult: {tflops:.3f} TFLOPS  ({ms:.3f} ms)", flush=True)
    print(f"Dashboard says MI355X={0.2 + M*0.01:.2f} TFLOPS (10ms range), B300={M*0.15:.1f} TFLOPS (0.19ms)", flush=True)
    print(f"Is dispatch hitting per-expert loop? dur/E/topk = {ms/E/K*1000:.3f} us per expert invocation", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
