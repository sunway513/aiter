#!/usr/bin/env python3
"""Spot-check dashboard 'low perf' rows on current MI355X / latest AITER.

Picks the worst rows per category and reruns them to see if the dashboard
snapshot is stale.
"""
import json
import sys

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


def flops_moe(M, N, K, topk):
    return M * topk * N * K * 2  # approximation


def run_moe(M, N, K, E, topk, label):
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk
    D, I = N, K  # assume label's N=model_dim, K=inter_dim
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, I * 2, D, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw, ti = fused_topk(x, gating, topk, renormalize=True)

    def fn():
        return fused_moe(x, w1, w2, tw, ti,
                         activation=ActivationType.Silu, quant_type=QuantType.No)

    # Warmup silently
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        fn()
    torch.cuda.synchronize()
    ms = bench(fn, w=3, n=10)
    flops = 6 * M * topk * D * I  # stage1 2I*D*2 + act + stage2 D*I*2
    return flops / ms / 1e9, ms


def run_gemm_bf16(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ms = bench(lambda: torch.matmul(a, b.T))
    return 2 * M * N * K / ms / 1e9, ms


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/op-perf.json'
    d = json.load(open(path))

    # Pick worst-ratio MoE rows (unique by shape) — first 5
    for cat in d['categories']:
        if 'MoE' not in cat['name']: continue
        rows = []
        for r in cat['results']:
            mi = r.get('devices', {}).get('mi355x', {})
            b3 = r.get('devices', {}).get('b300', {})
            if isinstance(mi, dict) and isinstance(b3, dict) and mi.get('tflops') and b3.get('tflops'):
                ratio = mi['tflops'] / b3['tflops']
                rows.append((ratio, r))
        rows.sort(key=lambda x: (x[0], id(x[1])))
        seen = set()
        pick = []
        for ratio, r in rows:
            key = (r.get('M'), r.get('N'), r.get('K'), r.get('E'), r.get('top_k'), r.get('model'))
            if key in seen: continue
            seen.add(key); pick.append((ratio, r))
            if len(pick) >= 8: break
        print(f"\n=== {cat['name']} — spot-check 8 worst (current MI355X vs dashboard) ===\n")
        print(f'{"model":<22} {"M":>5} {"N":>5} {"K":>6} {"E":>4} {"tk":>3} | {"dash_ratio":>10} | {"dash_mi355x":>12} | {"our_mi355x":>11} | {"b300":>7} | {"delta":>5}')
        for ratio, r in pick:
            M, N, K, E, topk = r.get('M'), r.get('N'), r.get('K'), r.get('E'), r.get('top_k')
            try:
                new_tf, new_ms = run_moe(M, N, K, E, topk, r.get('label',''))
                status = f"{new_tf:7.2f}"
                dash_mi = r['devices']['mi355x']['tflops']
                dash_b3 = r['devices']['b300']['tflops']
                delta = new_tf / dash_mi if dash_mi else -1
                print(f"{r.get('model','')[:22]:<22} {M:>5} {N:>5} {K:>6} {E:>4} {topk:>3} | {ratio:>10.3f} | {dash_mi:>12.2f} | {status:>11} | {dash_b3:>7.2f} | {delta:>5.1f}x")
            except Exception as e:
                print(f"{r.get('model','')[:22]:<22} {M:>5} {N:>5} {K:>6} {E:>4} {topk:>3} | ERROR: {repr(e)[:50]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
