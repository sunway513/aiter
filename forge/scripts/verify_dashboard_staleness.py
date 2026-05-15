#!/usr/bin/env python3
"""Spot-check one worst-row per low-perf dashboard category to detect stale data."""
import json
import sys
import traceback

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


def silent(fn):
    import io, contextlib
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn()


def run_fused_activation(M, D, op):
    # gelu_and_mul / silu_and_mul — input [M, 2*D], output [M, D]
    import aiter
    x = torch.randn(M, 2 * D, device="cuda", dtype=torch.bfloat16)
    fn_map = {
        "gelu_and_mul": getattr(aiter, "gelu_and_mul", None),
        "silu_and_mul": getattr(aiter, "silu_and_mul", None),
        "fused_activation": getattr(aiter, "silu_and_mul", None),
    }
    fn = fn_map.get(op, None) or getattr(aiter, "silu_and_mul", None)
    out = torch.empty(M, D, device="cuda", dtype=torch.bfloat16)
    silent(lambda: fn(out, x))
    ms = bench(lambda: fn(out, x))
    bytes_moved = M * (2 * D + D) * 2
    return bytes_moved / ms / 1e9, ms


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/op-perf.json'))

    target_cats = [
        'Diffusion GEGLU',
        'Diffusion AdaLN-Zero',
        'Fused RMSNorm (production kernels)',
        'MLA Decode (DeepSeek-R1/V3)',
        'Fused Activations (SiLU+Mul, GELU+Mul)',
        'GEMM',
    ]
    print(f"{'Category':<38} | {'op':<24} {'config':<35} | {'dash_mi355x':>12} {'dash_b3':>9} {'dash_ratio':>10} | {'quick_check':>12}")
    print('-' * 145)

    for cat in d['categories']:
        if cat['name'] not in target_cats: continue
        rows = []
        for r in cat['results']:
            mi = r.get('devices', {}).get('mi355x', {})
            b3 = r.get('devices', {}).get('b300', {})
            if isinstance(mi, dict) and isinstance(b3, dict) and mi.get('tflops') and b3.get('tflops'):
                ratio = mi['tflops'] / b3['tflops']
                rows.append((ratio, r))
        rows.sort(key=lambda x: (x[0], id(x[1])))
        # Take the single worst row
        if not rows: continue
        _, r = rows[0]
        mi = r['devices']['mi355x']
        b3 = r['devices']['b300']
        op = r.get('op','')
        label = (r.get('label') or '')[:30]
        cfg = f"{label} M={r.get('M','')}N={r.get('N','')}K={r.get('K','')}"[:35]
        # Only run a check for fused_activation shapes we know how to run quickly
        quick = '(skip)'
        try:
            if op in ('gelu_and_mul', 'silu_and_mul', 'fused_activation'):
                M = r.get('M', 1024)
                D = r.get('N') or r.get('K') or 4096
                tf, ms = run_fused_activation(M, D, op)
                quick = f"{tf:.1f} GB/s"
        except Exception as e:
            quick = 'ERR'
        ratio = mi['tflops'] / b3['tflops']
        print(f"{cat['name'][:38]:<38} | {op:<24} {cfg:<35} | {mi['tflops']:>12.2f} {b3['tflops']:>9.2f} {ratio:>10.3f} | {quick:>12}")


if __name__ == "__main__":
    sys.exit(main() or 0)
