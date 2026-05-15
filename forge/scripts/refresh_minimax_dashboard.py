#!/usr/bin/env python3
"""Re-measure ALL MiniMax-M2.5 rows on current MI355X / AITER main.

Emits a JSON patch:
  [{ "op": ..., "label": ..., "M": ..., "N": ..., "K": ...,
     "dashboard_mi355x_tflops": ..., "refreshed_mi355x_tflops": ...,
     "dashboard_b300_tflops": ..., "speedup_over_dashboard": ... }, ...]

Skips ops where we can't reconstruct the exact workload (rmsnorm /
fused_activation / mha_* — dashboard didn't ship shape params for those).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

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
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn()


def run_gemm_bf16(M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    silent(lambda: torch.matmul(a, b.T))
    ms = bench(lambda: torch.matmul(a, b.T))
    return 2 * M * N * K / ms / 1e9, ms


def run_gemm_fp8(M, N, K):
    # Use aiter.ops.gemm_op_a8w8 if available; else skip
    try:
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8
        a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        a_scale = torch.ones(M, 1, device="cuda", dtype=torch.float32)
        b_scale = torch.ones(1, N, device="cuda", dtype=torch.float32)
        out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        silent(lambda: gemm_a8w8(a, b, a_scale, b_scale, out))
        ms = bench(lambda: gemm_a8w8(a, b, a_scale, b_scale, out))
        return 2 * M * N * K / ms / 1e9, ms
    except Exception as exc:
        return None, None


def run_fused_moe(M, N, K, E, topk, label):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk
    # Dashboard convention: in MoE row, N=model_dim or intermediate, K=the other
    # For MiniMax: model_dim=3072. Look at label to figure out.
    # "gate_up" label: N= inter_dim, K=model_dim. hidden=[M, model_dim=K]
    # "down" label: N=model_dim, K=inter_dim. hidden=[M, model_dim=N]
    # "gate_up+down" combined: measure full moe with model_dim=K, inter_dim=N.
    if label.startswith("down"):
        D = N; I = K
    elif label.startswith("gate_up"):
        D = K; I = N
    else:
        D = K; I = N  # fallback
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, I * 2, D, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw, ti = fused_topk(x, gating, topk, renormalize=True)
    fn = lambda: fused_moe(x, w1, w2, tw, ti,
                           activation=ActivationType.Silu, quant_type=QuantType.No)
    silent(fn)
    ms = bench(fn)
    # FLOPS for full MoE; for "down" or "gate_up" sub-label we approximate half the full op
    # Full fused_moe cost: M*topk * (2*I*D + I*D + D*I) = M*topk * D*I*4 approx plus act
    # Dashboard labels sub-operations but AITER's fused_moe is monolithic — report full.
    flops_full = M * topk * D * I * 6
    if label.startswith(("down", "gate_up")) and not label.startswith("gate_up+"):
        flops = flops_full / 2
    else:
        flops = flops_full
    return flops / ms / 1e9, ms


def run_row(cat, r):
    op = r.get('op','')
    label = r.get('label','') or ''
    M, N, K = r.get('M'), r.get('N'), r.get('K')
    E, topk = r.get('E'), r.get('top_k')
    try:
        if op == 'gemm_bf16':
            return run_gemm_bf16(M, N, K)
        if op == 'gemm_fp8':
            return run_gemm_fp8(M, N, K)
        if op == 'fused_moe':
            return run_fused_moe(M, N, K, E, topk, label)
        # fused_moe_fp8 / rmsnorm / fused_activation / mha_* : skip for now
        return None, None
    except Exception as exc:
        return None, f"ERR: {repr(exc)[:60]}"


def main():
    d = json.load(open('/tmp/op-perf.json'))
    rows = []
    for c in d['categories']:
        for r in c['results']:
            if 'minimax' in (r.get('model') or '').lower():
                rows.append((c['name'], r))
    print(f"Total MiniMax rows: {len(rows)}", flush=True)

    out = []
    skipped = 0
    t0 = time.time()
    for i, (cat, r) in enumerate(rows):
        mi = r.get('devices', {}).get('mi355x', {})
        b3 = r.get('devices', {}).get('b300', {})
        dash_mi = mi.get('tflops') if isinstance(mi, dict) else None
        dash_b3 = b3.get('tflops') if isinstance(b3, dict) else None
        new_tf, new_ms = run_row(cat, r)
        if new_tf is None:
            skipped += 1
            continue
        if isinstance(new_ms, str) and new_ms.startswith('ERR'):
            skipped += 1
            continue
        speedup = new_tf / dash_mi if dash_mi else None
        out.append({
            "category": cat,
            "op": r.get('op'),
            "label": r.get('label'),
            "M": r.get('M'), "N": r.get('N'), "K": r.get('K'),
            "E": r.get('E'), "top_k": r.get('top_k'),
            "dashboard_mi355x_tflops": dash_mi,
            "refreshed_mi355x_tflops": round(new_tf, 2),
            "refreshed_mi355x_ms": round(new_ms, 4),
            "dashboard_b300_tflops": dash_b3,
            "speedup_over_dashboard": round(speedup, 2) if speedup else None,
            "new_ratio_vs_b300": round(new_tf / dash_b3, 3) if dash_b3 else None,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate else 0
            print(f"  [{i+1}/{len(rows)}] elapsed={elapsed:.0f}s eta={eta:.0f}s  skipped={skipped}", flush=True)

    output = {
        "refreshed_at": "2026-04-22",
        "dashboard_snapshot": d.get('lastUpdated'),
        "device": "MI355X",
        "aiter_commit": "origin/main@6890159 + minimax tuned CSVs",
        "n_rows_refreshed": len(out),
        "n_rows_skipped": skipped,
        "rows": out,
    }
    path = Path("/tmp/minimax_dashboard_refresh.json")
    path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {path}  ({len(out)} refreshed, {skipped} skipped)", flush=True)

    # Summary
    if out:
        speedups = [r["speedup_over_dashboard"] for r in out if r.get("speedup_over_dashboard")]
        if speedups:
            import statistics
            print(f"Speedup vs dashboard: min={min(speedups):.2f}x median={statistics.median(speedups):.2f}x max={max(speedups):.2f}x")
        # New ratio distribution
        newr = [r["new_ratio_vs_b300"] for r in out if r.get("new_ratio_vs_b300")]
        if newr:
            below70 = sum(1 for x in newr if x < 0.7)
            print(f"New MI355X/B300 ratio: median={statistics.median(newr):.2f}, rows below 0.7x: {below70}/{len(newr)} ({100*below70/len(newr):.0f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
