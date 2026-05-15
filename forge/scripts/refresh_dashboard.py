#!/usr/bin/env python3
"""Generalized op-perf dashboard refresh for any model.

Usage:
  python3 refresh_dashboard.py --model="MiniMax-M2.5" --op-perf=/path/to/op-perf.json
                               --output=/path/to/refresh_<model>.json [--dry-run]

Takes a dashboard op-perf.json and re-measures every row matching the
model filter on the current MI355X / AITER main. Emits a JSON patch with
(dashboard_tflops, refreshed_tflops, speedup, new_ratio_vs_b300) per row.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
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
        return None, f"ERR: {repr(exc)[:60]}"


def run_fused_moe(M, N, K, E, topk, label):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk
    if label.startswith("down"):
        D = N; I = K
    elif label.startswith("gate_up"):
        D = K; I = N
    else:
        D = K; I = N
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, I * 2, D, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    gating = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw, ti = fused_topk(x, gating, topk, renormalize=True)
    fn = lambda: fused_moe(x, w1, w2, tw, ti,
                           activation=ActivationType.Silu, quant_type=QuantType.No)
    silent(fn)
    ms = bench(fn)
    flops_full = M * topk * D * I * 6
    flops = flops_full / 2 if (label.startswith(("down", "gate_up")) and not label.startswith("gate_up+")) else flops_full
    return flops / ms / 1e9, ms


def run_row(cat, r):
    op = r.get('op', '') or ''
    label = r.get('label', '') or ''
    M, N, K = r.get('M'), r.get('N'), r.get('K')
    E, topk = r.get('E'), r.get('top_k')
    try:
        if op == 'gemm_bf16':
            return run_gemm_bf16(M, N, K)
        if op == 'gemm_fp8':
            return run_gemm_fp8(M, N, K)
        if op == 'fused_moe':
            return run_fused_moe(M, N, K, E, topk, label)
        return None, "unsupported op (needs schema)"
    except Exception as exc:
        return None, f"ERR: {repr(exc)[:60]}"


def select_rows(op_perf, model_filter):
    rows = []
    ml = model_filter.lower() if model_filter else None
    for c in op_perf['categories']:
        for r in c['results']:
            model = (r.get('model') or '').lower()
            if ml and ml not in model:
                continue
            rows.append((c['name'], r))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Substring to match op-perf row 'model' (e.g. 'MiniMax-M2.5')")
    ap.add_argument("--op-perf", required=True, help="Path to dashboard op-perf.json")
    ap.add_argument("--output", required=True, help="Output path for refresh JSON patch")
    ap.add_argument("--dry-run", action="store_true", help="List rows but do not run benchmarks")
    args = ap.parse_args()

    op_perf = json.load(open(args.op_perf))
    rows = select_rows(op_perf, args.model)
    print(f"Matched {len(rows)} rows for model={args.model!r}", flush=True)

    if args.dry_run:
        from collections import Counter
        breakdown = Counter((c, r.get('op', '')) for c, r in rows)
        for (c, op), n in breakdown.most_common():
            print(f"  {c[:40]:<40} op={op:<18} {n} rows")
        return

    out_rows = []
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
        out_rows.append({
            "category": cat,
            "op": r.get('op'),
            "label": r.get('label'),
            "M": r.get('M'), "N": r.get('N'), "K": r.get('K'),
            "E": r.get('E'), "top_k": r.get('top_k'),
            "dashboard_mi355x_tflops": dash_mi,
            "refreshed_mi355x_tflops": round(new_tf, 2),
            "refreshed_mi355x_ms": round(new_ms, 4),
            "dashboard_b300_tflops": dash_b3,
            "speedup_over_dashboard": round(new_tf / dash_mi, 2) if dash_mi else None,
            "new_ratio_vs_b300": round(new_tf / dash_b3, 3) if dash_b3 else None,
        })
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(rows) - i - 1) / rate if rate else 0
            print(f"  [{i+1}/{len(rows)}] elapsed={elapsed:.0f}s eta={eta:.0f}s skipped={skipped}", flush=True)

    output = {
        "refreshed_at": time.strftime("%Y-%m-%d"),
        "dashboard_snapshot": op_perf.get('lastUpdated'),
        "model_filter": args.model,
        "device": "MI355X",
        "aiter_commit_hint": "origin/main + shipped tuned CSVs",
        "n_rows_refreshed": len(out_rows),
        "n_rows_skipped": skipped,
        "rows": out_rows,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output} ({len(out_rows)} refreshed, {skipped} skipped)", flush=True)

    if out_rows:
        sp = [r['speedup_over_dashboard'] for r in out_rows if r.get('speedup_over_dashboard')]
        nr = [r['new_ratio_vs_b300'] for r in out_rows if r.get('new_ratio_vs_b300')]
        if sp:
            print(f"speedup vs dashboard: median={statistics.median(sp):.2f}x  min={min(sp):.2f}x  max={max(sp):.2f}x")
        if nr:
            below = sum(1 for x in nr if x < 0.7)
            print(f"new MI355X/B300 ratio: median={statistics.median(nr):.2f}  below-0.7x={below}/{len(nr)} ({100*below/len(nr):.0f}%)")


if __name__ == "__main__":
    sys.exit(main() or 0)
