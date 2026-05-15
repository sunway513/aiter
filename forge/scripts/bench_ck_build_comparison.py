#!/usr/bin/env python3
"""Build-level CK-on vs CK-off benchmark for AITER dispatcher entries.

Difference from ``bench_ck_removal.py`` (PR #41)
================================================
``bench_ck_removal.py`` measures the CK vs non-CK **backends** inside
the *same* AITER build with CK compiled in.  It answers: "given both
backends exist, which is faster?"

This script answers a different question: "if we rebuild AITER with
``ENABLE_CK=0``, does the same dispatcher call still work, and if so,
how fast?"  It targets the **user-facing dispatcher entries**
(``aiter.fused_moe``, ``aiter.gemm_a8w8``, ``aiter.rmsnorm2d_fwd``,
``aiter.batched_gemm_bf16``, ``torch.matmul``) rather than the
internal ``*_CK`` symbols.

Run modes
---------
``--mode ck_on``  (default)  -- use the CK-enabled build (from
    ``AITER_HOME`` env or default /app/aiter-test), records one row
    per (family, shape) via the dispatcher entry.
``--mode ck_off`` -- use the CK-off build (``AITER_HOME`` should point
    at a fresh clone/copy with ``ENABLE_CK=0`` set).  Records the same
    rows; rows that raise become ``status="BROKEN"`` with the error
    string.

Outputs one JSONL per run.  Downstream merge into
``results/ck_build_comparison/bench_results.jsonl``.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


def bench(fn, warmup: int = 20, iters: int = 100) -> float:
    """20-warmup + 100 timed iters.

    The big warmup is load-bearing on the CK-off build: any first-call
    JIT compile gets folded into warmup rather than into the timed
    window. 3-warmup is too low when the benchmark is also the JIT
    trigger.
    """
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def silent(fn):
    """Run fn, swallowing stdout/stderr so JIT compile noise doesn't taint records."""
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn()


# ---------------------------------------------------------------------------
# Shapes — identical to bench_ck_removal.py so rows line up 1:1.
SHAPES = {
    "fused_moe_bf16": [
        ("DeepSeek-R1",  "DSR1 M=1",      1,    4096, 7168, 256, 8),
        ("DeepSeek-R1",  "DSR1 M=128",    128,  4096, 7168, 256, 8),
        ("DeepSeek-R1",  "DSR1 M=1024",   1024, 4096, 7168, 256, 8),
        ("MiniMax-M2.5", "MiniMax M=128", 128,  3072, 6144,  64, 8),
        ("Kimi-K2",      "Kimi M=128",    128,  4096, 7168, 128, 8),
    ],
    "gemm_bf16": [
        ("DeepSeek-R1",  "DSR1 M=1",     1,    36864, 7168),
        ("DeepSeek-R1",  "DSR1 M=1024",  1024, 36864, 7168),
        ("DeepSeek-R1",  "DSR1 M=128",   128,   7168, 2048),
        ("DeepSeek-R1",  "DSR1 sq 4096", 4096,  8192, 8192),
    ],
    "gemm_fp8": [
        ("DeepSeek-R1",  "DSR1 M=1",     1,    7168,  2048),
        ("DeepSeek-R1",  "DSR1 M=128",   128,  7168,  2048),
        ("DeepSeek-R1",  "DSR1 wide-N",  128, 36864,  7168),
        ("MiniMax-M2.5", "MiniMax M=128", 128, 6144, 4096),
    ],
    "batched_gemm_bf16": [
        ("DeepSeek-R1",  "DSR1 attn K=576",  8, 128, 128, 576),
        ("MiniMax-M2.5", "MiniMax attn",    16, 128, 128, 128),
    ],
    "rmsnorm": [
        ("All", "H=4096 M=64",    64,   4096),
        ("All", "H=4096 M=1024",  1024, 4096),
        ("All", "H=8192 M=1024",  1024, 8192),
        ("All", "H=16384 M=128",  128, 16384),  # forces CK path on CK-on (>8192 dispatcher threshold)
    ],
}


# ---------------------------------------------------------------------------
# Runners -- call the *dispatcher entry*, not ck-specific symbols.
def run_fused_moe_dispatch(M, N, K, E, topk, iters):
    import aiter
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe, fused_topk
    D, I = K, N
    x = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(E, I * 2, D, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    g = torch.randn(M, E, device="cuda", dtype=torch.float32)
    tw, ti = fused_topk(x, g, topk, renormalize=True)

    fn = lambda: fused_moe(
        x, w1, w2, tw, ti,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
    )
    silent(fn)
    ms = bench(fn, iters=iters)
    flops = M * topk * 3 * 2 * D * I
    return flops / ms / 1e9, ms, "aiter.fused_moe(QuantType.No)"


def run_gemm_bf16_dispatch(M, N, K, iters):
    """AITER's documented BF16 GEMM entry is `torch.matmul` (via TunedGemm).
    This is what production code calls."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.matmul(a, b.T)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * M * N * K / ms / 1e9, ms, "torch.matmul"


def run_gemm_fp8_dispatch(M, N, K, iters):
    import aiter
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, 1, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(1, N, device="cuda", dtype=torch.float32)
    fn = lambda: aiter.gemm_a8w8(a, b, a_scale, b_scale)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * M * N * K / ms / 1e9, ms, "aiter.gemm_a8w8"


def run_batched_gemm_bf16_dispatch(B, M, N, K, iters):
    import aiter
    # dispatcher: aiter.batched_gemm_bf16 requires pre-allocated output
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(B, M, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: aiter.batched_gemm_bf16(a, b, out)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * B * M * N * K / ms / 1e9, ms, "aiter.batched_gemm_bf16"


def run_rmsnorm_dispatch(M, N, iters):
    """Dispatcher entry: `aiter.rmsnorm2d_fwd`.  Internally picks CK when
    `input.shape[-1] > 8192`, HIP otherwise.  This is the one row where
    shape matters for the build-level decision."""
    import aiter
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: aiter.rmsnorm2d_fwd(x, w, 1e-6)
    silent(fn)
    ms = bench(fn, iters=iters)
    bw = 3 * M * N * 2 / ms / 1e6  # GB/s (bf16=2B, 2 reads + 1 write)
    return bw, ms, "aiter.rmsnorm2d_fwd"


# ---------------------------------------------------------------------------
BENCHES = [
    ("fused_moe_bf16",    run_fused_moe_dispatch,          "fused_moe_bf16"),
    ("gemm_bf16",         run_gemm_bf16_dispatch,          "gemm_bf16"),
    ("gemm_fp8",          run_gemm_fp8_dispatch,           "gemm_fp8"),
    ("batched_gemm_bf16", run_batched_gemm_bf16_dispatch,  "batched_gemm_bf16"),
    ("rmsnorm",           run_rmsnorm_dispatch,            "rmsnorm"),
]


def shape_desc(shape_key, shape):
    if shape_key == "fused_moe_bf16":
        return {"M": shape[2], "N": shape[3], "K": shape[4], "E": shape[5], "topk": shape[6]}
    if shape_key in ("gemm_bf16", "gemm_fp8"):
        return {"M": shape[2], "N": shape[3], "K": shape[4]}
    if shape_key == "batched_gemm_bf16":
        return {"B": shape[2], "M": shape[3], "N": shape[4], "K": shape[5]}
    if shape_key == "rmsnorm":
        return {"M": shape[2], "N": shape[3]}
    return {}


def call_runner(runner, shape_key, shape, iters):
    if shape_key == "fused_moe_bf16":
        return runner(shape[2], shape[3], shape[4], shape[5], shape[6], iters)
    if shape_key in ("gemm_bf16", "gemm_fp8"):
        return runner(shape[2], shape[3], shape[4], iters)
    if shape_key == "batched_gemm_bf16":
        return runner(shape[2], shape[3], shape[4], shape[5], iters)
    if shape_key == "rmsnorm":
        return runner(shape[2], shape[3], iters)
    raise ValueError(shape_key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True,
                    help="JSONL output path (one row per family x shape)")
    ap.add_argument("--mode", choices=["ck_on", "ck_off"], required=True,
                    help="Label rows with which build was used; does not change "
                         "the actual ENABLE_CK env var (set that externally)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--only", default="",
                    help="Comma-separated subset of family names")
    ap.add_argument("--timeout-per-call-s", type=float, default=120.0,
                    help="Per-call wall-clock timeout for JIT + warmup + timing")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    t0 = time.time()

    # Record build mode + env once at top for provenance.
    try:
        import aiter
        enable_ck = aiter.jit.core.ENABLE_CK
    except Exception:
        enable_ck = None
    fh.write(json.dumps({
        "record_kind": "provenance",
        "mode": args.mode,
        "env_ENABLE_CK": os.environ.get("ENABLE_CK", "<unset>"),
        "aiter_ENABLE_CK": enable_ck,
        "aiter_path": sys.modules.get("aiter").__file__ if "aiter" in sys.modules else None,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }) + "\n")
    fh.flush()

    for family, runner, shape_key in BENCHES:
        if only and family not in only:
            continue
        for shape in SHAPES[shape_key]:
            model, label = shape[0], shape[1]
            sd = shape_desc(shape_key, shape)
            print(f"[{args.mode}][{family:<18}] {model:<14} {label}  shape={sd}",
                  flush=True)
            try:
                tflops, ms, tag = call_runner(runner, shape_key, shape, args.iters)
                rec = dict(
                    record_kind="bench", mode=args.mode,
                    family=family, model=model, label=label, shape=sd,
                    tflops=round(tflops, 2), ms=round(ms, 4),
                    dispatcher_tag=tag, status="OK",
                )
                print(f"   OK   {tag:<36} tflops/bw={tflops:8.1f}  ms={ms:7.3f}",
                      flush=True)
            except Exception as exc:
                rec = dict(
                    record_kind="bench", mode=args.mode,
                    family=family, model=model, label=label, shape=sd,
                    status="BROKEN",
                    error=repr(exc)[:400],
                    traceback=traceback.format_exc()[-600:],
                )
                print(f"   BROKEN: {rec['error'][:200]}", flush=True)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    fh.close()
    print(f"\nDone in {time.time()-t0:.1f}s; wrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
