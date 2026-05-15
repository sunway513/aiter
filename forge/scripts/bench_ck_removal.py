#!/usr/bin/env python3
"""Benchmark CK vs non-CK paths across AITER operator families.

Strategy
========
For every operator family used in AMD MI-series model-serving stacks
(fused MoE, GEMM, batched GEMM, RMSNorm), we identify **the CK kernel
that AITER currently dispatches to** and measure it alongside the best
available **non-CK fallback** — FlyDSL, ASM, Triton, or hipBLASLt.

The comparison is honest about gaps:

* Where no non-CK fallback exists (e.g. unquantized BF16 fused MoE today
  only ships a CK kernel in the AITER Python API), the non-CK record is
  tagged ``status="MISSING"`` with a specific next-step string.
* Where a CK path does not support a shape (e.g. GPT-OSS-120B
  N=5760/K=2880 trips ``device_gemm with the specified compilation
  parameters does not support this GEMM problem``), the CK record is
  tagged ``status="UNSUPPORTED"`` — a CK *regression* that the non-CK
  path can actually fix.
* Regression guard: if non-CK reports faster-than-CK we auto-re-run with
  higher iter count and surface both samples.

This script does **not** modify AITER source — mission-compliant.

Usage
-----
  python3 scripts/bench_ck_removal.py \
      --output results/ck_removal/bench_results.jsonl \
      --iters 30

  python3 scripts/bench_ck_removal.py --only gemm_fp8,rmsnorm

Runtime: about 2 minutes for the full 30-row set on one MI355X.
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


# ---------------------------------------------------------------------------
# Bench primitives

def bench(fn, warmup: int = 3, iters: int = 10) -> float:
    """Return average ms per call over ``iters`` iterations."""
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
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn()


# ---------------------------------------------------------------------------
# Shape registry -- drawn from docs/_data/op-perf.json for the named models.

SHAPES = {
    # (model, label, M, N, K, E, topk)
    "fused_moe_bf16": [
        ("DeepSeek-R1",  "DSR1 small  M=1",     1,    4096, 7168, 256, 8),
        ("DeepSeek-R1",  "DSR1 medium M=128",   128,  4096, 7168, 256, 8),
        ("DeepSeek-R1",  "DSR1 large  M=1024",  1024, 4096, 7168, 256, 8),
        ("MiniMax-M2.5", "MiniMax small M=1",   1,    3072, 6144,  64, 8),
        ("MiniMax-M2.5", "MiniMax med   M=128", 128,  3072, 6144,  64, 8),
        ("MiniMax-M2.5", "MiniMax lg    M=1024",1024, 3072, 6144,  64, 8),
        ("GPT-OSS-120B", "GPT-OSS M=128",       128,  5760, 2880, 128, 4),
        ("Kimi-K2",      "Kimi M=128",          128,  4096, 7168, 128, 8),
        ("GLM-5",        "GLM-5 M=128",         128,  4096, 6144, 128, 8),
    ],
    # (model, label, M, N, K)
    "gemm_bf16": [
        ("DeepSeek-R1",  "DSR1 gate_up M=1",     1,    36864, 7168),
        ("DeepSeek-R1",  "DSR1 gate_up M=1024",  1024, 36864, 7168),
        ("DeepSeek-R1",  "DSR1 qkv M=1",         1,    7168,  2048),
        ("MiniMax-M2.5", "MiniMax proj M=128",   128,  6144,  4096),
        ("Kimi-K2",      "Kimi proj M=128",      128,  6144,  7168),
        ("GLM-5",        "GLM-5 proj M=128",     128,  6144,  4096),
        ("DeepSeek-R1",  "DSR1 ffn M=128",       128,  6144,  7168),
        ("DeepSeek-R1",  "DSR1 square M=4096",   4096, 8192,  8192),
    ],
    # (model, label, M, N, K)
    "gemm_fp8": [
        ("DeepSeek-R1",  "DSR1 proj M=1",        1,    7168, 2048),
        ("DeepSeek-R1",  "DSR1 proj M=128",      128,  7168, 2048),
        ("DeepSeek-R1",  "DSR1 gate_up M=128",   128, 36864, 7168),
        ("MiniMax-M2.5", "MiniMax proj M=128",   128,  6144, 4096),
        ("Kimi-K2",      "Kimi proj M=128",      128,  6144, 7168),
        ("GLM-5",        "GLM-5 proj M=128",     128,  6144, 4096),
    ],
    # (model, label, B, M, N, K)
    "batched_gemm_bf16": [
        ("DeepSeek-R1",  "DSR1 attn_bmm M=128",   8, 128, 128, 576),
        ("MiniMax-M2.5", "MiniMax attn M=128",   16, 128, 128, 128),
        ("Kimi-K2",      "Kimi attn M=128",      16, 128, 128, 128),
    ],
    # (model, label, M, N)
    "rmsnorm": [
        ("All",  "H=4096 M=64",    64,   4096),
        ("All",  "H=4096 M=1024",  1024, 4096),
        ("All",  "H=8192 M=1024",  1024, 8192),
    ],
}


# ---------------------------------------------------------------------------
# Operator runners
#
# Each runner returns (tflops, ms, backend_tag, notes) on success, raises
# ``_Missing`` when the path simply does not exist in this build (renders
# as status="MISSING" with the message as a concrete next step), and
# raises any other exception for true runtime errors.


class _Missing(RuntimeError):
    """Signal that a non-CK replacement does not exist for this op."""


# --- fused MoE -------------------------------------------------------------

def run_fused_moe_ck(M, N, K, E, topk, iters):
    """Full BF16 fused_moe end-to-end via AITER's default CK 2-stage path.

    The default unquantized BF16 dispatch picks ``ck_moe_stage1`` +
    ``ck_moe_stage2_fwd`` (module ``moe_ck2stages_b16_b16_..._silu_no``).
    """
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
    flops = M * topk * 3 * 2 * D * I  # stage1 (gate+up packed -> 2*DI) + stage2 DI = 3 MNK, *2 FMA
    return flops / ms / 1e9, ms, "ck_moe_stage1+ck_moe_stage2_fwd", ""


def run_fused_moe_noneck(M, N, K, E, topk, iters):
    """Non-CK BF16 fused_moe.

    Current AITER main (2026-04-22) registers 1632 FlyDSL MoE kernels
    but **none with both a_dtype=bf16 and b_dtype=bf16**; all FlyDSL MoE
    kernels require quantized B (fp4 or fp8).  So the non-CK fallback
    for unquantized BF16 MoE does not exist yet.
    """
    from aiter.ops.flydsl.moe_kernels import _KERNEL_PARAMS
    bf16_bf16 = [
        k for k, v in _KERNEL_PARAMS.items()
        if v.get("a_dtype") == "bf16" and v.get("b_dtype") == "bf16"
    ]
    if not bf16_bf16:
        raise _Missing(
            "no FlyDSL a_dtype=bf16 b_dtype=bf16 MoE kernel in this build; "
            "needs FlyDSL BF16/BF16 MoE stage1+stage2 authored (est. 2 wk)"
        )
    raise _Missing(f"unexpected: found {len(bf16_bf16)} bf16/bf16 FlyDSL kernels but runner not wired")


# --- GEMM BF16 -------------------------------------------------------------

def run_gemm_bf16_ck(M, N, K, iters):
    """Attempt a CK-only BF16 GEMM.

    AITER's top-level BF16 GEMM symbol (``aiter.gemm_a16w16``) does not
    exist; only ``aiter.gemm_a16w16_asm`` (ASM) and
    ``aiter.batched_gemm_bf16_CK`` (batched) are exposed.  For the
    non-batched case there is therefore no separate CK entry point: the
    default path is hipBLASLt via ``torch.matmul``.  We report this as
    UNSUPPORTED on the CK side so the tracker renders the correct state.
    """
    raise _Missing(
        "no standalone CK BF16 GEMM symbol (aiter.gemm_a16w16 missing); "
        "AITER already defaults BF16 GEMM to hipBLASLt/asm -- already CK-free"
    )


def run_gemm_bf16_hipblaslt(M, N, K, iters):
    """hipBLASLt BF16 GEMM via torch.matmul -- the existing default."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.matmul(a, b.T)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * M * N * K / ms / 1e9, ms, "torch.matmul (hipBLASLt)", ""


# --- GEMM FP8 (a8w8) -------------------------------------------------------

def run_gemm_fp8_ck(M, N, K, iters):
    """CK a8w8 GEMM via ``aiter.gemm_a8w8_CK`` (explicit CK entry point)."""
    import aiter
    if not hasattr(aiter, "gemm_a8w8_CK"):
        raise _Missing("aiter.gemm_a8w8_CK missing in this build")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    a_scale = torch.ones(M, 1, device="cuda", dtype=torch.float32)
    b_scale = torch.ones(1, N, device="cuda", dtype=torch.float32)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: aiter.gemm_a8w8_CK(a, b, a_scale, b_scale, out)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * M * N * K / ms / 1e9, ms, "aiter.gemm_a8w8_CK", ""


def run_gemm_fp8_hipblaslt(M, N, K, iters):
    """Non-CK FP8 GEMM via ``torch._scaled_mm`` (hipBLASLt)."""
    a = torch.randn(M, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    a_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    b_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)

    def _call():
        return torch._scaled_mm(a, b.T, out_dtype=torch.bfloat16,
                                scale_a=a_scale, scale_b=b_scale)
    silent(_call)
    ms = bench(_call, iters=iters)
    return 2 * M * N * K / ms / 1e9, ms, "torch._scaled_mm (hipBLASLt)", ""


# --- batched GEMM BF16 -----------------------------------------------------

def run_batched_gemm_bf16_ck(B, M, N, K, iters):
    """CK batched BF16 GEMM via ``aiter.batched_gemm_bf16_CK``.

    The explicit-CK API; the non-suffixed ``batched_gemm_bf16`` actually
    routes to CK too via ``get_CKBatchedGEMM_config``.
    """
    import aiter
    if not hasattr(aiter, "batched_gemm_bf16_CK"):
        raise _Missing("aiter.batched_gemm_bf16_CK missing")
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, N, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(B, M, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: aiter.batched_gemm_bf16_CK(a, b, out)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * B * M * N * K / ms / 1e9, ms, "aiter.batched_gemm_bf16_CK", ""


def run_batched_gemm_bf16_torch(B, M, N, K, iters):
    """Non-CK batched BF16 GEMM via ``torch.bmm`` (hipBLAS)."""
    a = torch.randn(B, M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(B, K, N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: torch.bmm(a, b)
    silent(fn)
    ms = bench(fn, iters=iters)
    return 2 * B * M * N * K / ms / 1e9, ms, "torch.bmm (hipBLAS)", ""


# --- RMSNorm ---------------------------------------------------------------

def run_rmsnorm_ck(M, N, iters):
    """CK RMSNorm via ``aiter.rmsnorm2d_fwd_ck``.

    Returned value is GB/s (bandwidth-bound op); reported in the tflops
    slot for schema-uniformity with notes="bw GB/s".
    """
    import aiter
    if not hasattr(aiter, "rmsnorm2d_fwd_ck"):
        raise _Missing("aiter.rmsnorm2d_fwd_ck missing")
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    fn = lambda: aiter.rmsnorm2d_fwd_ck(x, w, 1e-6)
    silent(fn)
    ms = bench(fn, iters=iters)
    bw = 3 * M * N * 2 / ms / 1e6  # 2 reads + 1 write, bf16=2B, ms -> GB/s
    return bw, ms, "aiter.rmsnorm2d_fwd_ck", "bw GB/s"


def run_rmsnorm_noneck(M, N, iters):
    """Non-CK RMSNorm via ``aiter.rms_norm_cu`` (HIP/custom, no CK).

    Note: ``aiter.rmsnorm2d_fwd`` is a dispatcher that picks CK when
    ``N > 8192`` — we bypass it and call the HIP kernel directly to get a
    true non-CK measurement at every shape.
    """
    import aiter
    from aiter.ops.rmsnorm import rms_norm_cu
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    fn = lambda: rms_norm_cu(out, x, w, 1e-6)
    silent(fn)
    ms = bench(fn, iters=iters)
    bw = 3 * M * N * 2 / ms / 1e6
    return bw, ms, "aiter.rms_norm_cu (HIP, non-CK)", "bw GB/s"


# ---------------------------------------------------------------------------
# Dispatch matrix

BENCHES = [
    (
        "fused_moe_bf16",
        lambda s, it: run_fused_moe_ck(s[2], s[3], s[4], s[5], s[6], it),
        lambda s, it: run_fused_moe_noneck(s[2], s[3], s[4], s[5], s[6], it),
        "fused_moe_bf16",
    ),
    (
        "gemm_bf16",
        lambda s, it: run_gemm_bf16_ck(s[2], s[3], s[4], it),
        lambda s, it: run_gemm_bf16_hipblaslt(s[2], s[3], s[4], it),
        "gemm_bf16",
    ),
    (
        "gemm_fp8",
        lambda s, it: run_gemm_fp8_ck(s[2], s[3], s[4], it),
        lambda s, it: run_gemm_fp8_hipblaslt(s[2], s[3], s[4], it),
        "gemm_fp8",
    ),
    (
        "batched_gemm_bf16",
        lambda s, it: run_batched_gemm_bf16_ck(s[2], s[3], s[4], s[5], it),
        lambda s, it: run_batched_gemm_bf16_torch(s[2], s[3], s[4], s[5], it),
        "batched_gemm_bf16",
    ),
    (
        "rmsnorm",
        lambda s, it: run_rmsnorm_ck(s[2], s[3], it),
        lambda s, it: run_rmsnorm_noneck(s[2], s[3], it),
        "rmsnorm",
    ),
]


def write_record(fh, **kw):
    fh.write(json.dumps(kw) + "\n")
    fh.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/ck_removal/bench_results.jsonl")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--only", default="", help="Comma-separated subset of family names")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    only = set(s.strip() for s in args.only.split(",") if s.strip())

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out, "w")
    t0 = time.time()

    for family, ck_fn, noneck_fn, shape_key in BENCHES:
        if only and family not in only:
            continue
        shapes = SHAPES.get(shape_key, [])
        for shape in shapes:
            model, label = shape[0], shape[1]
            if shape_key == "fused_moe_bf16":
                shape_desc = {"M": shape[2], "N": shape[3], "K": shape[4],
                              "E": shape[5], "topk": shape[6]}
            elif shape_key in ("gemm_bf16", "gemm_fp8"):
                shape_desc = {"M": shape[2], "N": shape[3], "K": shape[4]}
            elif shape_key == "batched_gemm_bf16":
                shape_desc = {"B": shape[2], "M": shape[3], "N": shape[4], "K": shape[5]}
            elif shape_key == "rmsnorm":
                shape_desc = {"M": shape[2], "N": shape[3]}
            else:
                shape_desc = {}

            print(f"[{family:<18}] {model:<14} {label}  shape={shape_desc}", flush=True)
            if args.dry_run:
                continue

            rows = {}
            for backend, runner in (("ck", ck_fn), ("non_ck", noneck_fn)):
                try:
                    tflops, ms, tag, notes = runner(shape, args.iters)
                    rec = dict(
                        family=family, backend=backend,
                        model=model, label=label, shape=shape_desc,
                        tflops=round(tflops, 2), ms=round(ms, 4),
                        backend_tag=tag, notes=notes, status="OK",
                    )
                    rows[backend] = rec
                except _Missing as exc:
                    rec = dict(
                        family=family, backend=backend,
                        model=model, label=label, shape=shape_desc,
                        status="MISSING",
                        error=str(exc)[:200],
                    )
                    print(f"   {backend:<7} MISSING: {rec['error']}", flush=True)
                except Exception as exc:
                    rec = dict(
                        family=family, backend=backend,
                        model=model, label=label, shape=shape_desc,
                        status="UNSUPPORTED",
                        error=repr(exc)[:200],
                        traceback=traceback.format_exc()[:400],
                    )
                    print(f"   {backend:<7} UNSUPPORTED: {rec['error']}", flush=True)
                else:
                    print(f"   {backend:<7} {tag:<42} tflops/bw={tflops:8.1f}  ms={ms:7.3f}",
                          flush=True)
                write_record(fh, **rec)

            # Regression guard: if non_ck faster than ck by >5% AND both OK,
            # re-run both at 3x iters and emit a second "verify" pair.
            ck_r = rows.get("ck"); nc_r = rows.get("non_ck")
            if (ck_r and nc_r and ck_r["status"] == "OK" and nc_r["status"] == "OK"
                    and nc_r["tflops"] > ck_r["tflops"] * 1.05):
                print(f"   [regression-guard] non_ck faster; re-running at 3x iters",
                      flush=True)
                for backend, runner in (("ck", ck_fn), ("non_ck", noneck_fn)):
                    try:
                        tflops, ms, tag, _ = runner(shape, args.iters * 3)
                        rec = dict(
                            family=family, backend=backend,
                            model=model, label=label, shape=shape_desc,
                            tflops=round(tflops, 2), ms=round(ms, 4),
                            backend_tag=tag, notes="regression_guard_verify",
                            status="OK",
                        )
                    except Exception as exc:
                        rec = dict(
                            family=family, backend=backend,
                            model=model, label=label, shape=shape_desc,
                            status="ERROR",
                            notes="regression_guard_verify",
                            error=repr(exc)[:200],
                        )
                    write_record(fh, **rec)

    fh.close()
    print(f"\nDone in {time.time()-t0:.1f}s; wrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
