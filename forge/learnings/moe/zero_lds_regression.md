# Naive LDS removal: +181% regression (M-dim reuse lost)

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=1024, tile 96x128x128
- **Date**: 2026-04-04
- **Confidence**: verified

## Hypothesis
Triton's `_moe_gemm_a8w8` uses zero LDS — it loads A directly from GMEM per K-step, relies on 8 waves × 2w/SIMD occupancy for latency hiding. Triton is faster at large shapes. Hypothesis: removing our LDS ping-pong path would match Triton's architecture and close the gap.

## Result
391us → **1098us (+181% regression)**. Accuracy preserved.

## Root cause
In our tile structure, A-tile is **reused across every MFMA step in the K loop** when staged in LDS — one GMEM load amortizes over many MFMAs. Removing LDS caused the same A bytes to be re-fetched from GMEM on every K iteration. Memory bandwidth became the bottleneck (GMEM issue count exploded).

Triton's zero-LDS works because its tile structure has different data-reuse geometry: A is laid out per-MFMA such that each operand is used once. Our tile reuses A across the K-loop. You cannot port "zero-LDS" as a flag — it's a whole-kernel architectural decision.

## Reusable rule
**Never remove an LDS path without first mapping the data-reuse factor.** Compute: bytes_loaded_from_GMEM × reuse_factor = effective BW requirement. If reuse_factor > 1 (A reused across K iters), LDS staging is not optional — removing it multiplies GMEM pressure.

Corollary: do not port architectural decisions (zero-LDS, swizzle patterns, etc.) between kernels without re-deriving them from first principles for the new tile shape.

## References
- Parallel sub-agent experiment, 2026-04-04
- Triton source: `_moe_gemm_a8w8` reference
- Summary: `docs/optimization_summary.md` §"Zero-LDS path (DONE — SEVERE REGRESSION)"
