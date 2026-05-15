# Build-level vs API-level CK comparison — two different questions

- **Area**: tuning / infra / CK-removal
- **Kernel**: N/A (meta-level methodology note)
- **Shape**: all 5 op families tracked in `docs/ck_removal_tracker.md`
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis

After PR #41 landed ("CK vs non-CK across 5 op families × 29 shapes"),
it was tempting to conclude the CK-removal question was answered: rows
are green/yellow/red, per-op closure plan exists, ETA is set. The
implicit assumption was that those API-level numbers would **transfer
1:1 to a build-level switch** — i.e. if the CK backend for a given op is
≤10% from the non-CK backend, then flipping `ENABLE_CK=0` and running
production should cost ≤10% on that op.

## Result

That assumption is **wrong in both directions**. A separate audit (this
PR, `feat/ck-build-comparison`) rebuilt AITER with `ENABLE_CK=0` and
called the **dispatcher entry points** (not the internal `*_CK`
symbols) with a 20-warmup + 100-iter timer. Findings across 5 op
families × 19 production shapes (DSR1, MiniMax-M2.5, Kimi-K2, GLM-5):

| Bucket | Count | Families |
|---|---|---|
| 🟢 safe (CK-off ≥ 0.90× CK-on) | 13 | gemm_bf16 (4/4), gemm_fp8 (4/4), batched_gemm_bf16 (2/2), rmsnorm N≤8192 (3/3) |
| 🔴 broken | 6 | fused_moe_bf16 (5/5 BROKEN), rmsnorm N>8192 (1/1 BROKEN) |

Specifically:

1. **`aiter.fused_moe` is BROKEN at every shape** even though
   `ck_moe_stage1` vs FlyDSL was a tight API-level comparison (PR
   #41 tagged "MISSING" non-CK; still, one could have hoped CK-on
   would keep working on CK-off). It doesn't: `module_moe_sorting`
   (a common prerequisite of all MoE dispatch, not the stage1/2
   kernel) unconditionally includes `composable_kernel` headers
   that fail to resolve under `ENABLE_CK=0`. The failure is upstream
   of the stage1/2 kernel the tracker was discussing.

2. **`aiter.rmsnorm2d_fwd` is SAFE at N≤8192 even though PR #41
   flagged those shapes red.** PR #41 compared the HIP kernel
   (~2.6-3.7 TB/s) against the CK kernel (~4.5-6.7 TB/s) and flagged
   a 41-45% gap. But the dispatcher hardcodes `if input.shape[-1] >
   8192: rmsnorm2d_fwd_ck(...) else rms_norm_cu(...)` — at H=8192
   it already picks the HIP path (not >8192). So for the tracked
   models (MiniMax H=6144, DSR1 H=7168, Kimi H=7168, GLM-5 H=6144),
   the dispatcher never calls CK anyway and the build-level gap is
   ~0-8%. Only N>8192 shapes trip the red flag.

3. **`aiter.batched_gemm_bf16` is SAFE** even though CK was expected
   to be lost at build time. `module_batched_gemm_bf16` unexpectedly
   compiles under `ENABLE_CK=0` because its CK kernel sources don't
   use the `#if ENABLE_CK` guard — they include `composable_kernel/`
   headers directly from the submodule's include path, which remain
   on disk. Dispatched perf matches CK-on within noise.

4. **`aiter.gemm_a8w8` is SAFE** for the same reason as (3).

5. **Warmup is load-bearing.** A first-pass bench with 3-warmup
   reported CK-off rmsnorm and batched_gemm at 40-60% of CK-on —
   a dramatic regression. Re-run with 20-warmup: all within ±9%.
   The "regression" was the JIT compile of a fresh module being
   charged against the timed window. Any build-level bench must
   use enough warmup to fully amortise first-call JIT.

## Root cause

Three independent properties of AITER's extension system produce the
discrepancy:

**A. `ENABLE_CK` is only a compile-time flag for sources, NOT a Python-
dispatch switch.** Only `aiter/ops/mha.py` has `if not ENABLE_CK:
fallback(...)` guards. Every other dispatcher (`gemm_a8w8`,
`batched_gemm_bf16`, `rmsnorm2d_fwd`, `fused_moe` via `ck_moe_stage1`)
unconditionally calls the CK-backed module. When CK is compiled away
the Python wrapper still exists; its first invocation triggers a JIT
rebuild that fails (if the csrc dir uses the shim) or succeeds (if it
dodges the shim via `composable_kernel/` submodule headers).

**B. CK csrc dirs don't uniformly honour `#if !ENABLE_CK`.** The
shim `ck_tile_shim.h` was authored to cover the **FMHA** code path
only. The other CK modules are split:
  - `ck_gemm_a8w8/`, `ck_batched_gemm_bf16/` — include
    `composable_kernel/` headers directly from submodule; build fine
    under ENABLE_CK=0 since submodule stays on disk.
  - `py_itfs_ck/moe_sorting/`, `py_itfs_ck/rmsnorm2d_fwd.cpp` — include
    `ck_tile/` types that go through `aiter_hip_common.h`'s
    `#if !ENABLE_CK #include "ck_tile_shim.h"` switch; the shim is
    incomplete and the build fails.
The split is case-by-case and not documented.

**C. JIT-included-in-first-call-timing is a measurement trap.** If
the benchmark's warmup loop doesn't fully amortise the JIT compile,
the "CK-off slower" conclusion is indistinguishable from "first call
was slow because it was doing 2 minutes of hipcc". Big warmup
(20+ iters on a fresh process) is necessary to separate the two.

So "CK-on vs CK-off at the dispatcher entry" differs from "CK kernel
vs non-CK kernel at the same build" along two axes: **(i) Python
fallback routing presence/absence**, and **(ii) per-module build
tolerance for ENABLE_CK=0**.

## Reusable rule

**API-level comparisons (`bench_ck_removal.py`) tell you which backend
is faster when both are compiled in. Build-level comparisons
(`bench_ck_build_comparison.py`) tell you what will actually ship when
the user flips `ENABLE_CK=0`. These are different questions with
different answers; never substitute one for the other when gating a
release.**

Specifically: a dispatcher with no `if not ENABLE_CK: fallback` guard
in its Python wrapper is **unconditionally BROKEN at build level**,
regardless of how fast a "non-CK backend" is in API-level benchmarks.
Fixing the build-level gap requires patching the dispatcher, not just
authoring a faster non-CK kernel.

## When each is the right question

- **API-level (PR #41)**: planning which non-CK backend to tune next;
  gating "is the non-CK kernel mature enough to be the default when
  available?".
- **Build-level (this PR)**: gating a CK-off release; counting how
  many ops the dispatcher will fail to call; forecasting which
  dispatcher patches need to land before `ENABLE_CK=0` is safe.

## References

- Data: `results/ck_build_comparison/bench_results.jsonl`
- Tracker: `docs/ck_build_comparison.md`
- Audit: `docs/ck_build_audit.md`
- API-level comparator: `docs/ck_removal_tracker.md` / PR #41
- Bench harness: `scripts/bench_ck_build_comparison.py`
- ENABLE_CK plumbing: `aiter/jit/core.py:29, 793-795` + `csrc/include/ck_tile_shim.h`
