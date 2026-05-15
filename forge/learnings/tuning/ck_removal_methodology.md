# Methodology for CK-removal perf parity audits

- **Area**: tuning / infra
- **Kernel**: N/A (applies to any backend-swap audit)
- **Shape**: 5 op families × 29 MI355X production shapes
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis

"Based on what I have heard from the team, it should be very close" —
the prevailing assumption at the start of this audit was that removing
the Composable Kernel (CK) dependency from AITER would be a drop-in
swap: FlyDSL / hipBLASLt / HIP / Triton kernels exist for every CK op,
and their perf is "very close". If true, we could land a zero-CK AITER
build today with a single env flag.

## Result

Across 5 operator families and 29 MI355X production shapes
(DeepSeek-R1, MiniMax-M2.5, GPT-OSS-120B, Kimi-K2, GLM-5):

| Bucket | Count | Examples |
|---|---|---|
| Green (gap ≤ 10% OR CK unused) | 7 | BF16 GEMM (all 8 shapes — already CK-free), FP8 a8w8 M=128 on DSR1 (+8%), batched GEMM B=8/K=576 |
| Yellow (10%-30% gap) | 5 | FP8 a8w8 M=1 (-26%), batched GEMM B=16/K=128 (-14%), RMSNorm M=64/H=4096 (+24%) |
| Red (> 30% gap OR no path) | 17 | **All 9 BF16 fused MoE rows (no non-CK path)**, RMSNorm at M=1024 (-42% to -45%), FP8 a8w8 at wider shapes (non-CK actually 39-78% faster — i.e. CK is a regression) |

The "very close" statement is **true for 2 of 5 families** (BF16 GEMM is
already CK-free; FP8 a8w8 at M=128 is a net-positive swap). It is **false
for 3 of 5**: BF16 fused MoE has no non-CK path at all, RMSNorm loses
~40% at large shapes, and FP8 a8w8 has a 26% regression at M=1.

Data provenance — every row was confirmed:

1. Dispatching the claimed kernel via `torch.profiler` kineto trace
   (`results/ck_removal/kernel_sanity.json`).
2. Measured over 100 iters with CUDA events; regression-guard re-runs
   at 300 iters any case where non-CK appeared faster to rule out
   noise.
3. Any run-time error wasn't swallowed: CK UNSUPPORTED at GPT-OSS and
   FlyDSL MISSING at BF16 MoE are both explicitly surfaced in the
   tracker, not papered over.

## Root cause

Three independent effects drive the spread.

**A. Some AITER Python APIs are already CK-free.** `aiter.gemm_a16w16`
was never exposed as a standalone Python symbol; `aiter.tuned_gemm.mm`
dispatches through hipBLASLt / ASM / FlyDSL with **no** CK fallback
for BF16. Result: the "remove CK" task for this family is vacuous.

**B. hipBLASLt has overtaken CK for FP8 GEMM on wide N.**
`torch._scaled_mm` uses Tensile's `Cijk_Alik_Bljk_F8BS_..._MT64x64x512`
kernel — a fresh FP8 Tensile config that lands 39%-78% more TFLOPS than
`ck::kernel_gemm_xdl_cshuffle_v3_multi_d` on M=128, N=6144..36864. This
is a CK **regression** that has been hidden because AITER never
benchmarked the two paths side-by-side on current hipBLASLt. Removing
CK here is a perf improvement, not a cost.

**C. CK still dominates two narrow but important paths.**

- **BF16 fused MoE (unquantized)**: FlyDSL has 1632 MoE kernels in the
  registry and 0 support `a_dtype=bf16, b_dtype=bf16`. Without a new
  kernel, BF16 MoE breaks entirely when CK is removed.
- **RMSNorm at large shapes**: `ck_tile::Rmsnorm2dFwdPipelineOnePass`
  fuses the variance-reduction and the normalize pass into one
  persistent kernel; the HIP `rms_norm_cu` does two HBM reads. On
  bandwidth-bound shapes (M=1024, H=4096..8192) this is worth
  2.6 → 4.5-6.7 TB/s.

## Reusable rule

**Do NOT trust a "should be very close" claim about a backend swap
without a per-op × per-shape measurement.** Before proposing to remove
any dependency from a perf-critical library:

1. **Enumerate** the Python symbols that reference the dependency
   (`grep -E "ck_|_CK|composable_kernel"` on `aiter/__init__.py`).
2. **Probe** — `hasattr(aiter, name)` each candidate in the target
   container. Many symbols advertised in source don't compile into the
   wheel; the real surface is smaller than you think.
3. **Pair-bench** each live symbol against its intended replacement on
   a shape grid drawn from **production** (the repo's dashboard, not
   synthetic shapes).
4. **Sanity-tag every row** via `torch.profiler` kineto: the kernel
   name *must* contain the backend's signature (`ck::`, `Cijk_`,
   `aiter::`, `ck_tile::`). If CK silently falls back, your non-CK
   numbers are a lie.
5. **Auto-verify suspicious wins**: if the "non-dependency" backend
   reports a speed-up, re-run both at 3× iters before celebrating.
   Most "big wins" at sub-20μs call times are measurement noise; the
   real ones (FP8 GEMM +50-70%) survive the re-run.
6. **Don't patch the upstream library from the audit repo** —
   aiter-forge's mission is to measure and file issues, not to land
   new kernels. Record the gap as a specific, owner-tagged next step
   in a closure plan.

Corollary: the aiter-forge `dispatcher` should grow a first-class
`backend_swap_audit` task YAML that takes `(symbol_a, symbol_b, shape
grid)` tuples, sanity-tags, auto-reruns regressions, and emits the
green/yellow/red matrix. Items 2-5 above are the schema; the only
novelty is wiring.

## References

- Tracker: `docs/ck_removal_tracker.md` (5 families × 29 shapes,
  per-row kernel-name sanity)
- Closure plan: `docs/ck_removal_closure_plan.md` (6 ordered work
  items, owner + ETA per item)
- Harness: `scripts/bench_ck_removal.py` (re-runnable, <15s on MI355X)
- Raw data: `results/ck_removal/bench_results.jsonl` (70 records)
- Kernel sanity: `results/ck_removal/kernel_sanity.json` (8 captures)
- Prior dashboard-staleness learning:
  `learnings/tuning/dashboard_staleness_detection.md` (sister
  always-remeasure rule)
