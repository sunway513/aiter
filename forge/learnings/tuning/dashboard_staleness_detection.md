# Always re-measure dashboard "low-perf" rows on fresh AITER before tuning

- **Area**: tuning
- **Kernel**: N/A (applies to any external perf-ratio dashboard)
- **Shape**: N/A
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
`ROCm/AI-Frameworks-Dashboard` (`cuddly-adventure-1qwyj9p.pages.github.io`) flags specific kernels as "low performance" with ratios like MI355X/B300 = 0.02 - 0.2. An agent could pick the worst-ratio rows and start tuning them. We suspected a fraction of those rows are stale — fixed upstream but not yet reflected in the dashboard snapshot — which would waste any tuning effort.

## Result
Sampling the 8 worst-ratio rows in "Fused MoE (AITER vs best NV)" category on the current node (MI355X, AITER `origin/main @ 6890159`, 2026-04-22):

| Row | Dashboard MI355X | **Re-measured MI355X** | B300 (dashboard) | Staleness | New ratio |
|---|---|---|---|---|---|
| MiniMax-M2.5 down M=128 | 0.35 | **20.15** | 19.56 | **57×** stale | **+3%** (beats B300) |
| MiniMax-M2.5 down M=256 | 0.71 | 26.00 | 38.96 | 37× stale | -33% |
| MiniMax-M2.5 down M=512 | 1.41 | 51.57 | 75.19 | 37× stale | -31% |
| MiniMax-M2.5 down M=64 | 0.21 | 11.86 | 10.48 | 57× stale | **+13%** (beats B300) |
| MiniMax-M2.5 down M=32 | 0.15 | 7.81 | 6.85 | 52× stale | **+14%** |
| MiniMax-M2.5 down M=1024 | 2.78 | 98.32 | 116.03 | 35× stale | -15% |
| MiniMax-M2.5 down M=16 | 0.13 | 5.44 | 4.94 | 42× stale | **+10%** |
| MiniMax-M2.5 down M=2048 | 5.51 | 180.11 | 208.23 | 33× stale | -13% |

Then spot-checking GEMM worst rows:

| Row | Dashboard MI355X | Re-measured (torch.matmul) | Staleness |
|---|---|---|---|
| qkv_proj M=4096×6144×3584 | 316.0 | **1150.4** | 3.6× stale |
| o_proj M=256×6144×16384   | (ratio 0.24) | **642.2** | ~3× stale (estimate) |

Dashboard `lastUpdated: 2026-04-08`. AITER main had 14+ tuned-config PRs land between 2026-04-08 and 2026-04-22 (including `a8w8_blockscale_tuned_fmoe_minimax-m2_5.csv`), any of which invalidated MiniMax rows.

Profile evidence (rocprofv3 on M=256 MiniMax MoE): AITER uses `ck::kernel_moe_gemm_2lds` + `ck_tile::MoeSortingMultiPhaseKernel` — the high-performance CK 2-stage MoE path, **not** eager fallback. Kernel dispatch is correct; the dashboard snapshot just pre-dates the tuned config.

## Root cause
Dashboards publish batched snapshots (weekly-ish cadence) but the underlying codebase ships tuned configs daily. Any perf-ratio panel aged > 3 days needs re-measurement before becoming a tuning target, especially for categories where a single CSV drop can 10-50× perf (MoE tuned_fmoe CSVs are the canonical example).

## Reusable rule
**Before aiter-forge picks a "low-perf" kernel from any external dashboard, ALWAYS re-measure the worst 3-5 rows on fresh AITER main.** Specifically:

1. Check `dashboard.lastUpdated` — if > 3 days old, assume some rows are stale.
2. Pull the worst-ratio rows (e.g. MI355X/B300 < 0.3).
3. Re-run each on current AITER (latest `origin/main`) with the same (op, shape, quant_type, dtype) parameters.
4. **Only tune the rows where the re-measured ratio is still < 0.8×**. Anything above that is already competitive — the dashboard just hasn't caught up.
5. For stale rows, file an issue on the dashboard repo with the fresh measurements so the snapshot refreshes.

Corollary: extend `aiter_forge.predict_verify` task YAML schema with a `current_ratio_check` field that requires the measured-today ratio before the task body runs. No fresh measurement → no tuning budget.

## Implementation status in aiter-forge
- `scripts/bench_moe_minimax_down.py` — single-shape reproduction harness for MiniMax MoE.
- `scripts/bench_dashboard_low_perf_sample.py` — multi-shape spot-check that picks the 8 worst rows per category and reruns them, printing the `delta = our/dash` ratio.
- `scripts/verify_dashboard_staleness.py` — one-worst-row-per-category summary for fast triage.

All three scripts commit to PR alongside this learning so the next agent can re-run them before targeting any dashboard row.

## References
- Dashboard: https://cuddly-adventure-1qwyj9p.pages.github.io/#op-perf (repo: `ROCm/AI-Frameworks-Dashboard`)
- Raw data: `data/op-perf.json` snapshot (2026-04-08)
- Profile CSVs: `/tmp/prof_moe_m256_counter_collection.csv` (23 CK GEMM dispatches + 26 MoE sorting per call at M=256)
- Prior learning: `learnings/tuning/gpu_sharded_sweep.md` (companion "always use all GPUs" rule)
