# aiter-forge Supporting Ritual: Operator Dashboard Refresh

## Positioning

This ritual is **inside Priority 1 (AITER competitive performance health)** — it is no longer a standalone top-level mission. See `AGENTS.md` for the current mission hierarchy:

- **Priority 1**: AITER competitive performance on AMD MI-series (close gaps to NV B-series)
- **Priority 2**: CK replacement tracking + competitive performance benchmarking (`docs/ck_removal_tracker.md`, issue `#42`)
- **Supporting ritual (this doc)**: keep `ROCm/AI-Frameworks-Dashboard` op-perf tab numerically honest so Priority 1 and Priority 2 decisions rest on fresh measurements, not stale snapshots.

## Rationale

Dashboard snapshots lag landed AITER tuned-config PRs by days to weeks. A single `tuned_fmoe.csv` drop can invalidate hundreds of rows at once, pushing the dashboard's "worst offender" ranking completely out of sync with reality (we have measured up to **30× staleness** on MiniMax-M2.5 fused_moe down). Consuming stale rows as a tuning target wastes GPU time and misallocates attention.

aiter-forge does not touch AITER source. It:
1. Uses **current AITER main** plus its shipped tuned CSVs as-is.
2. Re-measures every dashboard row with the exact same (op, shape, quant, dtype) the dashboard claims to measure.
3. Emits a patch (JSON diff) for the dashboard maintainer.
4. Surfaces rows where the re-measured ratio is still below a threshold — those are the **genuine** tuning targets for downstream work (still without patching AITER — typically a tuned CSV update or a dispatch config).

## Cadence

- **On demand**: when someone points at a "low perf" dashboard row, refresh before any tuning commits.
- **Weekly**: automated run sweeping every model's op-perf rows, producing a patch PR to the dashboard repo.
- **Post-AITER-release**: trigger a full refresh within 24h of any `ROCm/aiter` tag or tuned-CSV merge.

## Scope

In-scope ops (refresh harness covers these today):
- `gemm_bf16` → `torch.matmul(a, b.T)` (hipBLASLt backend)
- `gemm_fp8` → `aiter.ops.gemm_op_a8w8.gemm_a8w8` **with proper tuned CSV path**
- `fused_moe` → `aiter.fused_moe.fused_moe` (BF16)
- `fused_moe_fp8` → same API with `quant_type=QuantType.a8w8blockscale` and proper scales

Ops that need dashboard schema extension before refresh (dashboard JSON drops shape params for these):
- `rmsnorm`
- `fused_activation` (silu_and_mul, gelu_and_mul)
- `mha_prefill` / `mha_decode`

For these we emit "schema-extension needed" entries in the patch; don't fabricate measurements.

## Runbook — how to refresh one model

### Step 1: Discover the rows
```bash
python3 scripts/refresh_dashboard.py --model="MiniMax-M2.5" --dry-run
```
Prints (category × op × shape) combos the dashboard has for this model and which our harness can run.

### Step 2: Re-measure on a live MI355X container
Inside `gemm-tune-1` (or any container with AITER main + MI355X):
```bash
docker cp scripts/refresh_dashboard.py gemm-tune-1:/tmp/
docker exec -e ROCR_VISIBLE_DEVICES=0 gemm-tune-1 \
  python3 /tmp/refresh_dashboard.py --model="MiniMax-M2.5" \
  --op-perf=/tmp/op-perf.json --output=/tmp/refresh_MiniMax.json
```
Typical runtime: 10-20 minutes for 400 rows.

### Step 3: Triage the refresh
```bash
python3 scripts/triage_dashboard_refresh.py results/refresh_<model>.json
```
Outputs three buckets:
- **stale** (refreshed ≥ 1.5× dashboard) — dashboard must update.
- **genuine-low** (refreshed ratio < 0.7×) — aiter-forge tuning target.
- **confirmed-ok** (refreshed ratio ≥ 0.8×) — no action.

### Step 4: Open the dashboard PR
```bash
scripts/open_dashboard_refresh_pr.sh <model>
```
Forks (if needed) / branches `ROCm/AI-Frameworks-Dashboard`, patches `docs/_data/op-perf.json`, opens a PR titled `data(op-perf): refresh <model> — N rows stale by ≥2×, median Yx`.

### Step 5: File genuine-low rows as aiter-forge tasks
For each remaining `ratio < 0.7×` row, create a `tasks/*.yaml` with hypothesis + candidates. These then go through the normal predict-verify flow (but still without patching AITER — only tuned CSVs or dispatcher hints).

## Non-goals

- **aiter-forge does NOT patch AITER source**. Any AITER code change is filed as a PR on `ROCm/aiter`, not this repo.
- **aiter-forge does NOT rewrite kernels in FlyDSL or custom code**. If a shape needs a new kernel, we file an issue on `ROCm/aiter` with a reproducer and measured gap.
- **aiter-forge does NOT publish refresh numbers without profile evidence**. Every dashboard patch must include a rocprofv3 counter snippet showing the winning kernel is a production CK / asm / FlyDSL-tuned path (not eager fallback).

## Authority & escalation

- aiter-forge has push access to `ROCm/AI-Frameworks-Dashboard` (verified 2026-04-22). PRs may be merged by aiter-forge maintainers after internal review.
- If refresh reveals a regression (ratio went DOWN vs dashboard), treat as a blocker: don't merge the patch, escalate to AITER team with profile evidence.
- If refresh fails (the harness can't reconstruct a shape), file an issue on the dashboard repo asking for the missing shape metadata.

## Prior art
- First refresh (MiniMax-M2.5, 2026-04-22): 350 of 412 rows refreshed. Fused MoE "down" was up to **29.6× stale**. Delivered as `PR #39` on sunway513/aiter-forge; dashboard-side PR pending.
- `learnings/tuning/dashboard_staleness_detection.md` — the reusable rule derived from this.
