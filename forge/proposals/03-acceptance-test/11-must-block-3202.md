# [forge][Mission B][acceptance] Guardian MVP must BLOCK PR series #3202 / #3204 / #3205 / #3206

Refs RFC #TBD-RFC. This issue is the canonical negative-test acceptance criterion for the guardian gate (Mission B). The MVP is not declared ready until guardian produces `gate-blocked` on the live PR series below, with rationale that any AITER maintainer reading the comment would agree with on first read.

## The series

- **RFC** [#3202](https://github.com/ROCm/aiter/issues/3202) — `[RFC] MiniMax-M2.5 FP8 — Marathon Optimized (MI355X)` — opened 2026-05-14 by @peymanr
- **PR** [#3204](https://github.com/ROCm/aiter/pull/3204) — `[Perf] Cache moe_sorting buffers and scale/quant objects to reduce per-step allocation overhead` — DRAFT, +123/-1, 1 file
- **PR** [#3205](https://github.com/ROCm/aiter/pull/3205) — `[Perf][Kernel] Add gfx950 1-stage ASM FP8-blockscale fast path and lower BLOCK_SIZE_M to 16 for decode` — DRAFT, +270/-9, 1 file
- **PR** [#3206](https://github.com/ROCm/aiter/pull/3206) — `[Perf] Add MiniMax-M2.5 GEMM and FMoE tuning configs with doweight_stage1=0 dispatch key fix` — DRAFT, +308/-13, 2 files (134-row CSV)

## Why this is the right negative-test fixture

Real, live, ongoing series. The author is internal (AMD email `prazaghi@amd.com`). The intent is plausibly genuine. The technical claims may even be correct. But the **submission posture** has every smell that guardian was built to surface:

| # | Signal | Evidence |
|---|---|---|
| 1 | **PR churn** — same change submitted in multiple cosmetic re-splits | Two-week history: #2987 (CLOSED 4/30) → #2992 (CLOSED 5/01) → #3020/#3021/#3022 (all CLOSED 5/04) → #3197/#3199 (CLOSED 5/14, same day opened) → #3204/#3205/#3206 (current OPEN). All on `aiter/fused_moe.py` + `tuned_fmoe.csv`. |
| 2 | **No relevant CI labels** despite self-asserted +50% perf | All three current PRs show every heavy-CI workflow as `SKIPPED`: `Aiter Test`, `Atom Test`, `OPUS Test`, `Sglang Downstream Test`, `vLLM Benchmark`. Only `ruff`/`black`/`Repository Dependency` ran. |
| 3 | **Throughput claim with no artifact** | RFC body claims `+50.5% vs Sprint @ CONC=64` and `+23.4% vs B200 TP=2 @ CONC=64`. No `benchmark_serving.py` log file, no `report.json` link, no rocprof bundle attached. |
| 4 | **Kernel-selection claim with no rocprof verification** | RFC body asserts `1-stage ASM fast path is selected for ntok ≤ 512`. PR #3205 changes the gate but supplies no rocprof trace showing the kernel name actually appears for the asserted shape range. |
| 5 | **Tuning CSV without per-row provenance** | PR #3206 adds 106 GEMM rows + 28 FMoE rows. No mapping from each row to the sweep run that produced it. Reviewers cannot tell which rows came from a real measurement vs. a copy of an adjacent shape. |
| 6 | **Reviewer concern answered with markdown table, not reproducible bench** | Reviewer raised "PyTorch allocator already reuses." Author replied with a comment containing a 2-row table claiming `7.6 KB` / `473 KB` per-call alloc delta. No script link, no commit hash for the micro-bench, no way to re-run. |

## Required guardian behavior

When PRs #3204, #3205, #3206 are dispatched through the guardian pipeline (issues #TBD-06 / 07 / 08 / 09 / 10), the verdict comment MUST contain:

1. **Verdict label**: `gate-blocked` (not `needs-evidence` — the churn signal in (1) plus the missing CI in (2) escalates).
2. **Churn section**: lists the 6 historical CLOSED PRs as evidence of similarity, with diff Jaccard scores ≥ 0.7.
3. **Per-claim evidence table**: each of the 4 claims surfaced (throughput +50.5%, fast-path selection, alloc-overhead 7.6KB/473KB, dispatch-key fix) shown with its binding and FAIL/SKIP reason.
4. **Action paragraph**: tells the author what to do — push a commit that adds the missing artifacts, run the relevant CI labels, consolidate the series.

## Required guardian non-behavior

- Do NOT close the PRs.
- Do NOT name @peymanr in the comment beyond the standard "Author:" header.
- Do NOT make claims about author intent. Stick to artifacts and submission posture.

## Acceptance

This issue is closed when:

1. Guardian produces a comment matching the structure above on each of #3204, #3205, #3206.
2. An AITER maintainer (any) reviews the comment and confirms in this issue thread that they would have asked for the same evidence on first-pass review.
3. The fixture (anonymized snapshot of these PRs) is added to `forge/tests/guardian/fixtures/churn-and-no-artifact/`.

## Notes on fairness

The author of #3202 may turn out to be entirely correct on the merits. Guardian is not a quality verdict on the engineering — it is a verdict on the **submission posture**. A `gate-blocked` outcome means "we cannot review this efficiently as posted; please tighten the evidence." It does not mean "this work is bad." Maintainers retain full authority to override.

This fixture exists because the cost of a false positive (one author re-opening with stronger evidence) is much smaller than the cost of false negatives (maintainer review queue eaten by AI-assisted unverified PRs).
