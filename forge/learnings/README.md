# learnings/

Structured record of kernel-tuning experiments. **Every experiment MUST append one entry here before completion** (tracked in issues #27 and #28).

## Purpose

- Prevent agents from repeating the same regression (scf.for, zero-LDS, depth-3 prefetch all regressed in this project).
- Capture *why* a config won or lost, not just the numbers — so the next experiment on a different shape can inherit the reasoning.
- Replace bulk sweeping with predict-verify: learnings compress past sweeps into rules.

## Entry format

```markdown
# <short title>

- **Area**: moe / gemm / rope / attn / infra / ...
- **Kernel**: path/to/kernel.py (if applicable)
- **Shape**: dtype, token range, dims
- **Date**: YYYY-MM-DD
- **Confidence**: verified / probable / hypothesis

## Hypothesis
What we expected to happen, with reasoning.

## Result
What actually happened. Absolute numbers + delta vs baseline.

## Root cause
Why the result differs from (or matches) the hypothesis. Grounded in
arch details: VGPR / LDS / occupancy / MFMA granularity / scheduling.

## Reusable rule
One-sentence generalization an agent can apply to new shapes.

## References
- commit SHA / PR / file paths / benchmark JSON
```

## Index (by area)

### MoE
- [tile_m_large_discovery](moe/tile_m_large_discovery.md) — tile_m=80/96 beats 64 at t=512/1024 by 13-14%
- [coderfeli_small_tile_feedback](moe/coderfeli_small_tile_feedback.md) — FlyDSL maintainer: 32x128 / 32x256 / 64x128 preferred at t≥512; gap is tile choice not compiler
- [depth2_prefetch_win](moe/depth2_prefetch_win.md) — depth-2 B prefetch: +3-4% at t=1024, tile_k=128
- [depth3_prefetch_regression](moe/depth3_prefetch_regression.md) — depth-3 B prefetch: +18% regression (VGPR spill)
- [scf_for_regression](moe/scf_for_regression.md) — scf.for K-loop: +18% regression (phi blocks cross-iter reg opt)
- [zero_lds_regression](moe/zero_lds_regression.md) — naive LDS removal: +181% regression (M-reuse lost)
- [8warp_occupancy_ceiling](moe/8warp_occupancy_ceiling.md) — 8-warp alone slower than 4-warp (compiler emits 192 VGPR, no occupancy gain)

### GEMM
- [bf16_b_to_lds_breakthrough](gemm/bf16_b_to_lds_breakthrough.md) — `b_to_lds=True` dominates BF16 HGEMM tuning, +11-75% across shapes on MI355X (ARGUS paper replica)
- [bf16_tile_128x128x64_universal](gemm/bf16_tile_128x128x64_universal.md) — universal anchor tile for BF16 HGEMM: 128×128×64 w2×2 b_lds async
- [bf16_hidden_knobs_stacking](gemm/bf16_hidden_knobs_stacking.md) — undocumented flydsl_hgemm kwargs (waves_per_eu, persistent_n_tiles, n_tile_repeat, b_to_lds_unroll) stack for +6-10% on top of b_to_lds
- [bf16_tile_scaling_vs_shape](gemm/bf16_tile_scaling_vs_shape.md) — tile_m should scale with sqrt(M) up to 128; ideal WG_count_M ≈ 32-128 for MI355X
- [bf16_small_m_kernel_family](gemm/bf16_small_m_kernel_family.md) — `kernel_family='small_m'` beats hipBLASLt at M=1024 (206.4 vs 202.2 TFLOPS), wins for M≤1024
- [bf16_profile_guided_compiler_ceiling](gemm/bf16_profile_guided_compiler_ceiling.md) — rocprofv3 evidence: FlyDSL gap at M≥4096 is compiler register-allocation policy (96 vs 256 VGPR), not tile choice — tile-size sweeps exhausted

### Tuning (cross-domain)
- [gpu_sharded_sweep](tuning/gpu_sharded_sweep.md) — never leave GPUs idle during a sweep; shard along an orthogonal axis (tile_m for GEMM) when shape-parallelism is narrower than GPU count
- [dashboard_staleness_detection](tuning/dashboard_staleness_detection.md) — always re-measure worst-ratio dashboard rows on fresh AITER before picking a tuning target; snapshots >3 days old often have 30× stale rows
- [ck_removal_methodology](tuning/ck_removal_methodology.md) — per-op × per-shape pair-bench with kernel-name sanity before trusting any "very close" backend-swap claim; "CK removal is drop-in" is false for BF16 MoE and RMSNorm large-M

### Infra
- [fp8_scale_format](infra/fp8_scale_format.md) — scale tensors must be flat [tokens] / [E*2*inter_dim], not 3D; 3D layout silently NaNs
