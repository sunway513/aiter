# depth-2 B prefetch: +3-4% at t=1024 with tile_k=128

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=1024, tile 96x128x128
- **Date**: 2026-04-03
- **Confidence**: verified

## Hypothesis
MFMA issue is stalling on B-tile GMEM latency in the inner K loop. Staging a second B tile in VGPR before the current MFMA completes should hide that latency, provided VGPR pressure stays within one-wave-per-SIMD budget.

## Result
At t=1024, tile_m=96, tile_k=128: 418us → **402us (-3.8%)**. VGPR usage rose from ~140 to ~160, occupancy unchanged (still 1w/SIMD).

## Root cause
Depth-2 keeps the B pipeline one iteration ahead of MFMA. The 20-VGPR overhead is tolerable at this tile size because tile_k=128 keeps the K-loop's A-tile buffer small. Switching to tile_k=256 would push VGPR to ~200 and trigger spill (see depth3 learning).

## Reusable rule
**depth-2 B prefetch is a Pareto win when tile_k ≤ 128 and VGPR headroom ≥ 20**. Verify VGPR budget before enabling: `FLYDSL_MOE_B_PREFETCH_DEPTH=2 python -c "..."` + ISA dump. Do not enable at tile_k=256 without spill check.

## References
- Env flag: `FLYDSL_MOE_B_PREFETCH_DEPTH` in `aiter/ops/flydsl/moe_kernels.py`
- Implementation commit: (TBD in ROCm/aiter PR)
- Bench: `scripts/definitive_bench.py`
