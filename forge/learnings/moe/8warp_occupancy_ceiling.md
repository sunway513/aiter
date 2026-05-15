# 8-warp alone doesn't help compute-bound MoE (VGPR=192, 1w/SIMD)

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=1024, tile 96x128x128
- **Date**: 2026-04-04
- **Confidence**: verified

## Hypothesis
Triton achieves 8 waves/WG × 2w/SIMD occupancy and is faster at t≥512. Hypothesis: switching our kernel from 4-warp (256 threads) to 8-warp (512 threads) would double in-flight waves, hide MFMA latency better, and close the ASM gap.

## Result
Fixed the `hipErrorLaunchFailure` via `@flyc.kernel(known_block_size=[512,1,1])` decorator + `rocdl.flat_work_group_size` attribute. **8-warp build SLOWER than 4-warp.** Also: NaN bugs at t=512 with 8-way N-dim split (wave mapping misassigns N tiles to waves).

## Root cause
With 8 warps, per-WG workload is unchanged but each wave now has half the MFMA work → compiler generates per-wave VGPR ~192 (more accumulators per wave for the wider tile spread). At 192 VGPR, occupancy is still 1w/SIMD (ceiling is ~200 for 1w, ~100 for 2w). So 8-warp gets no occupancy benefit and pays overhead of higher thread count per WG (LDS bank contention, more barriers).

Triton's 8-warp works because each wave does a *narrower* per-wave tile, keeping per-wave VGPR under ~100 → actually achieves 2w/SIMD. Our kernel keeps per-wave work the same → no occupancy gain.

## Reusable rule
**Switching warp count in isolation is almost never a win.** To benefit from 8-warp, the per-wave tile geometry must shrink enough to bring per-wave VGPR under ~100 (for 2w/SIMD on gfx950). That's a kernel redesign, not a parameter flip.

When testing higher warp count, always co-modify: (a) per-wave N-dim split, (b) accumulator tile size, (c) LDS A/B layout to match the new wave partition. Otherwise VGPR stays at the 1w/SIMD ceiling.

## References
- Parallel sub-agent experiment, 2026-04-03
- `known_block_size` fix: `moe_gemm_2stage.py` `@flyc.kernel` decorator
- Summary: `docs/optimization_summary.md` §"8-warp (512 threads)"
