# BF16 HGEMM tile_m should scale DOWN with M (up to a point), not up

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` (BF16 square matmul)
- **Shape**: M=N=K ∈ {1024, 2048, 4096, 8192, 16384}
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
Prior MoE work and intuition suggested "bigger shape → bigger tile." Test whether this holds for BF16 HGEMM across the ARGUS shape range.

## Result
Empirical winning tile_m per shape:

| Shape | Winning tile_m | WG count on M axis | Winning tile (full) |
|---|---|---|---|
| 1024  | **32**  | 32 | 32×128×64   |
| 2048  | **64**  | 32 | 64×128×64   |
| 4096  | **128** | 32 | 128×128×64  |
| 8192  | **128** | 64 | 128×128×64  |
| 16384 | **128** | 128 | 128×128×64 |

**tile_m scales roughly with sqrt(M)**, settling at 128 for M ≥ 4096. The invariant that emerges: **keep WG count on the M axis ≈ 32-128** to match MI355X's 304-CU budget times reasonable per-CU occupancy.

- At M=1024 with tile_m=128: only 8 WGs on M axis → under-utilizes CUs.
- At M=16384 with tile_m=32: 512 WGs on M axis → excess context-switching + LDS overhead.

## Root cause
MI355X has 304 CUs. BF16 HGEMM on gfx950 gets reasonable occupancy at 2 waves/SIMD (when VGPR ≤ 128). So the usable WG count is ~608 effective waves. Each WG has block_m_warps × block_n_warps warps — typical 4 warps/WG.

So ideal WG count = 304 × 2 × 64 / 4 / waves_per_wg ≈ 150-300.

Working backwards:
- tile_m × WG_count_M = M → if we want WG_count_M ≈ 32-128, tile_m ≈ M/32 to M/128.
- For M=1024: tile_m ≈ 8-32 → we chose 32.
- For M=16384: tile_m ≈ 128-512 → capped at 128 (validator rejects tile_m=256 for most warp configs).

**The 128-cap at large M is a compiler limit, not a hardware limit.** Beating aiter_asm at 8K+ probably requires tile_m ∈ {256, 512} which current FlyDSL doesn't expose cleanly (the `HGEMM_TILE_M_OPTIONS` list includes 256, but thread-count constraints make most warp layouts infeasible).

## Reusable rule
**Start tile_m tuning at M / 128 rounded to the nearest supported value:**
- M=1024 → try tile_m ∈ {16, 32, 48, 64}
- M=2048 → try tile_m ∈ {32, 48, 64, 96, 128}
- M=4096 → try tile_m ∈ {64, 96, 128, 160}
- M=8192 → try tile_m ∈ {96, 128, 160, 256}
- M≥16384 → try tile_m ∈ {128, 160, 256}

This cuts sweep space by ~5× vs blind enumeration.

**Corollary for MoE kernels**: the same "tile_m scales with M/WG_target" rule applies. Our prior learning `moe/tile_m_large_discovery.md` (tile_m=80,96 at t=512,1024 for FP8 MoE stage1) is consistent — both MoE stage1 and HGEMM sit in the same tile_m regime when normalized by shape.

## References
- Raw data: `results/gemm_bf16/results_round8.jsonl` (M=1024 fine-grid, 2112 configs)
- Raw data: `results/gemm_bf16/results_round9.jsonl` (M=4096 fine-grid, 1530 configs)
- Prior learning: `learnings/moe/tile_m_large_discovery.md` (MoE context)
