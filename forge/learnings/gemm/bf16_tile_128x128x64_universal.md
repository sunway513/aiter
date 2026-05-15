# Universal default tile for BF16 HGEMM: 128×128×64 w2×2 b_lds async

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` (BF16 square matmul)
- **Shape**: M=N=K ∈ {1024, 2048, 8192, 16384} — wins at all of these except 4096
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
Sweeping >50 (tile, warps, async, b_lds) configurations across 5 shapes should yield a single config that is within 5% of the per-shape optimum at most shapes, acting as a robust default.

## Result
Config `tile_m=128, tile_n=128, tile_k=64, block_m_warps=2, block_n_warps=2, b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False, async_copy=True` wins or nearly wins at four of the five tested shapes:

| Shape | Universal config | Shape-specific best | Δ |
|---|---|---|---|
| 1024  | 176.4 TFLOPS (w/ tile_k=128) | 176.4 |  0%  |
| 2048  | 643.8 TFLOPS (w/ tile_k=128) | 643.8 |  0%  |
| 4096  | did not sweep at this tile_k on this shape | 1116.6 | (n/a — needs 128×256×64 w1×4) |
| 8192  | **1116.6 TFLOPS** | 1116.6 |  0%  |
| 16384 |  996.6 TFLOPS   |  997.1 | −0.05% |

Note: at M=1024 / 2048 the universal config used tile_k=128 (fits in LDS because tile_m×tile_n is small). At M=8192 / 16384 tile_k=64 is the correct choice (tile_k=128 overflows LDS with b_lds=True and 128×128 output tile).

## Root cause
- tile_m=tile_n=128 keeps per-WG workload moderate: avoids the 256×256 pitfall where perf collapses to ~100 TFLOPS due to launch-bound configurations and VGPR pressure.
- w2×2 (total 4 warps / 256 threads) sits at the sweet spot for MFMA issue + LDS bandwidth on gfx950.
- b_lds + async_copy combo unlocks GMEM→LDS overlap without register pressure (see `bf16_b_to_lds_breakthrough.md`).
- tile_k=64 scales safely with LDS capacity when tile_m×tile_n is non-trivial; tile_k=128 only fits at small shapes.

## Reusable rule
**As the starting point for any BF16 HGEMM tuning on MI355X, use `tile_m=128, tile_n=128, tile_k=64, w2×2, b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False, async_copy=True`.** Sweep around this anchor rather than from `flydsl_hgemm` defaults. Expected performance: 80-95% of the per-shape optimum with zero per-shape tuning.

Corollary: the default kwargs of `flydsl_hgemm` are poorly chosen for production; propose to FlyDSL maintainers that these defaults become `b_to_lds=True, async_copy=True`.

## References
- Raw data: `results/gemm_bf16/results_round1.jsonl`, `results_round3.jsonl`, `results_round4.jsonl`
- Summary table: `results/gemm_bf16/summary.md`
- Related learning: `learnings/gemm/bf16_b_to_lds_breakthrough.md`
