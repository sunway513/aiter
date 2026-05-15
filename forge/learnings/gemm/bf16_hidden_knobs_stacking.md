# Hidden flydsl_hgemm knobs stack for +6-10% on top of b_to_lds

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` (BF16 square matmul)
- **Shape**: M=N=K ∈ {1024, 2048, 4096, 8192}
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
The `flydsl_hgemm` function exposes four kwargs that are **not** mentioned in `help(flydsl_hgemm)` but are present in the Python signature:
- `waves_per_eu` (0 = compiler default, 1-4 = explicit override)
- `persistent_n_tiles` (1 = one tile per WG; 2 = persistent multi-tile)
- `n_tile_repeat` (inner N-loop unroll factor)
- `b_to_lds_unroll` (B-load unroll depth)

We suspected these stack on top of the proven `b_to_lds=True` win to deliver an additional 5-10% at shapes where compute is near saturation.

## Result
Baselines (round 4 pre-stacking) vs round-6/7/8/9 stacked winners:

| Shape | Before (b_lds only) | After (b_lds + hidden knobs) | Δ |
|---|---|---|---|
| 1024  | 176.4  | **194.0**  | **+10.0%** |
| 2048  | 643.8  | **709.7**  | **+10.2%** |
| 4096  | 1116.6 | **1203.4** | **+7.8%**  |
| 8192  | 1116.6 | 1131.7     | +1.4%      |
| 16384 |  997.1 |  997.1     |  0%        |

Stacking pattern of the winning combos:
- M=1024: `n_tile_repeat=2, b_to_lds_unroll=4`
- M=2048: `waves_per_eu=2, persistent_n_tiles=2, b_to_lds_unroll=4`
- M=4096: `waves_per_eu=2, b_to_lds_unroll=4`
- M=8192: `n_tile_repeat=2, b_to_lds_unroll=4`

## Root cause
- `b_to_lds_unroll` unrolls the LDS→register load for the B tile, reducing loop overhead and exposing more ILP to the scheduler. It reliably adds 2-4% across all shapes when set to 4 or 8.
- `waves_per_eu` hints to the compiler the intended occupancy; `wpe=1` or `wpe=2` often let the compiler pick a cleaner register allocation. `wpe=0` (default) is too permissive and the compiler sometimes over-allocates.
- `persistent_n_tiles=2` keeps one WG resident across two N-tiles, eliminating WG-launch overhead. Only wins at M=2048 where launch overhead is significant; at larger M, the overhead is already amortized across the per-WG K-loop.
- `n_tile_repeat=2` and `persistent_n_tiles=2` are mutually exclusive in practice — both handle multi-tile per WG differently, and combining them often errors.

The reason gains collapse at M≥16384: the kernel is bandwidth-bound, not compute-bound. Loop-overhead optimizations help only when the inner MFMA issue rate is the bottleneck.

## Reusable rule
**When tuning `flydsl_hgemm`, always include the hidden-knob grid:**
```
waves_per_eu      ∈ {0, 1, 2}
persistent_n_tiles ∈ {1, 2}  (mutually exclusive with n_tile_repeat)
n_tile_repeat      ∈ {1, 2}  (mutually exclusive with persistent_n_tiles)
b_to_lds_unroll   ∈ {0, 2, 4, 8}
```

Expected wins: 5-10% at compute-bound shapes (M ≤ 4096), 1-2% at bandwidth-bound shapes (M ≥ 8192), 0% at very-bandwidth-bound (M ≥ 16384). Corollary: push the FlyDSL team to surface these kwargs in `help(flydsl_hgemm)` — they're currently discoverable only by reading source.

## References
- Raw data: `results/gemm_bf16/results_round6.jsonl`, `results_round7.jsonl`, `results_round8.jsonl`, `results_round9.jsonl`
- FlyDSL source: `/app/aiter-test/aiter/ops/flydsl/gemm_kernels.py` — `flydsl_hgemm` signature lines 837-862
