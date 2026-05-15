# small_m kernel family beats hipBLASLt at M=1024

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` with `kernel_family='small_m'`
- **Shape**: M=N=K=1024 (small_m wins for M ≤ 1024; loses for M ≥ 2048)
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
The standard `kernel_family='hgemm'` dispatch path in FlyDSL has an explicit
`small_m` sibling (`aiter.ops.flydsl.kernels.small_m_hgemm`). It fixes `tile_m=16` and `block_m_warps=1`, designed for small-M GEMMs. Our hgemm sweep topped at 194.0 TFLOPS at M=1024, still 4% behind hipBLASLt (202.2). We suspected the small_m family would unlock the remaining gap via higher WG count on the M axis.

## Result
| Variant | Best at M=1024 | vs hipBLASLt (202.2) |
|---|---|---|
| `flydsl_hgemm` default           | 100.6  | −50% |
| `flydsl_hgemm` b_to_lds tuned    | 194.0  | −4.0% |
| aiter_asm (`gemm_a16w16_asm`)    | 100.7  | −50% |
| **`kernel_family='small_m'` tuned** | **206.4** | **+2.1%** ✅ |

Winning config:
```python
flydsl_hgemm(a, b,
    kernel_family='small_m',
    tile_m=16, tile_n=64, tile_k=64,
    block_m_warps=1, block_n_warps=2,
    b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False,
    waves_per_eu=3, b_to_lds_unroll=4,
)
```

At M=2048, the same small_m family tops out at 309 TFLOPS (vs hgemm's 709.7). Crossover shape is ≈1024-2048.

## Root cause
At M=1024, MI355X's 304 CUs want as many workgroups as possible. With hgemm `tile_m=32/64/128` we get 8-32 WGs on the M axis; with `tile_n=128` that's 64-256 total WGs — some CUs idle.

`small_m` forces `tile_m=16`, producing **64 WGs on the M axis** × N_tile repartition = higher total WG count, better CU occupancy. The fixed tile_m=16 also matches the native MFMA 16×16×16 instruction dimensions exactly, eliminating tile decomposition.

At M=2048+, the extra WG count no longer compensates for the tile-setup overhead of 4× more WGs than hgemm would emit. hgemm's tile_m=64/128 amortizes MFMA issue better.

## Reusable rule
**For M ≤ 1024 on MI355X (gfx950), always include `kernel_family='small_m'` in the tuning sweep.**

Minimal code for small_m best config at M=1024:
```python
out = flydsl_hgemm(a, b,
    kernel_family='small_m',
    tile_m=16, tile_n=64, tile_k=64,
    block_m_warps=1, block_n_warps=2,
    b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False,
    waves_per_eu=3, b_to_lds_unroll=4,
)
```

**FlyDSL dispatcher feedback**: the auto-dispatch in `flydsl_hgemm` should prefer `kernel_family='small_m'` when M ≤ 1024. Currently requires manual override — users without this learning will not discover it.

## References
- Raw data: `results/gemm_bf16/results_round11.jsonl` (5120 configs, 544 valid, best 206.4)
- Bench script: `scripts/bench_gemm_bf16_round11_small_m.py`
- FlyDSL source: `aiter/ops/flydsl/kernels/small_m_hgemm.py` (defines the kernel family)
- Related: `learnings/gemm/bf16_b_to_lds_breakthrough.md`, `learnings/gemm/bf16_hidden_knobs_stacking.md`
