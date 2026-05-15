# tile_m=80/96 beats 64 at t=512/1024

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py` (FP8 per_Token)
- **Shape**: DSR1 FP8 stage1, t=512 and t=1024, inter_dim=512
- **Date**: 2026-04-03
- **Confidence**: verified

## Hypothesis
Prior sweeps capped tile_m at 64 on the assumption that power-of-two tiles were optimal for MFMA 16x16x32 alignment. We suspected larger non-pow2 tiles might under-utilize MFMA, so they were excluded.

## Result
| Shape | tile_m=64 (prev best) | tile_m=80/96 (new) | Delta |
|---|---|---|---|
| t=512, 64x256x128 | 335us | **80x128x128: 292us** | -13% |
| t=1024, 64x256x128 | 460us | **96x128x128: 402us** | -14% |

## Root cause
MFMA 16x16x32 dictates tile_m must be a multiple of 16, not pow-of-2. tile_m=80 = 5×16, tile_m=96 = 6×16 — both valid MFMA-aligned. Larger tile_m amortizes A-tile LDS load across more MFMA issues, hiding GMEM latency better at mid/large tokens. With tile_k=128 (not 256), LDS pressure drops, so the larger M-dimension fits.

The pow-2 constraint was cargo-cult from CUTLASS patterns, not from the hardware.

## Reusable rule
**tile_m must be a multiple of MFMA_M (16 for gfx950), not power-of-two.** When sweeping tiles, include 48, 80, 96, 112 as first-class candidates. Verify LDS budget first: larger tile_m + tile_k=128 usually fits; larger tile_m + tile_k=256 often spills.

## References
- PR discussion with coderfeli on ROCm/FlyDSL#348
- Bench script: `scripts/sweep_mxfp4_moe.py`
- Optimization summary: `docs/optimization_summary.md`
