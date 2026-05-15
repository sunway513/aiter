# Gap to ASM is tile choice, not compiler

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=512 / t=1024, MI355X
- **Date**: 2026-04-05
- **Confidence**: verified (maintainer claim, not yet reproduced on our side)

## Hypothesis
Our FlyDSL best (80x128x128 at t=512: 292us) still trails ASM+silu (238us). We suspected this was a FlyDSL compiler limitation — less aggressive pipelining or worse register allocation than hand-ASM.

## Result
**@coderfeli (FlyDSL maintainer) on ROCm/FlyDSL#348:** the perf gap is dominated by *tile size choice*, not impl. ASM uses 144x128; FlyDSL at 80x128 — but both are suboptimal for t=512. In his local tests **32x128, 32x256, and 64x128** beat both and FlyDSL matches ASM at those tiles.

## Root cause
Small tiles increase workgroup count, which at t=512 scales better across CUs (304 on MI355X) than fewer large tiles. Our sweep space was biased toward "bigger tiles = bigger throughput," missing that WG count × per-WG efficiency is the real objective.

The compiler wasn't the bottleneck. Our search space was.

## Reusable rule
**Include 32x128, 32x256, 64x128 as mandatory candidates in every MoE tile sweep at t≥512.** Don't assume big tiles are always better — CU count and WG occupancy matter more than per-tile throughput. For MI355X (304 CUs), shapes that give 200-600 WGs tend to win over shapes with 50-100 WGs.

## References
- ROCm/FlyDSL#348 — coderfeli comment (2026-04-05)
- Ported insight to `docs/optimization_summary.md` update
- Next sweep target in `docs/overnight_plan.md` Phase 2
