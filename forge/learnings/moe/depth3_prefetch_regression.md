# depth-3 B prefetch: +18% regression (VGPR spill)

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=1024, tile 96x128x128
- **Date**: 2026-04-04
- **Confidence**: verified

## Hypothesis
If depth-2 prefetch closed ~4% of the gap to ASM (which uses a hand-tuned 3-deep pipeline), depth-3 should close more. Plan: stage 3 B-tiles in VGPR, fully hiding L2 latency.

## Result
402us (depth-2) → **461us (depth-3, +14.7%)**. Occupancy dropped from 1w/SIMD.

## Root cause
Third B-tile buffer cost ~32 additional VGPRs on top of depth-2's ~160. Total crossed the ~200-VGPR spill threshold for this architecture, forcing register allocator to spill to scratch. Spill traffic on every MFMA swamped the latency-hiding gain.

ASM's depth-3 works because it uses hand-crafted register allocation with tighter reuse patterns — not a simple "add one more buffer" extension the compiler can do.

## Reusable rule
**Don't extend prefetch depth without verifying post-change VGPR.** On gfx950, the 1w/SIMD ceiling is ~200 VGPR; crossing it causes spill that dwarfs prefetch gains. If an extra pipeline stage needs > 20 more VGPR than current, expect regression.

**Rule for FlyDSL specifically**: 2-deep is the practical ceiling without redesigning the inner loop's register-reuse pattern. If you want 3-deep, you also need to reduce per-stage VGPR usage (e.g., narrower B tile, aggressive reuse of VGPR slots).

## References
- Parallel-agent experiment log, 2026-04-04
- Summary: `docs/optimization_summary.md` §"3-deep B prefetch (DONE — REGRESSED)"
