# scf.for K-loop: +18% regression (phi blocks cross-iter reg opt)

- **Area**: moe
- **Kernel**: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`
- **Shape**: FP8 stage1, t=1024, tile 96x128x128
- **Date**: 2026-04-04
- **Confidence**: verified

## Hypothesis
The K-loop currently uses `range_constexpr` which fully unrolls 15 iterations, producing a huge IR (~15x the looped form). Hypothesis: converting to `scf.for` with loop-carried values would shrink IR, let the compiler allocate registers more globally, and match or beat the unrolled version.

## Result
391us (unrolled) → **462us (scf.for, +18%)**. Accuracy preserved.

## Root cause
The `scf.for` version introduces 58 loop-carried values at the loop boundary. Each becomes an LLVM phi node. LLVM's register allocator cannot propagate live-range information across phi nodes as freely as across a fully-unrolled linear region — so accumulator VGPRs that could stay in register across iterations in the unrolled version get spilled or reassigned unnecessarily under phi.

Full unroll lets the compiler see every MFMA operand's lifetime explicitly, enabling global coloring.

## Reusable rule
**Prefer `range_constexpr` full unroll over `scf.for` when the loop body has many loop-carried accumulators (> ~20).** The IR-size win is real but the register allocation loss is larger. Only switch to `scf.for` if you're genuinely compile-time bound (minutes) or the loop has few carried values.

Corollary: "smaller IR = better codegen" is false for MFMA-heavy loops. The compiler's global view across unrolled iterations is the actual asset.

## References
- Parallel sub-agent experiment, 2026-04-04
- MLIR diff captured in session notes
- Summary: `docs/optimization_summary.md` §"scf.for K-loop (DONE — REGRESSED)"
