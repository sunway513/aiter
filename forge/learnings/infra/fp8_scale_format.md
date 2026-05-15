# FP8 scale tensors must be flat, not 3D

- **Area**: infra
- **Kernel**: `aiter/ops/flydsl/test_flydsl_moe_fp8.py`, fused_moe calling convention
- **Shape**: FP8 per_Token MoE, any token count
- **Date**: 2026-04-02
- **Confidence**: verified

## Hypothesis
When setting up accuracy tests, we initially shaped scale tensors to match their logical layout: `a1_scale` as `[tokens, 1]`, `w1_scale` as `[E, 2, inter_dim]`. This mirrored how the weights are organized conceptually.

## Result
Kernel ran without error but produced NaN across all outputs. No compile-time or launch-time signal — just silent numerical failure at verification.

## Root cause
The kernel's scale-load code reads scales as flat contiguous arrays: `a1_scale[token_id]` (flat `[tokens]`) and `w1_scale[expert_id * 2 * inter_dim + i]` (flat `[E*2*inter_dim]`). With 3D inputs, stride computation silently mismatched the kernel's pointer arithmetic, producing garbage scales → FP8 dequant produced NaN.

There's no type/shape validation at the dispatch boundary — Python accepts any tensor, kernel trusts the contiguous byte layout.

## Reusable rule
**Always allocate scale tensors for fused MoE as 1D flat buffers, not logical-shape tensors:**
- `a1_scale`: `torch.empty([tokens], dtype=torch.float32)`
- `w1_scale`: `torch.empty([E * 2 * inter_dim], dtype=torch.float32)` (or the equivalent for w2)

If an accuracy test NaN'd, the first thing to check is scale-tensor shape, not kernel arithmetic.

**Infra TODO**: add a shape-assert at the Python dispatch layer that rejects non-flat scale tensors with a clear error, so silent NaN is never the first symptom again.

## References
- Test harness: `aiter/ops/flydsl/test_flydsl_moe_fp8.py`
- Debug session: 2026-04-02 (scale format hunt)
