---
name: doc-refresh
description: Refresh AITER operator documentation guides based on recent code changes. Detects which source files changed, maps them to affected guides, and updates the guides to match current code.
argument-hint: "[days-back or commit-range, e.g. '7' or 'abc123..HEAD']"
---

# AITER Documentation Refresh

Refresh operator documentation guides in `docs/` to reflect recent code changes.

## Step 1: Detect Changes

Determine the change scope from `$ARGUMENTS`:
- If a number (e.g. `7`), use `git log --since="$ARGUMENTS days ago" --name-only --pretty=format: | sort -u`
- If a commit range (e.g. `abc..HEAD`), use `git diff --name-only $ARGUMENTS`
- If empty, default to 7 days: `git log --since="7 days ago" --name-only --pretty=format: | sort -u`

## Step 2: Map Changed Files to Guides

Use this mapping to identify which guides need updating:

| Changed file pattern | Affected guide |
|---------------------|----------------|
| `aiter/ops/mha.py`, `csrc/cpp_itfs/mha_*`, `csrc/py_itfs_cu/asm_mha_*`, `aiter/ops/triton/attention/mha*`, `hsa/*/fmha_*` | `docs/attention_variants_guide.md` |
| `aiter/mla.py`, `aiter/ops/triton/attention/mla_*`, `hsa/*/mla/` | `docs/mla_kernel_support_report.md` |
| `aiter/fused_moe.py`, `aiter/ops/triton/moe/`, `aiter/ops/triton/_triton_kernels/moe/`, `hsa/*/fmoe/` | `docs/moe_variants_guide.md` |
| `aiter/ops/gemm_op_*`, `aiter/tuned_gemm.py`, `aiter/ops/triton/gemm/`, `csrc/ck_gemm_*`, `csrc/cktile_gemm_*` | `docs/gemm_variants_guide.md` |
| `aiter/ops/quant.py`, `aiter/ops/triton/quant/`, `aiter/utility/fp4_utils.py`, `aiter/int4_utils.py` | `docs/quantization_guide.md` |
| `aiter/ops/rmsnorm.py`, `aiter/ops/norm.py`, `aiter/ops/groupnorm.py`, `aiter/ops/triton/normalization/` | `docs/normalization_guide.md` |
| `aiter/ops/rope.py`, `aiter/ops/triton/rope/`, `aiter/rotary_embedding.py` | `docs/rope_guide.md` |
| `aiter/ops/cache.py`, `csrc/kernels/cache_kernels.cu`, `aiter/ops/triton/fusions/fused_kv_cache.py` | `docs/kv_cache_guide.md` |
| `aiter/ops/activation.py`, `aiter/ops/aiter_operator.py`, `aiter/ops/triton/activation.py`, `csrc/kernels/activation_kernels.cu` | `docs/elementwise_activation_guide.md` |
| `aiter/ops/sample.py`, `aiter/ops/sampling.py`, `csrc/cpp_itfs/sampling/` | `docs/sampling_guide.md` |

Also check if new operators or files were added that don't map to any existing guide.

## Step 3: For Each Affected Guide

For each guide that needs updating:

1. **Read the current guide** completely
2. **Read every changed source file** that maps to this guide
3. **Check for**:
   - New functions/APIs added but not documented
   - Removed functions still documented
   - Changed function signatures (parameters added/removed/renamed)
   - New backend support (ASM/CK/Triton) added
   - New test files added
   - Changed kernel counts or configurations
   - New config files or environment variables
4. **Update the guide** with corrections and additions
5. **Preserve the guide structure**: Quick Reference -> Numbered sections -> Decision Tree -> Source Files -> Test Files

## Step 4: Update README if Needed

If new operators were added or operator descriptions changed significantly, update the README.md Supported Operators table.

## Step 5: Summary

Print a summary table:

| Guide | Status | Changes Made |
|-------|--------|-------------|
| guide_name | Updated / No changes needed | Brief description |

## Critical Rules

### Negative claims require exhaustive proof
- **NEVER claim a feature doesn't exist without exhaustive search.** Search ALL directories: `csrc/py_itfs_cu/`, `csrc/cpp_itfs/`, `csrc/kernels/`, `hsa/`, `aiter/ops/`, `aiter/ops/triton/`
- **Use "?" if uncertain** about a feature rather than "-"
- Check BOTH `hsa/gfx942/` and `hsa/gfx950/` when documenting ASM support
- ASM backward attention files are in `csrc/py_itfs_cu/` (not `csrc/cpp_itfs/`)
- A feature may exist via an indirect API (e.g., HIP GQA support via two-channel `_2c_` RoPE APIs)

### Import paths — always verify, never assume
- **Read `aiter/__init__.py`** before writing `from aiter import X` — most functions are NOT re-exported
- Functions often live in submodules: `from aiter.mla import` not `from aiter import`
- Triton attention functions may be split across multiple files (e.g., `mha.py` vs `mha_v3.py`) — don't assume one module holds all variants

### Parameter names — always read the actual signature
- **NEVER guess parameter names** based on what sounds right. Always `grep 'def function_name'` and read the actual signature.
- Common traps: `fc1_scale` vs `w1_scale`, `residual` vs `res`, `top_k` vs `top_k_val`, `temperatures` vs `temperature`, `pad_to_multiple` vs `x_pad_to_multiple`
- Check **both Python stubs and C++ signatures** — Python stubs in `aiter/ops/` may show params as `Optional` when the C++ binding requires them

### Numbers — count, don't approximate
- **Count actual CSV rows** for kernel counts — subtract 1 for headers. Never write "22+" when the actual count is 21.
- **Read test files and config files** for block sizes, GQA ratios — don't guess from memory
- Use `ceil(K/128)` not `K//128` for block-scale shapes — off-by-one matters

### Recommend exported APIs, not internal implementations
- In Quick Reference tables, recommend the **highest-level exported API** (e.g., `rope_2d_fwd` from HIP, not `rope_fwd_2d` from Triton internals)
- If a Triton-internal function must be documented, show the full import path explicitly

### Terminology
- Use **"HIP"** not "CUDA" for AMD GPU code — this is an AMD project
- Distinguish **MXFP4 vs INT4** — test file named `a4w4` may use MXFP4, not INT4. Always check actual imports in the test.
- An enum existing (`per_128x128`) does NOT mean the feature is implemented — verify Python backend support

### Code examples must be runnable
- Every code example should work if copy-pasted. Verify: correct imports, correct param names, correct dtypes (e.g., `dtypes.fp4x2` not `torch.float8_e4m3fnuz` for FP4 quant)
- Check return types — fused functions may return tuples of 4 items, not 2
