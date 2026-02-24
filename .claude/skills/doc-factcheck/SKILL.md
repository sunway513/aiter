---
name: doc-factcheck
description: Rigorous fact-checking of AITER documentation guides against the actual codebase. Verifies every API function, import path, parameter name, file path, backend support claim, and feature claim.
argument-hint: "[guide-name or 'all', e.g. 'attention' or 'all']"
---

# AITER Documentation Fact-Check

Perform a rigorous, production-grade fact-check of documentation guides against the actual codebase.

## Scope

Determine which guides to check from `$ARGUMENTS`:
- If `all` or empty: check all guides in `docs/*_guide.md` and `docs/*_report.md`
- If a specific name (e.g. `attention`): find the matching guide file

## For Each Guide: Verification Checklist

Launch parallel fact-check agents (one per guide, use `subagent_type=Explore`) that verify ALL of the following:

### 1. Import Paths & API Functions
- [ ] Every `from X import Y` statement — verify `Y` exists in module `X`
- [ ] Every function call — verify the function exists with that exact name
- [ ] Check if functions are exported from `aiter/__init__.py` before using `from aiter import`
- [ ] If not exported, use the full module path (e.g. `from aiter.mla import`)

### 2. Parameter Names & Signatures
- [ ] Every parameter name in code examples matches the actual function signature
- [ ] Default values shown match actual defaults
- [ ] Required parameters are not shown as optional (with `=None` or `=default`)
- [ ] Parameter order matches actual signature

### 3. Source File Paths
- [ ] Every file path in Source Files tables actually exists on disk
- [ ] No stale paths pointing to moved/deleted files

### 4. Test File Paths
- [ ] Every file path in Test Files tables actually exists on disk
- [ ] Test file descriptions match what the test actually tests

### 5. Backend Support Tables
- [ ] For each "Yes" cell: find actual implementation code proving it exists
- [ ] For each "-" cell: do an **exhaustive search** across ALL directories to confirm it truly doesn't exist
  - Search: `csrc/py_itfs_cu/`, `csrc/cpp_itfs/`, `csrc/kernels/`, `hsa/gfx942/`, `hsa/gfx950/`, `aiter/ops/`, `aiter/ops/triton/`
- [ ] If uncertain, mark as "?" rather than "-"

### 6. Feature Claims
- [ ] Kernel counts: count actual files/CSV rows, don't guess
- [ ] Data type support: verify against actual kernel code or dispatch logic
- [ ] Hardware support: check both GFX942 and GFX950 directories
- [ ] Block sizes, GQA ratios: verify from config files or test parameters

### 7. Code Examples
- [ ] Code examples would actually run without errors
- [ ] Return types/shapes match what the function actually returns
- [ ] Variable names are consistent within each example

### 8. Terminology
- [ ] Use "HIP" not "CUDA" for AMD GPU code
- [ ] Use correct quantization type names (MXFP4 not INT4 where applicable)

## Output Format

For each guide, produce a report:

```
## Guide: <filename>

### ISSUES FOUND

#### [CRITICAL/HIGH/MODERATE/MINOR] Issue Title
- **Location**: Line N
- **Guide says**: <what the guide claims>
- **Code says**: <what the code actually has>
- **Fix**: <specific correction>

### VERIFIED CORRECT
- Brief list of major claims verified as accurate

### SUMMARY
| Severity | Count |
|----------|-------|
| Critical | N |
| High | N |
| Moderate | N |
| Minor | N |
```

## After Fact-Check

1. Present the consolidated report to the user
2. Ask if they want all issues fixed
3. If yes, apply all fixes, amend the commit, and force push

## Critical Rules

### Negative claims
- **"I didn't find it" does NOT mean "it doesn't exist"** — search exhaustively before marking any feature as unsupported
- ASM backward attention files are in `csrc/py_itfs_cu/` (not `csrc/cpp_itfs/`)
- ASM kernel binaries are in `hsa/gfx{arch}/` subdirectories
- Always check BOTH `gfx942` and `gfx950` directories
- A feature may exist via an indirect API (e.g., HIP GQA support via `_2c_` two-channel RoPE APIs)

### Common mistake patterns to check for
These are the most frequent error categories found in past reviews — prioritize checking these:

1. **Wrong import paths**: Functions assumed to be re-exported at `aiter` level but actually require `aiter.mla`, `aiter.ops.triton.attention.mha_v3`, etc. Always verify against `aiter/__init__.py`.
2. **Guessed parameter names**: `fc1_scale` vs `w1_scale`, `residual` vs `res`, `top_k` vs `top_k_val`, `temperatures` vs `temperature`. Always `grep 'def func_name'` and read the actual signature.
3. **Required params shown as optional**: Python stubs may not show C++ binding constraints. Check `csrc/include/*.h` for the real signature — params like `k_scale`, `v_scale`, `scale` may be required even when they look optional.
4. **Approximate numbers**: Kernel counts like "22+" when actual count is 21. Block sizes "8,16,128,256" when actual is "16,1024". Always count CSV rows (minus header).
5. **Confused formats**: INT4 vs MXFP4 (test file `a4w4` may be MXFP4), `torch.float8_e4m3fnuz` vs `dtypes.fp4x2` for FP4 quant. Always check the `assert` in the function body.
6. **Wrong terminology**: "CUDA" instead of "HIP" for AMD GPU code. "CUDA/CK" when only HIP exists.
7. **Recommending internal APIs**: Quick Reference should recommend highest-level exported API (e.g., `rope_2d_fwd` from HIP), not Triton internals.
8. **Extrapolated signatures**: Guessing a function's API from a similar variant (e.g., assuming 3D RoPE takes cos/sin like 2D). Always read the actual function.
