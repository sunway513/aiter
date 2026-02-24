---
name: doc-new-guide
description: Generate a new AITER operator documentation guide from scratch by analyzing the source code, tests, and kernel configurations.
argument-hint: "<operator-name, e.g. 'flash_attention_v4' or 'new_quantization_op'>"
---

# Generate New AITER Operator Guide

Create a comprehensive documentation guide for a new or undocumented AITER operator.

## Step 1: Discover the Operator

From `$ARGUMENTS`, search the codebase to find all related files:

1. Search `aiter/ops/` for Python API files
2. Search `aiter/ops/triton/` for Triton implementations
3. Search `csrc/` for C++/HIP/CK implementations
4. Search `hsa/` for ASM kernel binaries
5. Search `op_tests/` for test files
6. Search `aiter/configs/` for tuning configurations
7. Search `aiter/__init__.py` for exported functions

## Step 2: Analyze the Implementation

For each discovered file:

1. **Read the full source code**
2. **Extract**:
   - All public function names and their exact signatures (parameter names, types, defaults)
   - Supported data types (check dispatch macros, type checks)
   - Backend support (ASM, CK, CKTile, Triton, hipBLASLt)
   - Hardware support (check for GFX942/GFX950-specific code paths)
   - Kernel configurations (block sizes, tile sizes, tuning parameters)
   - Environment variables
   - Dependencies on other AITER operators

## Step 3: Write the Guide

Follow this exact structure (matching existing guides):

```markdown
# AITER <Operator Name> Guide

<1-2 sentence description of what this operator does and when to use it.>

---

## Quick Reference

| Use Case | Recommended Operation | Backend | Why |
|----------|---------------------|---------|-----|
| ... | ... | ... | ... |

---

## 1. <First Variant or Concept>

### Backend Support

| Feature | CK (C++) | Triton | ASM |
|---------|:---:|:---:|:---:|
| ... | ... | ... | ... |

### Key API Functions

```python
from <exact.module.path> import <function>
# Example usage with correct parameter names
```

### When to Use
- ...

---

## N. Choosing Between Variants (Decision Tree)

```
<ASCII decision tree>
```

---

## N+1. Source Files

| Component | Path |
|---|---|
| ... | ... |

---

## N+2. Test Files

| Test | Path |
|------|------|
| ... | ... |
```

## Step 4: Verify Before Saving

Before writing the guide file:

1. **Verify every import path** — test that the module and function exist
2. **Verify every parameter name** — match against actual function signatures
3. **Verify every file path** — confirm existence on disk
4. **Verify backend support** — search exhaustively; use "?" if uncertain, never false "-"
5. **Check both GFX942 and GFX950** for ASM kernel support

## Step 5: Update README

Add a row to the `## Supported Operators` table in `README.md`:

```markdown
| <Operator Name> | <brief description> | [<Name> Guide](docs/<filename>.md) |
```

If the operator is infrastructure/tooling rather than a compute operator, add it to `## Infrastructure & Tuning` instead.

## Step 6: Summary

Print what was created and where, and suggest the user run `/doc-factcheck <guide-name>` to verify.

## Critical Rules

### Feature discovery
- **NEVER claim a feature doesn't exist without exhaustive search** across ALL directories
- ASM backward files may be in `csrc/py_itfs_cu/` (not just `csrc/cpp_itfs/`)
- ASM kernel binaries are in `hsa/gfx{arch}/` subdirectories — check both `gfx942` and `gfx950`
- Features may exist via indirect APIs (e.g., GQA via two-channel `_2c_` variants)

### Import paths
- **Read `aiter/__init__.py`** to check what is re-exported before writing `from aiter import X`
- Most functions require full path: `from aiter.mla import`, `from aiter.ops.triton.attention.mha_v3 import`
- In Quick Reference, recommend the **highest-level exported API**, not Triton internals

### Parameter names & signatures
- **Always `grep 'def function_name'` and read the actual signature** — never guess names
- Past mistakes: `fc1_scale`→`w1_scale`, `residual`→`res`, `top_k`→`top_k_val`, `pad_to_multiple`→`x_pad_to_multiple`
- Check C++ bindings (`csrc/include/*.h`) for required vs optional — Python stubs may not show the C++ constraints
- Don't extrapolate signatures from similar variants (e.g., 3D RoPE API is completely different from 2D)

### Numbers & counts
- **Count actual CSV rows** (minus header) for kernel counts — never approximate with "~" or "+"
- Read test files for actual block sizes and GQA ratios — don't guess
- Use `ceil(K/N)` not `K//N` for block-scale shapes

### Terminology & types
- Use **"HIP"** not "CUDA" — this is an AMD project
- Distinguish **MXFP4 vs INT4** — a test named `a4w4` may use MXFP4. Check actual imports.
- Use `dtypes.fp4x2` not `torch.float8_e4m3fnuz` for FP4 quantization
- An enum existing does NOT mean the feature is implemented — verify Python backend support

### Code examples
- Every example must be copy-paste runnable with correct imports, param names, and dtypes
- Check return types — fused ops may return tuples of 4 items, not 2

### Conventions
- Guide filename: `docs/<operator_name>_guide.md`
