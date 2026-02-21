# AITER Kernel Coverage Analysis: CK-Free Build Gap Assessment

This document maps every kernel backend in AITER (ASM, CK, Triton, FlyDSL) to identify
gaps when building with `ENABLE_CK=0`. The goal is to determine what FlyDSL needs to cover
so AITER can ship a fast, CK-free build without losing operator coverage.

**Principle**: If an operator is already covered by ASM or Triton, it does not need a FlyDSL
replacement. We only care about operators that *exclusively* depend on CK.

## 1. Backend Overview

| Backend | Description | Build Cost | Performance |
|---------|-------------|-----------|-------------|
| **ASM** | Hand-tuned assembly `.co` files | Zero (prebuilt) | Highest |
| **CK** | Composable Kernel (HIP C++) | Very high (2782+ HIP files) | High |
| **Triton** | Triton JIT kernels | Low (Python) | Good |
| **FlyDSL** | MLIR-based DSL kernels | Low (Python + MLIR) | Good–High |

**Total ASM kernels**: 2,693 (gfx942: 1,340 / gfx950: 1,353)
**Total CK modules**: 26 primary + 9 tuning modules
**Total Triton kernels**: ~115 files across all operator categories

## 2. Coverage Matrix

### 2.1 Attention Operators

| Operator | ASM | CK | Triton | FlyDSL | CK=0 Status |
|----------|:---:|:--:|:------:|:------:|-------------|
| Flash Attention Fwd | 56 | Yes | Yes | V3/V5 | Covered (ASM primary, Triton fallback) |
| Flash Attention Bwd | 279 | Yes | Yes | — | Covered (ASM primary, Triton fallback) |
| Paged Attention | 87 | Yes | Yes | — | Covered (ASM primary, Triton fallback) |
| MLA Decode | 46 | — | Yes | — | Covered (ASM primary, Triton fallback) |
| Prefill Attention | — | — | Yes | — | Covered (Triton only) |

**Verdict**: All attention ops are fully covered without CK.

### 2.2 MOE Operators

| Operator | ASM | CK | Triton | FlyDSL | CK=0 Status |
|----------|:---:|:--:|:------:|:------:|-------------|
| Fused MOE (1-stage) | 1,676 | — | Yes | — | Covered (ASM primary, Triton fallback) |
| MOE 2-Stage Pipeline | 368 | Yes | Yes | Yes | Covered (ASM primary, Triton+FlyDSL fallback) |
| MOE Sorting | — | Yes | Yes (fallback) | — | Covered (Triton fallback with try/except) |
| MOE TopK Sigmoid | 41 | Yes | Yes | — | Covered (ASM primary, Triton fallback) |

**Verdict**: All MOE ops are covered without CK. MOE sorting has an explicit Triton fallback
in `fused_moe.py`.

### 2.3 GEMM Operators

| Operator | ASM | CK | Triton | FlyDSL | CK=0 Status |
|----------|:---:|:--:|:------:|:------:|-------------|
| GEMM BF16 | 46 | — | Yes | — | Covered (ASM+Triton) |
| GEMM FP8 (a8w8) | — | 6+ modules | Yes | Yes | Covered (Triton) |
| GEMM FP8 Blockscale | 4 | 4 modules | Yes | — | Covered (ASM+Triton) |
| GEMM FP8 Preshuffle | — | 2 modules (CK+CKTile) | — | Yes | **Partial** — FlyDSL only |
| GEMM FP8 Blockscale+Preshuffle | — | 1 module | — | — | **GAP** |
| GEMM FP4 (a4w4) | 35 (gfx950) | 1 module | Yes | Yes | Covered (ASM+Triton) |
| GEMM INT8 | 3 (gfx942) | — | Yes | Yes | Covered (ASM+Triton) |
| Batched GEMM BF16 | — | 2 modules | Yes | — | Covered (Triton) |
| Batched GEMM FP8 | — | 2 modules | Yes | — | Covered (Triton) |
| DeepGEMM (FlatMM) | — | 1 module | — | — | **GAP** |

**Verdict**: Three gaps remain — DeepGEMM, FP8 preshuffle (FlyDSL-only, no Triton), and
FP8 blockscale+preshuffle.

### 2.4 Normalization & Quantization

| Operator | ASM | CK | Triton | FlyDSL | CK=0 Status |
|----------|:---:|:--:|:------:|:------:|-------------|
| RMSNorm | 3 | Yes | Yes | Yes | Covered (ASM+Triton) |
| LayerNorm | 1 | Yes | Yes | Yes | Covered (ASM+Triton) |
| SmoothQuant | — | Yes | — | — | **GAP** |
| FP8 Quantization | — | Yes (ck_tile headers) | Yes | — | Covered (Triton) |
| MXFP4 Quantization | — | Yes (ck_tile headers) | Yes | — | Covered (Triton) |

**Verdict**: SmoothQuant is a gap — no ASM, no Triton fallback.

### 2.5 Other Operators

| Operator | ASM | CK | Triton | FlyDSL | CK=0 Status |
|----------|:---:|:--:|:------:|:------:|-------------|
| Activation (SiLU/GELU) | — | ck_tile headers | Yes | — | Covered (Triton) |
| RoPE | — | — | Yes | — | Covered (Triton) |
| KV Cache | — | ck_tile headers | Yes | — | Covered (Triton) |
| Sampling (top-p/top-k) | — | ck_tile headers | — | — | **Needs verification** |
| Softmax | — | — | Yes | Yes | Covered (Triton) |
| Top-K Selection | 41 | Yes | Yes | — | Covered (ASM+Triton) |
| Gated Delta Net | — | — | Yes | — | Covered (Triton) |
| Causal Conv1D | — | — | Yes | — | Covered (Triton) |
| AllReduce + Comms | — | — | Yes | — | Covered (Triton) |

**Verdict**: Sampling kernels need verification — the C++ code uses `ck_tile` headers.

## 3. Identified Gaps (CK=0 Blockers)

### 3.1 True Gaps — No ASM, No Triton Fallback

| # | Operator | CK Module | Severity | FlyDSL Coverage |
|---|----------|-----------|----------|-----------------|
| 1 | **DeepGEMM (FlatMM)** | `ck_deepgemm/` | High — used for large-tile GEMM | Not yet — extend preshuffle_gemm |
| 2 | **SmoothQuant** | `py_itfs_ck/smoothquant_kernels.cu` | Medium — INT8 quantization path | Not yet — element-wise kernel |
| 3 | **GEMM FP8 Blockscale+Preshuffle** | `ck_gemm_a8w8_blockscale_bpreshuffle/` | Medium — specific GEMM variant | Not yet |

### 3.2 Partial Gaps — FlyDSL Only (No Triton)

| # | Operator | CK Module | FlyDSL Status |
|---|----------|-----------|---------------|
| 4 | **GEMM FP8 Preshuffle** | `ck_gemm_a8w8_bpreshuffle/` | Done (`preshuffle_gemm.py`) — needs AITER integration |

### 3.3 `ck_tile` Header Dependencies (Build Risk)

Several C++ modules include `ck_tile` headers for utility types even though they are not
"CK kernels" per se. With `ENABLE_CK=0`, `setup.py` copies only `ck_helper/` (a minimal
header subset). If `ck_helper/` doesn't cover all needed types, these modules will fail
to compile:

| C++ Module | File | What It Needs |
|------------|------|---------------|
| `module_cache` | `cache_kernels.cu` | ck_tile buffer addressing |
| `module_activation` | `activation_kernels.cu` | ck_tile types |
| `module_quant` | `quant_kernels.cu` | ck_tile types |
| `module_sample` | `sample_kernels.cu` | ck_tile types |
| `module_moe_asm` | MOE ASM launcher | ck_tile types |

**Mitigation**: These modules have Triton alternatives (except possibly sampling). If
`ck_helper/` headers are sufficient, no action needed. If not, either expand `ck_helper/`
or add Triton fallbacks.

## 4. FlyDSL Kernel Inventory (Current)

FlyDSL already has production-ready kernels for several operator categories:

| Kernel | File | Data Types | MFMA | Status |
|--------|------|-----------|------|--------|
| Flash Attention V3 | `flash_attention_v3.py` | f16, bf16 | 16x16x16f16 | Production |
| Flash Attention V5 | `flash_attention_v5.py` | f16 | 16x16x16f16 | Production |
| Preshuffle GEMM | `preshuffle_gemm.py` | fp8, int8, int4, f16, bf16 | Multiple | Production |
| Mixed Preshuffle GEMM | `mixed_preshuffle_gemm.py` | fp8/fp4 mixed | Multiple | Production |
| MOE GEMM 2-Stage | `moe_gemm_2stage.py` | fp8, f16, int8, int4 | Multiple | Production |
| Mixed MOE GEMM | `mixed_moe_gemm_2stage.py` | fp8/fp4 + SwiGLU | Multiple | Production |
| RMSNorm | `rmsnorm_kernel.py` | f32, f16, bf16 | — | Production |
| LayerNorm | `layernorm_kernel.py` | f32, f16, bf16 | — | Production |
| Softmax | `softmax_kernel.py` | f32, f16, bf16 | — | Production |

## 5. Recommended Action Plan

### Phase 1: Verify `ENABLE_CK=0` Build (Immediate)

```bash
ENABLE_CK=0 GPU_ARCHS=gfx950 MAX_JOBS=64 python setup.py develop 2>&1 | tee ck_free_build.log
grep -i "error\|fatal\|undefined" ck_free_build.log
```

This reveals which `ck_tile` header dependencies are actually broken and whether `ck_helper/`
is sufficient.

### Phase 2: Fill True Gaps (Short-term)

**Priority order by model impact:**

1. **SmoothQuant** — Straightforward element-wise kernel. Similar complexity to
   RMSNorm/LayerNorm. FlyDSL can implement this quickly. Alternatively, write a Triton
   version since the operation is simple (per-channel scale + per-token scale + INT8 cast).

2. **DeepGEMM (FlatMM)** — Extension of the existing preshuffle GEMM with larger tiles.
   CK's FlatMM uses the same `ck_tile/18_flatmm` example as CKTile preshuffle GEMM.
   FlyDSL's `preshuffle_gemm.py` is architecturally close — needs tile size expansion and
   the FlatMM-specific scheduling.

3. **GEMM FP8 Blockscale+Preshuffle** — Combine existing FP8 blockscale Triton kernel with
   FlyDSL preshuffle layout. Or add blockscale epilogue to `preshuffle_gemm.py`.

### Phase 3: AITER Integration (Medium-term)

1. **Register FlyDSL as backend** in AITER's dispatch system alongside ASM/CK/Triton
2. **FP8 preshuffle GEMM autotuning** — Run FlyDSL kernel across AITER's 130 F8F8BBS shapes,
   generate tuned CSV for `get_GEMM_config()`
3. **MOE 2-stage integration** — Wire FlyDSL MOE kernels into `fused_moe.py` dispatch

### Phase 4: Performance Parity (Long-term)

1. FP8 GEMM autotuning for all production shapes
2. Flash Attention optimization (close the 4.7x gap vs ASM)
3. MOE 2-stage performance tuning

## 6. Appendix: Kernel Counts by Architecture

### ASM Kernels (2,693 total)

| Category | gfx942 | gfx950 | Total |
|----------|--------|--------|-------|
| Flash Attention Fwd | 48 | 8 | 56 |
| Flash Attention Bwd | 156 | 123 | 279 |
| Paged Attention | 46 | 41 | 87 |
| MLA | 19 | 27 | 46 |
| MOE (fmoe) | 834 | 842 | 1,676 |
| MOE (2-stage) | 186 | 182 | 368 |
| GEMM BF16 | 22 | 24 | 46 |
| GEMM FP8 Blockscale | 2 | 2 | 4 |
| GEMM INT8 | 3 | 0 | 3 |
| GEMM FP4 | 0 | 35 | 35 |
| Top-K / Sampling | 21 | 20 | 41 |
| Normalization | 3 | 0 | 3 |
| **Total** | **1,340** | **1,353** | **2,693** |

### CK Modules (26 primary)

| Category | Modules | Key Directories |
|----------|---------|-----------------|
| GEMM | 10 | `ck_gemm_a8w8*/`, `ck_batched_gemm*/`, `ck_deepgemm/` |
| GEMM (CKTile) | 2 | `cktile_gemm_a8w8_bpreshuffle/`, `ck_tile_gemm_moe_2stages/` |
| Attention | 4 | `py_itfs_ck/attention_kernels.cu`, `mha_*_kernels.cu` |
| MOE | 4 | `moe_sorting_kernels.cu`, `moe_ck_2stages_kernel.cu`, codegen |
| Normalization | 2 | `norm_kernels.cu`, `rmsnorm_ck_kernels.cu` |
| Quantization | 1 | `smoothquant_kernels.cu` |
| TopK | 1 | `topk_sigmoid_kernels.cu` |
| **Tuning** | 9 | `*_tune.cu` variants for autotuning |
