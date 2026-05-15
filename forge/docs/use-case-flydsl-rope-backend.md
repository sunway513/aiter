# Use Case: Adding FlyDSL RoPE Backend to AITER via aiter-forge

## Overview

This document describes how aiter-forge was used to validate and benchmark a new FlyDSL-based RoPE kernel backend for AITER. The process took approximately **22 minutes** from initial analysis to full validation across 48 model configurations.

## Problem Statement

AITER's fused RoPE + KV Cache operation (`fused_qk_rope_reshape_and_cache`) had only Triton and HIP backends. FlyDSL (AMD's MLIR-based GPU kernel DSL) had an existing fused RoPE+KVCache kernel that was ~1.5x faster than the Triton path. Goal: integrate FlyDSL RoPE into AITER with **zero accuracy loss** and **no interface changes**.

## Challenges

1. **Data format mismatch**: AITER uses int64 positions/slot_mapping; FlyDSL kernel expected int32
2. **Cos/Sin cache shape**: AITER passes 4D `[max_pos, 1, 1, D//2]`; FlyDSL expected 2D `[max_pos, D//2]`
3. **Validation at scale**: Need to verify across 12 models × 4 token counts × 2 layouts × 2 dtypes

## How aiter-forge Helped

### Phase 1: Analysis (operator-dashboard integration)

Used ROCm operator-dashboard data to identify RoPE as a high-impact optimization target (5.5% avg SOL on MI300X with pure-torch). This drove the decision to pursue FlyDSL integration.

### Phase 2: Baseline Benchmarking (Triton auto-tune)

Created `targets/rope/` target to run Triton auto-tune across 16 dashboard shapes on 4 GPUs simultaneously. Results showed Triton best at 67.5% SOL — good but with room for improvement.

### Phase 3: FlyDSL Validation (target-driven)

Created `targets/flydsl_rope/target.yaml` with:
- **Correctness test**: pytest against PyTorch reference
- **15 benchmark shapes**: decode, small batch, medium, large, non-flash, TP8
- **Scoring**: bandwidth (GB/s) as primary metric
- **Validation rules**: precision checks, GPU isolation

### Phase 4: Automated Sweep

The aiter-forge benchmark script (`bench_flydsl_rope.py`) outputs aiter-forge compatible tables:
```
TOKENS  QHEADS  KVHEADS  DIM  LAYOUT  LATENCY_US  GB/s
128     128     8        128  flash   39.1        248.97
```

This enables aiter-forge's scoring system to track performance across shapes and detect regressions.

## Results

| Metric | Value |
|--------|-------|
| Total configs tested | 48 (12 models × 4 token counts) |
| Accuracy vs PyTorch ref | **0.00 max error** (bit-identical for bf16) |
| FlyDSL/AITER speedup | **1.50x average** (1.46x–1.55x range) |
| Time to complete | **22 minutes** |

## Files Created

```
aiter-forge/
└── targets/
    └── flydsl_rope/
        ├── target.yaml           # aiter-forge target definition
        └── bench/
            └── bench_flydsl_rope.py  # Benchmark script

aiter/
├── ops/flydsl/
│   └── rope_kernels.py           # (planned) AITER wrapper
└── op_tests/flydsl_tests/
    └── test_flydsl_rope.py       # Unit tests

FlyDSL/
└── kernels/
    └── fused_rope_cache_kernel.py  # Modified: +pos_int64, +cos_sin_4d params

flydsl_rope_validation/
├── golden/                       # Step 0: 20 shapes × full tensor snapshots
├── step0_golden_reference.py
├── step1_2_3_validate.py
├── step4_5_6_full_sweep.py
└── *.json                        # All results
```

## Kernel Modification Summary

Only 3 changes to FlyDSL kernel (`fused_rope_cache_kernel.py`):

1. **New parameter**: `pos_int64: bool = False` — compile-time flag
2. **Position loading**: `buffer_load(pos_rsrc, pid_t*2, vec_width=2, dtype=T.i32)` then extract low 32 bits
3. **Slot loading**: Same pattern as positions

Total diff: ~20 lines changed. Zero performance impact (verified: 0 regression across 48 configs).
