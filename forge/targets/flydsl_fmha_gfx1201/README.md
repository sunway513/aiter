# FlyDSL FMHA for gfx1201 (RDNA4)

## Goal

Port `kernels/flash_attn_func.py` (CDNA MFMA 32x32) to RDNA4 WMMA 16x16x16 for R9600D.

**Target**: Wan2.1 DiT attention — B=2, H=12, S=18900, D=128, bf16, non-causal.

## Current Performance

| Method | Time (ms) | TFLOPS | Gap to 4090 |
|--------|----------|--------|-------------|
| SDPA (baseline) | 148 | ~30 | 4.4x |
| Triton FA (tuned) | 95 | ~46 | 2.8x |
| **FlyDSL FMHA (target)** | **<60** | **>73** | **<1.8x** |
| 4090 Flash Attention 2 | 34 | ~130 | 1.0x |

## Adaptation Checklist

### Must Change (CDNA → RDNA4)

- [ ] **WMMA instruction**: `mfma_f32_32x32x16_bf16` → `wmma_f32_16x16x16_bf16`
  - Fragment: v16f32 (32x32 acc) → v4f32 (16x16 acc)
  - A/B operand: v8bf16 → v4bf16
  - Tile: 32x32xK → 16x16x16
  - Location: `flash_attn_func.py` lines 247-261

- [ ] **Wave size**: `WARP_SIZE = 64` → `32`
  - `NUM_WAVES = flat_work_group_size // WARP_SIZE` stays 256/32=8 waves
  - Location: line 83

- [ ] **Lane/wave decomposition**: `wave_id = tid // 64, lane = tid % 64`
  - MFMA 32x32: lane maps to 32 rows × 2 groups
  - WMMA 16x16: lane maps to 16 rows × 2 groups (wave32)
  - Location: lines 279-280, all lane arithmetic

- [ ] **Tile sizes**: `BLOCK_M=128/256, BLOCK_N=64`
  - RDNA4: `BLOCK_M=64/128, BLOCK_N=32` (fit WMMA 16x16)
  - Per-wave Q rows: 32 → 16
  - Location: throughout

- [ ] **Online softmax reduction**: warp shuffle widths
  - wave64: `[32, 16, 8, 4, 2, 1]`
  - wave32: `[16, 8, 4, 2, 1]`
  - Location: softmax reduction in inner loop

- [ ] **LDS swizzle**: XOR patterns
  - K swizzle: `col ^ ((row & 7) << 4)` — verify bank layout for RDNA4
  - V swizzle: `col ^ ((row & 3) << 4)` — same
  - Location: lines 393-398, 547

### May Need Change

- [ ] DMA-to-LDS: verify `buffer_load` + `ds_write_b128` works on gfx1201
- [ ] `s_waitcnt` encoding: RDNA4 may use different waitcnt format
- [ ] Register pressure: RDNA4 has 256 VGPRs per wave32 (vs 512 per wave64 on CDNA)
- [ ] `rocdl.sched_barrier` / scheduling hints: may not apply to RDNA4

### Keep As-Is

- [x] Online softmax algorithm (mathematically identical)
- [x] P kept in registers between GEMM1 and GEMM2 (no LDS roundtrip)
- [x] Non-causal path (no masking needed)
- [x] Global memory layout (BSHD)
- [x] SmemAllocator usage pattern

## Reference PRs

| PR | Description | Status |
|----|-------------|--------|
| ROCm/FlyDSL#225 | FMHA kernel (CDNA, merged) | **Base code** |
| ROCm/FlyDSL#335 | gfx1250 FA + MLA | Closed, has gfx12 patterns |
| ROCm/FlyDSL#109 | FA V4.x (Peng's MFMA 16x16x16) | Closed, simpler reference |
| ROCm/FlyDSL#288 | bf16 16x16x16 MFMA support | Merged, ROCDL bindings |

## How to Develop

```bash
# Set up environment
export FLYDSL_ROOT=/path/to/FlyDSL
export FLIR_CHIP=gfx1201

# Quick test (small shape)
python tests/kernels/test_flash_attn_func_gfx1201.py -b 1 -h 12 -s 1024 -d 128

# Wan2.1 DiT shape
python tests/kernels/test_flash_attn_func_gfx1201.py -b 2 -h 12 -s 18900 -d 128 --benchmark

# Compare against Triton FA baseline
python fw-bringup/scripts/bench_attn_r9600d_final.py
```

## R9600D Server Access

```bash
sshpass -p 'admin' ssh root@10.67.79.140
docker exec -it wan-bench bash
# 16x R9600D GPUs, PyTorch 2.10.0+rocm7.2.2
```
