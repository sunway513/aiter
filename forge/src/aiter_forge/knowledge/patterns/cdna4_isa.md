# CDNA4 ISA Optimization Patterns (MI355X)

## Architecture Overview
- MI355X: CDNA4 architecture, 256 Compute Units (CUs)
- Wavefront size: 64 threads (always)
- Max wavefronts per CU: 8 (occupancy target)
- VGPR per CU: 512 (each 64-wide, so 32KB per wavefront if using 256 VGPRs)
- LDS per CU: 64 KB shared across wavefronts on that CU
- HBM3E: ~8 TB/s aggregate bandwidth

## MFMA Instructions
- Matrix Fused Multiply-Add: the primary compute primitive
- FP16: `v_mfma_f32_32x32x16_f16` (32x32 output, K=16, throughput ~1024 FLOPs/cycle/CU)
- FP8: `v_mfma_f32_32x32x32_fp8` (32x32 output, K=32, ~2048 FLOPs/cycle/CU)
- BF16: `v_mfma_f32_32x32x16_bf16` (same shape as FP16)
- Smaller tiles: 16x16 variants exist but lower throughput per instruction

## Key Optimization Patterns
1. **Tile to MFMA size**: Block dimensions should be multiples of 32 (MFMA native size)
2. **Minimize VGPR usage**: Each VGPR above 128 per wavefront reduces max occupancy
3. **LDS bank conflicts**: 32 banks, 4-byte stride. Pad shared memory to avoid conflicts
4. **Async global loads**: Use `buffer_load` with `s_waitcnt` for latency hiding
5. **Register rebalancing across warp groups**: Distribute accumulator registers to avoid spills
6. **Occupancy sweet spot**: 4-5 wavefronts/CU often better than max 8 (more registers available)

## Memory Hierarchy
- Registers: fastest, but limited (256 VGPRs max per wavefront for full occupancy)
- LDS: 64KB/CU, ~16 TB/s bandwidth, 1-2 cycle latency
- L2 Cache: shared across all CUs, ~4 TB/s
- HBM3E: ~8 TB/s, 100+ cycle latency
