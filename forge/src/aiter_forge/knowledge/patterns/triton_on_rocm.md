# Triton on ROCm Optimization Patterns

## Block Size Recommendations for CDNA4
- BLOCK_M, BLOCK_N: use multiples of 32 (maps to MFMA tile size)
- Common effective sizes: 64, 128, 256
- BLOCK_K: 32 for FP16, 64 for FP8 (matches MFMA K dimension)
- Avoid BLOCK < 32 on any dimension (underutilizes MFMA units)

## tl.dot → MFMA Mapping
- `tl.dot(a, b)` compiles to MFMA instructions
- a shape: (BLOCK_M, BLOCK_K), b shape: (BLOCK_K, BLOCK_N)
- For best throughput: BLOCK_M=BLOCK_N=128, BLOCK_K=32 (FP16) or 64 (FP8)
- Accumulator is always FP32

## LDS (Shared Memory) Usage
- Triton automatically uses LDS for tl.load with block pointers
- LDS budget: 64KB per CU, shared across wavefronts
- Large blocks (256x128) may exceed LDS → reduce block size or num_warps
- `num_warps`: 4 or 8 typical. More warps = more LDS pressure

## Known Performance Pitfalls on ROCm
1. **num_stages**: Software pipelining stages. ROCm Triton supports 1-2 stages (not 4+ like CUDA)
2. **Atomic operations**: `tl.atomic_add` on FP16 can be slow; accumulate in FP32 then convert
3. **Predicated loads**: Masked loads generate scalar predicates; avoid complex mask expressions
4. **Compilation time**: Large kernels can take 60s+ to compile; use persistent kernel cache

## Autotune Parameter Ranges for MI355X
```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'num_warps': w, 'num_stages': s})
    for m in [64, 128, 256]
    for n in [64, 128, 256]
    for k in [32, 64]
    for w in [4, 8]
    for s in [1, 2]
], key=['M', 'N', 'K'])
```
