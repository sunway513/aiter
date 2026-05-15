# Attention Kernel Optimization Patterns

## FlashAttention Tiling Strategy
- Tile Q over BLOCK_M (rows of Q), iterate over KV in BLOCK_N chunks
- Each tile computes partial softmax; rescale accumulators when max changes
- Memory complexity: O(N) instead of O(N^2) by never materializing full attention matrix
- On CDNA4: BLOCK_M=128, BLOCK_N=64 is often optimal for head_dim=128

## Online Softmax with Rescaling
- Track running max `m_i` and sum `l_i` per row
- When new block has larger max: rescale accumulator by `exp(m_old - m_new)`
- AVO insight: **branchless accumulator rescaling** (+8.1% over branched version)
- Always use FP32 for max/sum tracking even with FP16 inputs

## Key Optimization Strategies (from AVO paper)
1. **Branchless accumulator rescaling**: Remove if/else on max comparison, always rescale
2. **Correction/MMA pipeline overlap**: Start next MFMA while correction factor is being applied
3. **Register rebalancing across warp groups**: Distribute Q/K/V/O accumulators to avoid spills

## Head Dimension Considerations
- head_dim=64: BLOCK_K=64 (single MFMA pass), very efficient
- head_dim=128: BLOCK_K=32 or 64 (2-4 MFMA passes per dot product)
- head_dim=256: High register pressure, may need to reduce BLOCK_M

## Multi-Head Attention Variants
- **MHA**: nheads_q == nheads_k, standard case
- **MQA**: nheads_k == 1, broadcast K/V across Q heads (memory bandwidth bound)
- **GQA**: nheads_q = n * nheads_k, group broadcast (intermediate case)
- For MQA/GQA: K/V reuse across heads → higher arithmetic intensity

## FP8 Attention
- Use `v_mfma_f32_32x32x32_fp8` (double the K dimension throughput)
- Quantize Q, K to FP8 before MFMA; keep accumulator in FP32
- Softmax probabilities: quantize to FP8 before V multiplication
- Watch for accuracy: FP8 E4M3 range is limited, may need per-head scaling
