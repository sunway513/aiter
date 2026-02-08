# AITER vs FlashInfer Feature Comparison

This document compares AITER (AMD, ROCm/HIP) with FlashInfer v0.6.3 (NVIDIA, CUDA) to identify feature gaps and opportunities for the AITER project.

---

## Platform Overview

| | AITER | FlashInfer |
|---|---|---|
| **Vendor** | AMD | NVIDIA |
| **GPU Support** | MI250X (gfx90a), MI300X (gfx942), MI350X (gfx950) | Turing (SM75) through Blackwell (SM120) |
| **Backends** | ASM, CK/CKTile, Triton, hipBLASLt | CUTLASS, cuDNN, Triton, cuBLAS |
| **Compilation** | HIP JIT (`@compile_ops`), AOT precompile | TVM-FFI JIT, AOT |
| **License** | MIT | Apache 2.0 |

---

## Feature Comparison Matrix

### Attention

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Flash Attention (forward) | Yes | Yes | Both have optimized implementations |
| Flash Attention (backward) | Yes | Yes | |
| Paged Attention (decode) | Yes | Yes | |
| Paged Attention (prefill) | Yes | Yes | |
| Chunked Prefill | Yes | Yes | |
| GQA / MQA | Yes | Yes | |
| MLA (DeepSeek) | Yes | Yes | Both have dedicated MLA kernels |
| Sliding Window | Yes | Yes | |
| ALiBi | Yes | Yes | |
| Logits Soft Cap | Yes | Yes | |
| Custom Masks (bit-packed) | No | Yes | FlashInfer: `packbits`, ragged mask tensors |
| **POD Attention** | **No** | **Yes** | Fuses prefill + decode for mixed batching |
| **Cascade/Multi-level Attention** | **No** | **Yes** | Hierarchical KV-cache with shared prefixes, 30x claimed for long-context |
| **Block-Sparse Attention** | **No** | **Yes** | Fixed and variable block sizes |
| **Attention Sinks** | **No** | **Yes** | Built-in wrapper for sink tokens |
| **Attention State Merging** | **No** | **Yes** | `merge_state`, `merge_states` for recursive attention |
| Variable-length (unpadded) | Yes | Yes | Both support cu_seqlens |
| FP8 Attention | Yes | Yes | |

### KV-Cache

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Paged KV-Cache | Yes | Yes | |
| Flash (contiguous) KV-Cache | Yes | Yes | |
| MLA KV-Cache | Yes | Yes | |
| FP8 KV-Cache | Yes | Yes | |
| INT8 KV-Cache | Yes | No | AITER advantage |
| Fused RoPE + Cache Write | Yes | No | AITER advantage |
| **Cascade KV-Cache (shared prefixes)** | **No** | **Yes** | Multi-level hierarchical reuse |
| **KV-Cache Append Operations** | **No** | **Yes** | `append_paged_kv_cache` with page management |

### Sampling

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Greedy (argmax) | Yes | Yes | |
| Temperature-scaled random | Yes | Yes | |
| Mixed (greedy/random per row) | Yes | No | AITER advantage |
| Top-k sampling | Yes | Yes | |
| Top-p (nucleus) sampling | Yes | Yes | |
| Joint top-k + top-p | Yes | Yes | |
| **Min-p sampling** | **No** | **Yes** | Minimum probability threshold |
| **Chain speculative sampling** | **No** | **Yes** | With acceptance metrics |
| **Logits Processing Pipeline** | **No** | **Yes** | Composable `LogitsPipe` with Temperature, TopK, TopP, MinP, Sample, Softmax operators and compiler fusion |
| Outer-exponential (reproducible) | Yes | No | AITER advantage |
| Deterministic sampling | Yes | Yes | |

### GEMM

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| BF16/FP16 GEMM | Yes | Yes | |
| FP8 GEMM | Yes | Yes | |
| FP4 GEMM (MXFP4) | Yes | Yes | |
| A8W8 per-tensor | Yes | Yes | |
| A8W8 blockscale | Yes | Yes | |
| Batched GEMM | Yes | Yes | |
| Weight preshuffle | Yes | Yes | Both optimize weight layout |
| **Segment GEMM (LoRA)** | **No** | **Yes** | Grouped GEMM for batched LoRA inference |
| CSV-based tuning system | Yes | No | AITER advantage — automated dispatch |
| hipBLASLt integration | Yes | N/A | AMD-specific |
| Assembly GEMM kernels | Yes | No | AITER advantage — hand-tuned ASM |
| **cuDNN GEMM backend** | **N/A** | **Yes** | NVIDIA-specific |

### MOE (Mixture of Experts)

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Fused MOE (FP8) | Yes | Yes | |
| Fused MOE (FP4/MXFP4) | Yes | Yes | |
| Grouped top-k routing | Yes | Yes | |
| Biased grouped top-k (DeepSeek-V3) | Yes | Yes | |
| 2-stage MOE | Yes | No | AITER advantage |
| MOE sorting | Yes | No | AITER advantage |
| **CuTe DSL-based MOE** | **No** | **Yes** | NVIDIA CuTe DSL |
| **MoE All-to-All** | **Partial** | **Yes** | FlashInfer: TRT-LLM all-to-all; AITER: MoRI/DeepEP |
| **Non-gated ReLU2 MOE** | **No** | **Yes** | Nemotron support |

### Normalization

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| RMSNorm | Yes | Yes | |
| LayerNorm | Yes | Yes | |
| GroupNorm | Yes | No | AITER advantage |
| Fused Add + RMSNorm | Yes | Yes | |
| Fused RMSNorm + Quant (FP8) | Yes | No | AITER advantage |
| SmoothQuant fusion | Yes | No | AITER advantage |
| **Gemma RMSNorm** | **No** | **Yes** | Gemma-specific variant |
| **Fused RMSNorm + FP4 Quant** | **No** | **Yes** | `rmsnorm_fp4quant` |
| Distributed fused all-reduce + RMSNorm | Yes | No | AITER advantage |

### RoPE

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Standard RoPE (SBHD/THD) | Yes | Yes | |
| NeoX / GPT-J styles | Yes | Yes | |
| 2D/3D RoPE | Yes | No | AITER advantage |
| Scaling methods (Linear, YaRN, Llama3) | Yes | Yes | |
| **LLaMA 3.1 RoPE** | **No** | **Yes** | Dedicated `apply_llama31_rope` |
| In-place RoPE | Yes | Yes | |
| Fused RoPE + Cache Write | Yes | No | AITER advantage |

### Communication / Distributed

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Custom All-Reduce (IPC) | Yes | Yes | |
| Quantized All-Reduce (FP8/INT4/INT6) | Yes | No | AITER advantage |
| Iris GPU-initiated comms | Yes | N/A | AMD-specific |
| Fused All-Reduce + RMSNorm | Yes | No | AITER advantage |
| **NVSHMEM** | **N/A** | **Yes** | NVIDIA-specific |
| **MNNVL (Multi-Node NVLink)** | **N/A** | **Yes** | NVIDIA-specific |
| ASM-level All-Reduce | Yes | No | AITER advantage |
| Shared Memory Broadcast | Yes | No | AITER advantage |

### Linear Attention / SSM

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| Gated Delta Net (recurrent) | Yes | Yes | Both have GDN |
| Gated Delta Net (chunk) | Yes | Yes | |
| Causal Conv1D | Yes | No | AITER advantage (Triton) |
| Fused Conv1D + QKV split | Yes | No | AITER advantage |
| **Mamba MTP (Multi-Token Prediction)** | **No** | **Yes** | FlashInfer: `gated_delta_rule_mtp` |

### Other

| Feature | AITER | FlashInfer | Notes |
|---------|:-----:|:----------:|-------|
| BERT Padding/Unpadding | Yes | No | AITER: dedicated utilities |
| **SM Partitioning (Green Context)** | **No** | **Yes** | Concurrent kernels on different SM groups |
| **torch.compile integration** | **No** | **Yes** | FlashInfer supports torch.compile |
| CUDA Graph support | Partial | Yes | FlashInfer: dedicated CUDA graph wrappers |

---

## Top Feature Gaps in AITER (Prioritized)

### Priority 1 — High Impact for LLM Serving

| # | Feature | FlashInfer API | Impact | Effort |
|---|---------|---------------|--------|--------|
| 1 | **POD Attention** | `PODWithPagedKVCacheWrapper` | Fuses prefill + decode batches for mixed serving — reduces scheduling complexity and improves GPU utilization | High |
| 2 | **Cascade / Multi-level Attention** | `MultiLevelCascadeAttentionWrapper` | Shared-prefix KV-cache reuse for system prompts / few-shot — up to 30x claimed for long-context large-batch scenarios | High |
| 3 | **Segment GEMM (LoRA)** | `SegmentGEMMWrapper` | Efficient batched LoRA inference — multiple LoRA adapters in one batch without padding or sequential dispatch | High |
| 4 | **Logits Processing Pipeline** | `LogitsPipe`, `Compiler` | Composable, fused logits pipeline (Temperature → TopK → TopP → Sample) with compiler optimization — avoids multiple kernel launches | Medium |
| 5 | **Chain Speculative Sampling** | `chain_speculative_sampling` | Speculative decoding acceptance with metrics — enables efficient multi-draft verification | Medium |

### Priority 2 — Model-Specific & Algorithmic

| # | Feature | FlashInfer API | Impact | Effort |
|---|---------|---------------|--------|--------|
| 6 | **Block-Sparse Attention** | `BlockSparseAttentionWrapper` | Efficient attention with structured sparsity patterns — reduces compute for long sequences | Medium |
| 7 | **Attention Sinks** | `BatchAttentionWithAttentionSinkWrapper` | StreamingLLM-style attention with sink tokens — enables infinite-length generation | Low |
| 8 | **Min-p Sampling** | `min_p_sampling_from_probs` | Alternative to top-p — dynamically adjusts threshold based on max probability | Low |
| 9 | **Attention State Merging** | `merge_state`, `merge_states` | Split-and-merge attention states — enables distributed attention, prefix sharing | Medium |
| 10 | **Mamba MTP** | `gated_delta_rule_mtp` | Multi-token prediction for Mamba-style models — speculative decoding | Medium |

### Priority 3 — Infrastructure & Tooling

| # | Feature | FlashInfer API | Impact | Effort |
|---|---------|---------------|--------|--------|
| 11 | **Custom Bit-Packed Masks** | `packbits`, `segment_packbits` | Memory-efficient attention masks — useful for complex masking patterns | Low |
| 12 | **Gemma RMSNorm** | `gemma_rmsnorm` | Model-specific normalization — needed for Gemma family | Low |
| 13 | **LLaMA 3.1 RoPE** | `apply_llama31_rope` | Dedicated RoPE for LLaMA 3.1 scaling — currently handled by general scaling methods | Low |
| 14 | **Fused RMSNorm + FP4 Quant** | `rmsnorm_fp4quant` | Fused normalization + FP4 quantize — reduces memory bandwidth | Low |

---

## AITER Advantages Over FlashInfer

Features that AITER has which FlashInfer lacks or cannot support:

| Feature | AITER Capability |
|---------|-----------------|
| **AMD/ROCm Support** | First-class support for MI250X, MI300X, MI350X — FlashInfer is NVIDIA-only |
| **Hand-tuned ASM Kernels** | Assembly-level GEMM and attention for maximum AMD GPU performance |
| **Composable Kernel (CK) Backend** | AMD's CK library provides optimized tensor operations |
| **Quantized All-Reduce** | FP8/INT6/INT4 quantized communication — reduces interconnect bandwidth |
| **CSV-based GEMM Autotuning** | Automated per-shape kernel dispatch with offline tuning — production-ready |
| **Fused All-Reduce + RMSNorm** | Single-kernel distributed normalization — reduces latency |
| **2-Stage MOE** | Efficient two-stage MoE execution with sorting |
| **GroupNorm** | GroupNorm support with fused variants |
| **SmoothQuant Fusion** | Fused SmoothQuant in normalization kernels |
| **Fused RoPE + Cache Write** | Single-kernel RoPE + KV-cache update |
| **2D/3D RoPE** | Multi-dimensional position encodings |
| **Mixed Sampling** | Per-row greedy/random dispatch for batch inference |
| **INT8 KV-Cache** | 8-bit integer KV-cache quantization |
| **Causal Conv1D** | Triton-based causal convolution with fused QKV split |
| **ASM All-Reduce** | Assembly-level all-reduce for CUDA graph compatibility |

---

## Recommendations

1. **POD Attention** and **Cascade Attention** are the highest-impact gaps — they directly affect LLM serving throughput for mixed-batch and shared-prefix workloads (the most common production patterns).

2. **Segment GEMM for LoRA** is increasingly important as LoRA-based fine-tuning becomes standard — serving multiple LoRA adapters efficiently requires grouped GEMM with adapter indirection.

3. **Logits Processing Pipeline** is a quality-of-life improvement that reduces kernel launch overhead and simplifies sampling code in serving frameworks.

4. **Block-Sparse Attention** and **Attention State Merging** become critical for long-context models (128K+ tokens).

5. Many FlashInfer features are NVIDIA-specific (cuDNN, NVSHMEM, MNNVL, CuTe DSL, SM partitioning) and have no direct AMD equivalent — these should be addressed through AMD-native alternatives rather than direct ports.
