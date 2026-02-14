Quickstart
==========

This guide will get you started with AITER in 5 minutes.

Installation
------------

.. code-block:: bash

   # Install from source
   git clone --recursive https://github.com/ROCm/aiter.git
   cd aiter
   python3 setup.py develop

Verify Installation
-------------------

.. code-block:: python

   import aiter
   print(f"AITER version: {aiter.__version__}")

First Example: Flash Attention
-------------------------------

Here's a simple example using AITER's optimized attention kernel:

.. code-block:: python

   import torch
   import aiter

   # Input tensors (batch_size=2, seq_len=1024, num_heads=16, head_dim=64)
   batch_size, seq_len, num_heads, head_dim = 2, 1024, 16, 64

   query = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)
   key = torch.randn(batch_size, seq_len, num_heads, head_dim,
                     device='cuda', dtype=torch.float16)
   value = torch.randn(batch_size, seq_len, num_heads, head_dim,
                       device='cuda', dtype=torch.float16)

   # Run optimized flash attention
   output = aiter.flash_attn_func(query, key, value, causal=True)

   print(f"Output shape: {output.shape}")
   # Output shape: torch.Size([2, 1024, 16, 64])

Variable-Length Sequences
-------------------------

AITER excels at handling variable-length sequences with page tables:

.. code-block:: python

   import torch
   import aiter

   # Query with variable lengths per batch
   query = torch.randn(5, 2048, 16, 64, device='cuda', dtype=torch.float16)

   # Page table configuration (see tutorials for details)
   page_table = torch.tensor([[0, 1, 2], [3, 4, 5]], device='cuda', dtype=torch.int32)

   # KV cache in paged format
   kv_cache = torch.randn(6, 16, 128, 64, device='cuda', dtype=torch.float16)

   # Variable-length attention with page tables
   output = aiter.flash_attn_with_kvcache(
       query, kv_cache, page_table,
       block_size=128, causal=True
   )

Mixture of Experts (MoE)
------------------------

Efficient grouped GEMM for MoE layers:

.. code-block:: python

   import torch
   import aiter

   # Expert selection (batch_size * seq_len, top_k)
   expert_ids = torch.randint(0, 8, (4096, 2), device='cuda')

   # Input tokens
   x = torch.randn(4096, 512, device='cuda', dtype=torch.float16)

   # Expert weights (num_experts, hidden_dim, ffn_dim)
   expert_weights = torch.randn(8, 512, 2048, device='cuda', dtype=torch.float16)

   # Grouped GEMM for MoE
   output = aiter.grouped_gemm(x, expert_weights, expert_ids)

   print(f"MoE output shape: {output.shape}")

RMSNorm
-------

Optimized normalization for LLM inference:

.. code-block:: python

   import torch
   import aiter

   # Input tensor (batch_size, seq_len, hidden_dim)
   x = torch.randn(2, 1024, 4096, device='cuda', dtype=torch.float16)

   # Weight for normalization
   weight = torch.ones(4096, device='cuda', dtype=torch.float16)

   # Fast RMSNorm
   output = aiter.rmsnorm(x, weight, eps=1e-6)

Performance Tips
----------------

1. **Use FP16/BF16**: AITER kernels are optimized for half-precision
2. **Enable compilation**: Set ``PREBUILD_KERNELS=2`` for inference workloads
3. **Batch when possible**: Larger batches better utilize GPU
4. **Profile first**: Use ROCm profiler to identify bottlenecks

.. code-block:: bash

   # Example: Profile your workload
   rocprof --stats python your_script.py

Next Steps
----------

* :doc:`tutorials/attention` - Deep dive into attention mechanisms
* :doc:`tutorials/moe` - Learn about MoE optimizations
* :doc:`tutorials/variable_length` - Handle variable-length sequences
* :doc:`api/attention` - Full API reference
* :doc:`benchmarks` - Performance comparisons

Common Issues
-------------

**ImportError: No module named 'aiter'**
   Make sure ROCm libraries are in your library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   Verify GPU is accessible:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx

**Compilation errors during first run**
   JIT compilation may take time on first use. Pre-compile kernels:

   .. code-block:: bash

      PREBUILD_KERNELS=2 GPU_ARCHS="native" python3 setup.py install

Get Help
--------

* **Documentation**: https://doc.aiter.amd.com
* **GitHub Issues**: https://github.com/ROCm/aiter/issues
* **ROCm Community**: https://github.com/ROCm/ROCm/discussions
