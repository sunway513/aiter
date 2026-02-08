# AITER MORI All-to-All Communication Guide

This guide documents MORI (AMD's optimized all-to-all communication library) integration in AITER for Expert Parallelism (EP) in Mixture-of-Experts models.

---

## Quick Reference

| Use Case | Recommended API | Why |
|----------|----------------|-----|
| **EP in inference engine (ATOM, vLLM)** | `MoriAll2AllManager` via EP group communicator | Automatic handle caching, lazy init, proper EP group setup |
| **Standalone EP testing** | `MoriAll2AllManager` directly | Same caching benefits, explicit control |
| **Low-level benchmarking** | `mori.ops.EpDispatchCombineOp` directly | No AITER overhead, direct MORI API |

---

## 1. Overview

MORI provides optimized GPU-to-GPU all-to-all communication for MoE Expert Parallelism. In EP, each GPU holds a subset of experts. The all-to-all communication has two phases:

1. **Dispatch**: Scatter tokens from all GPUs to the GPUs hosting their assigned experts
2. **Combine**: Gather expert outputs back to the originating GPUs

```
GPU 0 (Experts 0-3)    GPU 1 (Experts 4-7)    GPU 2 (Experts 8-11)   GPU 3 (Experts 12-15)
     │                       │                       │                       │
     ├── tokens for E0-3 ────┤── tokens for E4-7 ────┤── tokens for E8-11 ──┤── tokens for E12-15
     │      DISPATCH          │      DISPATCH          │      DISPATCH         │      DISPATCH
     ▼                       ▼                       ▼                       ▼
  [local MoE GEMM]       [local MoE GEMM]       [local MoE GEMM]       [local MoE GEMM]
     │                       │                       │                       │
     ├── results back ────────┤── results back ────────┤── results back ───────┤── results back
     │      COMBINE           │      COMBINE           │      COMBINE          │      COMBINE
     ▼                       ▼                       ▼                       ▼
  [aggregated output]    [aggregated output]    [aggregated output]    [aggregated output]
```

MORI automatically selects between two kernel types based on topology:

| Topology | Kernel Type | `block_num` | `rdma_block_num` | `warp_num_per_block` |
|----------|-------------|-------------|-------------------|----------------------|
| **Single node** (all GPUs on same node) | `IntraNode` | 80 | 0 | 16 |
| **Multi-node** (GPUs span nodes) | `InterNodeV1` | 32 | 16 | 16 |

---

## 2. Architecture

### Class Hierarchy

```
DeviceCommunicatorBase          (base_device_communicator.py)
    └── CudaCommunicator        (communicator_cuda.py)
            └── .all2all_manager  [lazy property]
                    └── MoriAll2AllManager  (all2all.py)

All2AllManagerBase              (base_device_communicator.py)
    └── MoriAll2AllManager      (all2all.py)
            └── .handle_cache → Cache (WeakValueDictionary)
```

### Initialization Flow

```
aiter.init_dist_env(tp_size, rank, ...)
    └── creates EP group (merged from DP × TP ranks)
        └── EP group has CudaCommunicator as device_communicator
            └── CudaCommunicator.all2all_manager (lazy property)
                └── first access creates MoriAll2AllManager(cpu_group)
                    ├── registers process group with MORI:
                    │   torch._C._distributed_c10d._register_process_group("mori", cpu_group)
                    ├── initializes shared memory:
                    │   mori.shmem.shmem_torch_process_group_init("mori")
                    └── creates handle_cache (WeakValueDictionary)
```

The `all2all_manager` property is **lazy** — it is not created until first access. This avoids initialization ordering issues since EP groups depend on both TP and DP groups being formed first.

### EP Group Formation

The EP group merges DP and TP ranks. From `parallel_state.py`:

```python
# EP group = DP × TP ranks merged
all_ranks.transpose(1, 2).reshape(-1, dp_size * tp_size)
```

For example, with `dp_size=4, tp_size=2`:
- EP group size = 8 (all 4 DP ranks × 2 TP ranks)
- Each EP rank holds `total_experts / ep_size` local experts

---

## 3. Handle Creation and Caching

Each MoE layer may have different dimensions (hidden size, intermediate size, etc.). MORI handles are created per-configuration and cached via `WeakValueDictionary`:

```python
mori_manager = ep_group.device_communicator.all2all_manager
handle_kwargs = dict(
    rank=rank_id,
    num_ep_ranks=world_size,
    input_dtype=torch.bfloat16,          # original token dtype
    quant_dtype=tokens_qt.dtype,          # quantized dtype (e.g., fp8)
    token_hidden_size=hidden_dim,         # model hidden dimension
    scale_dim=scale.shape[-1],            # quantization scale dimension
    scale_type_size=scale.dtype.itemsize, # bytes per scale element
    max_num_tokens_per_dp_rank=max_tokens,# buffer size per DP rank
    num_local_experts=E // world_size,    # experts on this GPU
    num_experts_per_token=topk,           # top-k routing
    gpu_per_node=gpus_per_node,           # for IntraNode/InterNode selection
)
mori_op = mori_manager.get_handle(handle_kwargs)
```

Internally, `get_handle()` calls `_make_all2all_kwargs()` which maps these to MORI's native config parameters:

| `_make_all2all_kwargs` param | → MORI config param | Description |
|------------------------------|---------------------|-------------|
| `rank` | `rank` | GPU rank in EP group |
| `num_ep_ranks` | `world_size` | Total GPUs in EP group |
| `quant_dtype` | `data_type` | Quantized data type for communication |
| `token_hidden_size` | `hidden_dim` | Hidden dimension per token |
| `scale_dim` | `scale_dim` | Quantization scale vector length |
| `scale_type_size` | `scale_type_size` | Bytes per scale element |
| `input_dtype.itemsize` | `max_token_type_size` | Bytes per original token element |
| `max_num_tokens_per_dp_rank` | `max_num_inp_token_per_rank` | Max tokens any single rank can send |
| `num_local_experts` | `num_experts_per_rank` | Local experts per GPU |
| `num_experts_per_token` | `num_experts_per_token` | Top-k value |
| *(auto-selected)* | `warp_num_per_block` | Always 16 |
| *(auto-selected)* | `block_num` | 80 (IntraNode) or 32 (InterNode) |
| *(auto-selected)* | `kernel_type` | `IntraNode` or `InterNodeV1` based on topology |
| *(auto-selected)* | `rdma_block_num` | 0 (IntraNode) or 16 (InterNode) |
| `gpu_per_node` | `gpu_per_node` | GPUs per node (for topology selection) |

The handle is then created as:
```python
mori_config = mori.ops.EpDispatchCombineConfig(**mori_kwargs)
handle = mori.ops.EpDispatchCombineOp(mori_config)
```

---

## 4. Dispatch and Combine Operations

### 4.1 Dispatch

Scatters quantized tokens to the GPUs hosting their assigned experts.

```python
(
    dispatch_output,        # [max_recv_tokens, hidden_dim] — received token activations
    dispatch_weights,       # [max_recv_tokens, topk]       — routing weights for received tokens
    dispatch_scale,         # [max_recv_tokens, scale_dim]  — quantization scales
    dispatch_ids,           # [max_recv_tokens, topk]       — expert IDs for received tokens
    dispatch_recv_token_num,# scalar tensor (int32)         — actual number of received tokens
) = mori_op.dispatch(tokens_qt, topk_weights, scale, topk_ids)
```

**Parameters:**
- `tokens_qt` — `[num_local_tokens, hidden_dim]`, quantized token activations (e.g., FP8)
- `topk_weights` — `[num_local_tokens, topk]`, routing weights from top-k selection
- `scale` — `[num_local_tokens, scale_dim]`, quantization scales (or empty tensor if no quantization)
- `topk_ids` — `[num_local_tokens, topk]`, global expert IDs assigned by the router

### 4.2 Local MoE Computation

After dispatch, run `fused_moe` on the received tokens with only local experts:

```python
expert_mask = torch.zeros(total_experts, dtype=torch.int32, device=device)
expert_mask[local_start:local_end] = 1

output = fused_moe(
    dispatch_output, w1, w2,
    dispatch_weights, dispatch_ids,
    expert_mask,
    num_local_tokens=dispatch_recv_token_num,
    w1_scale=w1_scale, w2_scale=w2_scale,
    a1_scale=dispatch_scale,
    quant_type=quant_type,
    dtype=torch.bfloat16,
)
```

### 4.3 Combine

Gathers expert outputs back to originating GPUs and applies routing weight aggregation.

```python
combine_output = mori_op.combine(output, topk_weights, topk_ids)
# combine_output: [max_recv_tokens, hidden_dim] — use [:num_local_tokens] for final result
```

The combine result may be a tuple (implementation-dependent); extract the tensor:
```python
if isinstance(combine_output, tuple):
    combine_output = combine_output[0]
result = combine_output[:num_local_tokens]
```

---

## 5. Supported All-to-All Backends

AITER's `CudaCommunicator` supports 7 all-to-all backends via the `all2all_backend` field. MORI is the default:

| Backend | Class | Status | Description |
|---------|-------|--------|-------------|
| `"mori"` | `MoriAll2AllManager` | **Implemented (default)** | AMD's optimized all-to-all with IntraNode/InterNode kernels |
| `"naive"` | `NaiveAll2AllManager` | Referenced | Simple reference implementation |
| `"allgather_reducescatter"` | `AgRsAll2AllManager` | Referenced | All-gather + reduce-scatter based |
| `"pplx"` | `PPLXAll2AllManager` | Referenced | PPLX backend |
| `"deepep_high_throughput"` | `DeepEPHTAll2AllManager` | Referenced | DeepEP high-throughput variant |
| `"deepep_low_latency"` | `DeepEPLLAll2AllManager` | Referenced | DeepEP low-latency variant |
| `"flashinfer_all2allv"` | `FlashInferAllToAllManager` | Referenced | FlashInfer-based all-to-all |

Only `MoriAll2AllManager` is currently defined in `all2all.py`. The other backends are referenced in `CudaCommunicator.all2all_manager` (communicator_cuda.py) but not yet implemented. MORI is hardcoded as the default backend in `DeviceCommunicatorBase.__init__()` (line 128 of `base_device_communicator.py`).

---

## 6. Quantization Support

MORI dispatch transfers quantized tokens to minimize communication bandwidth. The tests validate two quantization modes:

| Quantization Type | `QuantType` | Description |
|-------------------|-------------|-------------|
| **Per-token FP8** | `QuantType.per_Token` | One scale per token row |
| **Per-128x128 block FP8** | `QuantType.per_128x128` | Block-wise scales (128×128 tiles) |

Token quantization before dispatch:
```python
from aiter import get_hip_quant, QuantType, dtypes

quant_func = get_hip_quant(
    quant_type if quant_type != QuantType.per_128x128 else QuantType.per_1x128
)
tokens_qt, scale = quant_func(tokens, quant_dtype=dtypes.fp8)
```

Note: For `per_128x128` quantization, the activation quantization function uses `per_1x128` (per-row blocks), while weight quantization uses the full 128×128 block scheme.

---

## 7. Usage Examples

### 7.1 Via EP Group (Recommended for Inference Engines)

This is how ATOM and similar engines use MORI — the manager is automatically created when the EP communicator is first accessed:

```python
import aiter
from aiter.dist.parallel_state import get_ep_group

# Initialize distributed environment
aiter.init_dist_env(
    tensor_model_parallel_size=1,
    rankID=0,
    backend="nccl",
    distributed_init_method="tcp://127.0.0.1:12345",
    data_parallel_size=world_size,
    data_parallel_rank=rank_id,
)

# Access EP group — all2all_manager is lazily created
ep_group = get_ep_group()
mori_manager = ep_group.device_communicator.all2all_manager

# Get handle for specific MoE layer dimensions
mori_op = mori_manager.get_handle(handle_kwargs)

# Dispatch → MoE compute → Combine
dispatch_result = mori_op.dispatch(tokens_qt, topk_weights, scale, topk_ids)
output = fused_moe(...)
result = mori_op.combine(output, topk_weights, topk_ids)

# Cleanup
aiter.destroy_dist_env()
```

### 7.2 Direct MORI API (Low-Level)

For benchmarking or standalone testing without AITER's distributed infrastructure:

```python
import torch
import mori

# Initialize MORI with the world process group
world_group = torch.distributed.group.WORLD
torch._C._distributed_c10d._register_process_group("default", world_group)
mori.shmem.shmem_torch_process_group_init("default")

# Create config and op handle
mori_config = mori.ops.EpDispatchCombineConfig(
    data_type=tokens_qt.dtype,
    rank=rank_id,
    world_size=world_size,
    hidden_dim=hidden_dim,
    scale_dim=scale.shape[-1],
    scale_type_size=scale.dtype.itemsize,
    max_token_type_size=torch.bfloat16.itemsize,
    max_num_inp_token_per_rank=max_tokens,
    num_experts_per_rank=num_experts // world_size,
    num_experts_per_token=topk,
)
mori_op = mori.ops.EpDispatchCombineOp(mori_config)

# Dispatch → MoE compute → Combine
(out, weights, scale, ids, recv_num) = mori_op.dispatch(tokens_qt, topk_weights, scale, topk_ids)
moe_output = fused_moe(...)
result = mori_op.combine(moe_output, topk_weights, topk_ids)
```

---

## 8. IntraNode vs InterNode Detection

MORI determines topology during `All2AllManagerBase.__init__()` by checking whether all ranks in the EP process group share the same memory system:

```python
# From base_device_communicator.py
from aiter.dist.parallel_state import in_the_same_node_as

self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
```

`in_the_same_node_as()` is a collective operation that tests shared memory accessibility between ranks. If any rank is on a different node, `internode=True` and MORI uses the `InterNodeV1` kernel with RDMA blocks.

---

## Decision Tree

```
Need MoE Expert Parallelism?
├── Yes
│   ├── Using AITER distributed infrastructure (init_dist_env)?
│   │   ├── Yes → Access via ep_group.device_communicator.all2all_manager (automatic)
│   │   └── No → Create MoriAll2AllManager(cpu_group) manually
│   ├── Single node?
│   │   ├── Yes → IntraNode kernel (80 blocks, shared memory, no RDMA)
│   │   └── No → InterNodeV1 kernel (32 blocks + 16 RDMA blocks)
│   └── Quantization?
│       ├── per_Token → One FP8 scale per token row
│       └── per_128x128 → Block-wise FP8 scales
└── No → Not applicable (MORI is EP-only)
```

---

## Source Files

| File | Description |
|------|-------------|
| `aiter/dist/device_communicators/all2all.py` | `MoriAll2AllManager` — MORI integration, handle creation, kernel type selection |
| `aiter/dist/device_communicators/base_device_communicator.py` | `All2AllManagerBase` — abstract base with `dispatch()`/`combine()` interface; `Cache` — WeakValueDictionary handle cache; `DeviceCommunicatorBase` — base communicator with `all2all_backend` field |
| `aiter/dist/device_communicators/communicator_cuda.py` | `CudaCommunicator` — lazy `all2all_manager` property, `dispatch()`/`combine()` delegation, all 7 backend options |
| `aiter/dist/parallel_state.py` | `get_ep_group()` — returns EP `GroupCoordinator`; `in_the_same_node_as()` — topology detection; `init_dist_env()` / `destroy_dist_env()` |

## Test Files

| File | Description | Run Command |
|------|-------------|-------------|
| `op_tests/multigpu_tests/test_mori_all2all.py` | Integration test via `MoriAll2AllManager` — tests dispatch/combine through AITER's EP group infrastructure. Tests `world_size=8, E=16, topk=2` with `per_Token` and `per_128x128` quantization. | `python op_tests/multigpu_tests/test_mori_all2all.py` |
| `op_tests/multigpu_tests/test_dispatch_combine.py` | Direct MORI API test — creates `EpDispatchCombineConfig` and `EpDispatchCombineOp` directly, includes correctness checks via `checkAllclose()` against reference EP and non-EP results. | `python op_tests/multigpu_tests/test_dispatch_combine.py` |

Both tests require 8 GPUs and use `multiprocessing.Pool` to spawn one process per GPU.
