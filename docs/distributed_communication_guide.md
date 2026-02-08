# AITER Distributed Communication Guide

This guide documents the distributed communication infrastructure in AITER, covering process group management, all-reduce backends, all-to-all for Expert Parallelism (MORI), shared memory broadcasting, and Triton-based communication primitives.

---

## Quick Reference

| Use Case | Recommended API | Backend |
|----------|----------------|---------|
| **Initialize distributed environment** | `aiter.init_dist_env(tp_size, rank, ...)` | NCCL + Gloo |
| **All-reduce (Tensor Parallelism)** | `tensor_model_parallel_all_reduce(input)` | Quick Reduce > Custom AR > PyNCCL |
| **Fused all-reduce + RMSNorm** | `tensor_model_parallel_fused_allreduce_rmsnorm(...)` | Custom AR or split kernel fallback |
| **All-gather (TP)** | `tensor_model_parallel_all_gather(input)` | Custom AR or PyNCCL |
| **Reduce-scatter (TP)** | `tensor_model_parallel_reduce_scatter(input)` | Custom AR or PyNCCL |
| **MoE all-to-all (Expert Parallelism)** | `ep_group.device_communicator.dispatch(...)` | [MORI](mori_all2all_guide.md) |
| **Shared memory broadcast** | `MessageQueue.create_from_process_group(pg, ...)` | Shared memory + ZMQ |
| **Triton-based reduce-scatter/all-gather** | `IrisCommContext` + `reduce_scatter`/`all_gather` | [Iris](triton_comms.md) |

---

## 1. Architecture Overview

### Communication Stack

```
┌──────────────────────────────────────────────────────────┐
│  High-Level API                                          │
│  aiter.init_dist_env() / aiter.destroy_dist_env()        │
│  tensor_model_parallel_all_reduce()                      │
│  tensor_model_parallel_fused_allreduce_rmsnorm()         │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│  GroupCoordinator         (parallel_state.py)             │
│  Manages TP/PP/DP/EP process groups                      │
│  Owns CudaCommunicator per group (when world_size > 1)   │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────┐
│  CudaCommunicator         (communicator_cuda.py)          │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────────────┐ │
│  │ Quick Reduce │ │ Custom AR    │ │ PyNCCL            │ │
│  │ (ROCm MI3*) │ │ (IPC-based)  │ │ (RCCL ctypes)     │ │
│  └──────┬──────┘ └──────┬───────┘ └────────┬──────────┘ │
│         │               │                   │            │
│         ▼               ▼                   ▼            │
│  ┌─────────────────────────────────────────────────┐     │
│  │ all_reduce priority: QR → CA → PyNCCL → PyTorch │     │
│  └─────────────────────────────────────────────────┘     │
│                                                          │
│  ┌─────────────────────────────────────┐                 │
│  │ all2all_manager (lazy, EP only)     │                 │
│  │ └── MoriAll2AllManager (default)    │                 │
│  │     dispatch() / combine()          │                 │
│  └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────┘
```

### All-Reduce Backend Selection

`CudaCommunicator.all_reduce()` tries backends in priority order:

1. **Quick Reduce** — If enabled via `AITER_QUICK_REDUCE_QUANTIZATION` env var, ROCm MI300/MI350 only, intra-node. Uses FP8/INT6/INT4 quantization for bandwidth reduction.
2. **Custom All-Reduce** — IPC-based shared memory all-reduce. Intra-node only, world sizes 2/4/6/8. Supports CUDA graph capture.
3. **PyNCCL** — Pure Python RCCL wrapper via ctypes. Works in all topologies. CUDA graph compatible (avoids problematic CUDA APIs).
4. **PyTorch distributed** — Standard `torch.distributed.all_reduce()` fallback.

---

## 2. Initialization and Process Groups

### `init_dist_env()`

The main entry point for distributed setup. Creates TP, PP, DP, and EP process groups:

```python
import aiter

aiter.init_dist_env(
    tensor_model_parallel_size=8,   # TP degree
    rankID=0,                        # this process's TP rank
    backend="cpu:gloo,cuda:nccl",    # default: Gloo for CPU, NCCL for GPU
    distributed_init_method="env://",# or "tcp://host:port"
    local_rank=-1,                   # auto-detected if -1
    data_parallel_size=1,            # DP degree
    data_parallel_rank=0,            # this process's DP rank
)
```

Internally calls:
1. `init_distributed_environment()` — Creates the PyTorch distributed process group
2. `ensure_model_parallel_initialized()` → `initialize_model_parallel()` — Creates TP/PP/DP/EP `GroupCoordinator` objects

### Process Group Layout

Groups are formed from a 4D rank tensor with layout: `ExternalDP × DP × PP × TP`

```python
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size, tensor_model_parallel_size
)
```

| Group | Formation | Example (DP=2, TP=4) |
|-------|-----------|---------------------|
| **TP** | `all_ranks.view(-1, tp_size)` | [0,1,2,3], [4,5,6,7] |
| **PP** | `all_ranks.transpose(2,3).reshape(-1, pp_size)` | Single PP group per TP slice |
| **DP** | `all_ranks.transpose(1,3).reshape(-1, dp_size)` | [0,4], [1,5], [2,6], [3,7] |
| **EP** | `all_ranks.transpose(1,2).reshape(-1, dp_size*tp_size)` | [0,1,2,3,4,5,6,7] |

Each group gets a `GroupCoordinator` with:
- `cpu_group` (Gloo backend) — for metadata exchange, IPC handle sharing
- `device_group` (NCCL backend) — for GPU tensor communication
- `device_communicator` (`CudaCommunicator`) — custom allreduce, quick reduce, all2all

### `GroupCoordinator`

Central class managing a process group's communication:

```python
class GroupCoordinator:
    rank: int              # global rank
    ranks: List[int]       # all global ranks in this group
    world_size: int        # group size
    local_rank: int        # local device index
    rank_in_group: int     # rank within the group
    cpu_group: ProcessGroup
    device_group: ProcessGroup
    device_communicator: CudaCommunicator | None  # only when world_size > 1
    mq_broadcaster: MessageQueue | None           # only for TP group
```

Key methods:
- `all_reduce(input)` — delegates to `device_communicator.all_reduce()`
- `all_gather(input, dim)` — standard or custom all-gather
- `reduce_scatter(input, dim)` — standard or custom reduce-scatter
- `fused_allreduce_rmsnorm(input, residual, weight, eps)` — fused AR+RMSNorm
- `graph_capture()` — context manager for CUDA graph capturing with custom AR buffer registration
- `broadcast_tensor_dict(tensor_dict, src)` — broadcast mixed tensor/metadata dicts

### Cleanup

```python
aiter.destroy_dist_env()
# Internally: destroy_model_parallel() + destroy_distributed_environment()
```

---

## 3. Quick Reduce

AMD ROCm-optimized all-reduce that uses quantization to reduce communication bandwidth. Designed as a complement to Custom All-Reduce for large tensor sizes.

### Requirements

- ROCm MI300 series (gfx94*) or MI350 series (gfx95*)
- Intra-node only (all ranks on same node)
- World sizes: 2, 4, 8
- Input dtypes: `float16`, `bfloat16`
- Must set `AITER_QUICK_REDUCE_QUANTIZATION` environment variable

### Quantization Regimes

| Regime | `AITER_QUICK_REDUCE_QUANTIZATION` | Description |
|--------|-----------------------------------|-------------|
| FP | `"FP"` | No quantization, full precision |
| FP8 | `"FP8"` | FP8 quantized communication |
| INT6 | `"INT6"` | 6-bit integer quantization |
| INT4 | `"INT4"` | 4-bit integer quantization |
| NONE | `"NONE"` (default) | Disabled |

### Size Thresholds

Quick Reduce only activates for tensors within a specific size range. Minimum sizes (in MB) by dtype, world size, and regime \[FP, FP8, INT6, INT4\]:

| Config | FP | FP8 | INT6 | INT4 |
|--------|-----|------|------|------|
| float16, ws=2 | 1 | 2 | 2 | 1 |
| float16, ws=4 | 1 | 16 | 4 | 2 |
| float16, ws=8 | 16 | 4 | 4 | 8 |
| bfloat16, ws=2 | 2 | 8 | 8 | 8 |
| bfloat16, ws=4 | 8 | 64 | 64 | 16 |
| bfloat16, ws=8 | 16 | 2048 | 2048 | 2048 |

Maximum size defaults to 2 GB (configurable via `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITER_QUICK_REDUCE_QUANTIZATION` | `"NONE"` | Quantization regime: FP, FP8, INT6, INT4, NONE |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `0` (= 2 GB) | Max allreduce size in MB |
| `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` | `1` | Cast bfloat16 to float16 before reduce (faster kernels) |

---

## 4. Custom All-Reduce

IPC-based shared memory all-reduce for intra-node tensor parallelism. Uses HIP IPC handles for direct GPU-to-GPU memory access without NCCL overhead.

### Requirements

- Intra-node only (all ranks on same node)
- World sizes: 2, 4, 6, 8
- Input size must be a multiple of 16 bytes
- Max allreduce size: 128 MB total buffer (effective limit 64 MB per allreduce due to 2-stage write mode)

### Features

- **CUDA Graph compatible** — Registers IPC buffers for graph capture/replay via `capture()` context manager
- **Fused All-Reduce + RMSNorm** — `custom_fused_ar_rms()` combines all-reduce with RMSNorm in a single kernel pass. Falls back to separate all-reduce + RMSNorm if input is too large.
- **Out-of-place** — Input is copied to pre-registered IPC buffer, reduced, output written to separate buffer
- **FP8 quantization** — Optional FP8 quantized communication via `open_fp8_quant` parameter

### Fused All-Reduce + RMSNorm Eligibility

```python
can_use_fuse_ar_rms = (
    hidden_dim <= 16384 and
    total_bytes < 8 * 1024 * 8192 and
    world_size != 6
)
```

When ineligible, falls back to: all-reduce → `rmsnorm2d_fwd_with_add()`.

### Key C++ Operations

All backed by HIP C++ kernels exposed via `aiter`:
- `aiter.allocate_meta_buffer()` / `aiter.get_meta_buffer_ipc_handle()` — IPC buffer allocation
- `aiter.init_custom_ar()` — Initialize all-reduce with IPC handles
- `aiter.all_reduce()` — Perform all-reduce
- `aiter.fused_allreduce_rmsnorm()` — Fused AR + RMSNorm
- `aiter.register_input_buffer()` / `aiter.register_output_buffer()` — Register IPC buffers
- `aiter.register_graph_buffers()` — Register graph capture buffers

---

## 5. PyNCCL Communicator

Pure Python wrapper for RCCL (AMD's NCCL) via ctypes. Provides CUDA graph-safe communication without PyTorch's NCCL backend limitations.

### Supported Operations

| Operation | Method | Notes |
|-----------|--------|-------|
| All-reduce | `all_reduce(in_tensor, out_tensor, op)` | Out-of-place, SUM default |
| All-gather | `all_gather(output, input)` | Fixed-size per rank |
| All-gatherv | `all_gatherv(output, input, sizes)` | Variable-size per rank |
| Reduce-scatter | `reduce_scatter(output, input, op)` | Fixed-size per rank |
| Reduce-scatterv | `reduce_scatterv(output, input, sizes, op)` | Variable-size per rank |
| Send | `send(tensor, dst)` | Point-to-point |
| Recv | `recv(tensor, src)` | Point-to-point |
| Broadcast | `broadcast(tensor, src)` | From source rank |
| Group ops | `group_start()` / `group_end()` | Batch multiple NCCL calls |

### Key Design Choices

- **Attached to non-NCCL group** — The PyNCCL communicator is initialized from a Gloo CPU group, not from PyTorch's NCCL group, to avoid conflicts
- **RCCL library loading** — Uses `NCCLLibrary` wrapper from `pynccl_wrapper.py` with ctypes bindings
- **Warmup** — Performs a small all-reduce during init to verify connectivity
- **Window registration** — `register_comm_window()` for advanced IPC memory registration

---

## 6. MoE All-to-All (MORI)

MORI is AMD's optimized all-to-all communication library for Expert Parallelism. It handles the dispatch (scatter tokens to expert GPUs) and combine (gather results back) phases of MoE computation.

See the dedicated [MORI All-to-All Guide](mori_all2all_guide.md) for full details covering:
- Dispatch/combine protocol and tensor shapes
- IntraNode vs InterNode kernel selection
- Handle caching and configuration
- EP group formation
- Quantization support (per_Token, per_128x128)
- Integration examples

### Integration with Downstream Projects

MORI is used widely beyond AITER, including:
- **ATOM** — AMD's inference engine, via `MoriPrepareAndFinalize` in `atom/model_ops/fused_moe/`
- **vLLM** — Upstream vLLM uses MORI as an all-to-all backend for Expert Parallelism
- **SGLang** — SGLang integrates MORI for MoE model serving

---

## 7. Shared Memory Broadcast (MessageQueue)

ZMQ + shared memory ring buffer for low-latency intra-node broadcasting. Used by the TP group to broadcast scheduling decisions from rank 0 to other ranks.

### Architecture

```
Writer (rank 0)
    ├── ShmRingBuffer → Local readers (same node, shared memory)
    └── ZMQ XPUB     → Remote readers (cross-node, TCP)
```

### ShmRingBuffer

Lock-free ring buffer using POSIX shared memory with per-chunk metadata flags:

```
Buffer layout:
+-------------------------------+------------------------------------------+
| chunk0 | chunk1 | ... | chunkN | metadata0 | metadata1 | ... | metadataN |
+-------------------------------+------------------------------------------+
| max_chunks × max_chunk_bytes  | max_chunks × (1 + n_reader) bytes        |

Per-chunk metadata: [written_flag, reader0_flag, reader1_flag, ..., readerN_flag]
```

State machine:
- `0???...???` — Not written, can write
- `1000...000` — Written, can read (no reader has read)
- `1???...???` — Written, partially read
- `1111...111` — All readers done, can overwrite

### Usage

The TP group automatically creates a `MessageQueue` broadcaster when `use_message_queue_broadcaster=True`:

```python
# Automatically created during init for TP group
mq = MessageQueue.create_from_process_group(pg, max_chunk_bytes=4*1024*1024, max_chunks=6)

# Writer side
mq.enqueue(obj)

# Reader side
obj = mq.dequeue(timeout=30.0)
```

For large objects exceeding `max_chunk_bytes`, the system automatically overflows to ZMQ pub-sub socket.

---

## 8. Triton-Based Communication (Iris)

GPU-initiated communication using the [Iris library](https://github.com/ROCm/iris). Provides Triton-based reduce-scatter and all-gather primitives.

See the [Triton Comms Guide](triton_comms.md) for installation and usage. Key points:

```python
from aiter import IrisCommContext, reduce_scatter, all_gather, calculate_heap_size

heap_size = calculate_heap_size(M=8192, N=7168, dtype=torch.float32,
                                 world_size=2, quant_mode="fp8_per_token")

with IrisCommContext(heap_size=heap_size) as ctx:
    output = reduce_scatter(input_tensor, ctx)
    result = all_gather(output, ctx)
```

---

## 9. Fused Communication + Compute Kernels

AITER provides fused kernels that combine communication with computation:

### Fused All-Reduce + RMSNorm

Available through both Custom AR and ASM backends:

```python
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm

# Via high-level API
output, residual = tensor_model_parallel_fused_allreduce_rmsnorm(
    input_, residual_inp_, weight_, eps
)
```

### ASM All-Reduce Variants

Low-level ASM all-reduce + RMSNorm + quantization fusions (from `communication.py`):

```python
from aiter.ops.communication import all_reduce_rmsnorm, all_reduce_rmsnorm_quant

# All-reduce + RMSNorm (fused, ASM)
result = all_reduce_rmsnorm(input, residual_in, weight, bias, epsilon)

# All-reduce + RMSNorm + FP8 quantization (fused, ASM)
result = all_reduce_rmsnorm_quant(input, residual_in, xscale, weight, bias, epsilon)
```

These require Custom All-Reduce to be initialized (`ca_comm` must exist on the TP group).

---

## 10. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AITER_QUICK_REDUCE_QUANTIZATION` | `"NONE"` | Quick Reduce quantization regime (FP, FP8, INT6, INT4, NONE) |
| `AITER_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | `0` (2 GB) | Quick Reduce max allreduce size in MB |
| `AITER_QUICK_REDUCE_CAST_BF16_TO_FP16` | `1` | Cast bfloat16 inputs to float16 for Quick Reduce |
| `MASTER_ADDR` | — | PyTorch distributed master address |
| `MASTER_PORT` | — | PyTorch distributed master port |
| `HIP_VISIBLE_DEVICES` | auto | GPU visibility (set automatically by `init_distributed_environment` if not present) |

---

## Decision Tree

```
What communication do you need?
├── Tensor Parallelism (all-reduce after linear layers)
│   ├── MI300/MI350, large tensors, env var set?
│   │   └── Yes → Quick Reduce (quantized, bandwidth-efficient)
│   ├── Intra-node, world_size ∈ {2,4,6,8}, size ≤ 64MB?
│   │   └── Yes → Custom All-Reduce (IPC-based, CUDA graph compatible)
│   └── Fallback → PyNCCL → PyTorch distributed
│
├── Fused all-reduce + RMSNorm?
│   ├── hidden_dim ≤ 16384 and total < 64MB and ws ≠ 6?
│   │   └── Yes → Custom AR fused kernel (single pass)
│   └── No → Separate all-reduce + RMSNorm kernel
│
├── Expert Parallelism (MoE token routing)
│   └── MORI dispatch/combine (see MORI guide)
│
├── Intra-node metadata broadcast (scheduling, config)
│   └── MessageQueue (shared memory ring buffer + ZMQ)
│
└── Triton-based communication (fused with compute)
    └── Iris library (reduce-scatter, all-gather)
```

---

## Source Files

| File | Description |
|------|-------------|
| `aiter/ops/communication.py` | `init_dist_env()`, `destroy_dist_env()`, ASM all-reduce variants |
| `aiter/dist/parallel_state.py` | `GroupCoordinator`, `initialize_model_parallel()`, TP/PP/DP/EP group formation, `in_the_same_node_as()` |
| `aiter/dist/communication_op.py` | High-level wrappers: `tensor_model_parallel_all_reduce()`, `tensor_model_parallel_fused_allreduce_rmsnorm()`, etc. |
| `aiter/dist/device_communicators/communicator_cuda.py` | `CudaCommunicator` — backend priority dispatch, lazy all2all_manager |
| `aiter/dist/device_communicators/custom_all_reduce.py` | `CustomAllreduce` — IPC-based intra-node all-reduce with CUDA graph support |
| `aiter/dist/device_communicators/quick_all_reduce.py` | `QuickAllReduce` — ROCm quantized all-reduce (FP8/INT6/INT4) |
| `aiter/dist/device_communicators/communicator_pynccl.py` | `PyNcclCommunicator` — RCCL ctypes wrapper |
| `aiter/dist/device_communicators/pynccl_wrapper.py` | `NCCLLibrary` — Low-level NCCL/RCCL ctypes bindings |
| `aiter/dist/device_communicators/all2all.py` | `MoriAll2AllManager` — MORI all-to-all integration |
| `aiter/dist/device_communicators/base_device_communicator.py` | `DeviceCommunicatorBase`, `All2AllManagerBase`, `Cache` |
| `aiter/dist/shm_broadcast.py` | `ShmRingBuffer`, `MessageQueue` — shared memory broadcast |
| `aiter/dist/utils.py` | Networking utilities, distributed helpers |

## Test Files

| File | Description |
|------|-------------|
| `op_tests/multigpu_tests/test_custom_allreduce.py` | Custom all-reduce correctness and performance |
| `op_tests/multigpu_tests/test_custom_allreduce_fp8.py` | Custom all-reduce with FP8 quantization |
| `op_tests/multigpu_tests/test_quick_all_reduce.py` | Quick Reduce correctness validation |
| `op_tests/multigpu_tests/test_fused_ar_rms.py` | Fused all-reduce + RMSNorm tests |
| `op_tests/multigpu_tests/test_allgather.py` | All-gather collective test |
| `op_tests/multigpu_tests/test_reduce_scatter.py` | Reduce-scatter collective test |
| `op_tests/multigpu_tests/test_communication.py` | General communication operations |
| `op_tests/multigpu_tests/test_collective_profile.py` | Communication performance profiling |
| `op_tests/multigpu_tests/test_mori_all2all.py` | MORI all-to-all via MoriAll2AllManager |
| `op_tests/multigpu_tests/test_dispatch_combine.py` | MORI dispatch/combine direct API test |
| `op_tests/multigpu_tests/triton_test/test_reduce_scatter_all_gather.py` | Triton-based RS + AG |
| `op_tests/multigpu_tests/triton_test/test_fused_rs_rmsnorm_quant_ag.py` | Fused RS + RMSNorm + Quant + AG |
