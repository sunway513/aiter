# HSACO Kernel Launch Guide

This guide explains how to load and run pre-compiled HSACO (HSA Code Object) kernels
from AITER using the HIP runtime API. No AITER build or Python environment is needed —
just the `.co` files, `hipcc`, and a HIP-capable GPU.

## Table of Contents

1. [Overview](#1-overview)
2. [HSACO File Organization](#2-hsaco-file-organization)
3. [CSV Configuration Files](#3-csv-configuration-files)
4. [Step-by-Step: Loading and Launching a Kernel](#4-step-by-step-loading-and-launching-a-kernel)
5. [Argument Packing ABI](#5-argument-packing-abi)
6. [Complete Examples](#6-complete-examples)
7. [AITER Internal Architecture](#7-aiter-internal-architecture)
8. [Troubleshooting](#8-troubleshooting)

## 1. Overview

AITER ships pre-compiled GPU kernels as `.co` (code object / HSACO) files for each
supported architecture (`gfx942`, `gfx950`, etc.). These are standalone GPU binaries
that can be loaded and launched using three HIP runtime calls:

```cpp
hipModuleLoad(&module, "kernel.co");                   // Load the binary
hipModuleGetFunction(&func, module, "kernel_name");    // Get the entry point
hipModuleLaunchKernel(func, grid, block, args);        // Launch it
```

No CK, no Triton, no Python — just the HIP runtime.

## 2. HSACO File Organization

Pre-compiled kernels live under `hsa/` organized by GPU architecture and kernel type:

```
hsa/
├── codegen.py               # Generates C++ config headers from CSV
├── gfx942/                  # MI300X kernels
│   ├── bf16gemm/            # BF16 GEMM
│   ├── f4gemm/              # FP4 GEMM
│   ├── f8f8bbs/             # FP8 block-scale GEMM
│   ├── fmha_v3_fwd/         # Flash Attention forward
│   ├── fmha_v3_bwd/         # Flash Attention backward
│   ├── fmoe/                # Fused MoE
│   └── topksoftmax/         # Top-K softmax
└── gfx950/                  # MI355X kernels
    ├── bf16gemm/
    ├── f4gemm/
    ├── fmoe/
    ├── topksoftmax/
    └── ...
```

Each directory contains:
- **`.co` files** — the GPU binaries
- **`.csv` files** — metadata mapping kernel names to `.co` files with tuning parameters

## 3. CSV Configuration Files

Each CSV maps kernel function names to their `.co` files and configuration parameters.

### Top-K Softmax (`topksoftmax/topksoftmax.csv`)

```csv
knl_name,co_name,subm,num_experts,topk,dtype
_ZN5aiter19topksoftmax_4x128x4E,topksoftmax_4x128x4.co,4,128,4,fp32
_ZN5aiter20topksoftmax_12x128x6E,topksoftmax_12x128x6.co,12,128,6,fp32
```

| Column | Meaning |
|--------|---------|
| `knl_name` | Mangled C++ symbol name (entry point in the `.co` binary) |
| `co_name` | Filename of the `.co` binary |
| `subm` | Tile size in the token dimension (affects grid calculation) |
| `num_experts` | Number of experts this kernel variant supports |
| `topk` | Top-K value this kernel variant supports |
| `dtype` | Input data type (`fp32` or `bf16`) |

### BF16 GEMM (`bf16gemm/bf16gemm_fp32bf16.csv`)

```csv
knl_name,co_name,tn,tileM,tileN,pf,bPreshuffle,splitK,subK,bias
_ZN5aiter24bf16gemm_bf16_tn_256x256E,bf16gemm_bf16_tn_256x256.co,1,256,256,0,0,0,64,0
```

| Column | Meaning |
|--------|---------|
| `tileM`, `tileN` | Tile dimensions (grid = M/tileM x N/tileN) |
| `bPreshuffle` | Whether weights must be pre-shuffled |
| `splitK` | Whether split-K is supported (for small M) |
| `bias` | Whether kernel supports fused bias add |

### FMoE (`fmoe/silu/fmoe_fp16_blockscaleFp8_g1u1_silu.csv`)

```csv
knl_name,co_name,atm,vskip,smf,tg_num_perCU,ps,subGU_m,subGU_n
_ZN5aiter50fmoe_fp16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E,...,0,1,0,1,1,32,256
```

## 4. Step-by-Step: Loading and Launching a Kernel

### Step 1: Load the HSACO Module

```cpp
#include <hip/hip_runtime.h>

hipModule_t module;
hipError_t err = hipModuleLoad(&module, "/path/to/hsa/gfx950/topksoftmax/topksoftmax_4x128x4.co");
// Or load from memory:
// hipModuleLoadData(&module, binary_data_ptr);
```

### Step 2: Get the Kernel Function

The function name is the `knl_name` from the CSV — a mangled C++ symbol:

```cpp
hipFunction_t func;
hipModuleGetFunction(&func, module, "_ZN5aiter19topksoftmax_4x128x4E");
```

### Step 3: Pack Kernel Arguments

Arguments must be packed into a `__attribute__((packed))` struct that exactly matches
the kernel's expected ABI. See [Section 5](#5-argument-packing-abi) for the padding rules.

```cpp
struct __attribute__((packed)) KernelArgs {
    void*        ptr_T;       // output: topk_indices
    p2           _pad0;       // 8-byte padding after each pointer
    void*        ptr_W;       // output: topk_weights
    p2           _pad1;
    void*        ptr_A;       // input: gating_output
    p2           _pad2;
    unsigned int batch;       // num_tokens
    p3           _pad4;       // 12-byte padding after each scalar
    unsigned int expert;      // num_experts
    p3           _pad5;
    unsigned int topk;
    p3           _pad6;
    unsigned int renormalize; // 0 or 1
    p3           _pad7;
    unsigned int out_stride;  // stride in bytes
    p3           _pad8;
};

KernelArgs args = {};
args.ptr_T  = d_indices;
args.ptr_W  = d_weights;
args.ptr_A  = d_gating;
args.batch  = num_tokens;
args.expert = num_experts;
args.topk   = topk;
// ...
```

### Step 4: Configure Launch Parameters

```cpp
size_t arg_size = sizeof(args);
void* config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &arg_size,
    HIP_LAUNCH_PARAM_END
};
```

### Step 5: Calculate Grid Dimensions and Launch

Grid dimensions depend on the kernel type. Block size is always **256 threads**
(4 wavefronts of 64) for all AITER ASM kernels.

```cpp
// For topksoftmax: grid.x = ceil(num_tokens / SUBM)
int gdx = (num_tokens + SUBM - 1) / SUBM;

// For GEMM: grid = (ceil(N/tileN), ceil(M/tileM), splitK)
// int gdx = (N + tileN - 1) / tileN;
// int gdy = (M + tileM - 1) / tileM;
// int gdz = splitK;

hipModuleLaunchKernel(
    func,
    gdx, 1, 1,      // grid dimensions
    256, 1, 1,       // block dimensions (always 256)
    0,               // shared memory (0 — managed by kernel)
    stream,          // HIP stream (nullptr for default)
    nullptr,
    (void**)&config
);

hipDeviceSynchronize();  // Wait for completion
```

### Step 6: Cleanup

```cpp
hipModuleUnload(module);
```

## 5. Argument Packing ABI

AITER ASM kernels use a specific argument packing convention with padding structs.
**Getting this wrong produces silent garbage or GPU faults.**

### Padding Structs

```cpp
struct p3 { unsigned int _p0, _p1, _p2; };  // 12 bytes
struct p2 { unsigned int _p0, _p1; };        // 8 bytes
struct p1 { unsigned int _p0; };             // 4 bytes
```

### Rules

| Argument Type | Size | Padding After | Total Slot |
|---------------|------|---------------|------------|
| `void*` (pointer) | 8 bytes | `p2` (8 bytes) | 16 bytes |
| `unsigned int` (scalar) | 4 bytes | `p3` (12 bytes) | 16 bytes |
| `float` (scalar) | 4 bytes | `p3` (12 bytes) | 16 bytes |

Every argument occupies exactly **16 bytes** (value + padding). The struct must use
`__attribute__((packed))` to prevent compiler-inserted padding.

### Layout Visualization

```
Offset  Size  Field
0x00    8     ptr_T (pointer)
0x08    8     p2 padding
0x10    8     ptr_W (pointer)
0x18    8     p2 padding
0x20    8     ptr_A (pointer)
0x28    8     p2 padding
0x30    4     batch (uint32)
0x34    12    p3 padding
0x40    4     expert (uint32)
0x44    12    p3 padding
...     ...   (each field at 16-byte aligned offset)
```

### How to Determine Argument Order

Look at the corresponding `asm_*.cu` file in `csrc/py_itfs_cu/`. The `KernelArgs` struct
defines the exact layout. Key files:

| Kernel Type | Argument Struct Location |
|-------------|------------------------|
| Top-K Softmax | `csrc/py_itfs_cu/asm_topksoftmax.cu:10-28` |
| BF16 GEMM | `csrc/py_itfs_cu/asm_gemm_a16w16.cu:12-58` |
| FP8 GEMM | `csrc/py_itfs_cu/asm_gemm_a8w8.cu` |
| FP4 GEMM | `csrc/py_itfs_cu/asm_gemm_a4w4.cu` |
| FMoE | `csrc/py_itfs_cu/asm_fmoe.cu` |
| Flash Attention | `csrc/py_itfs_cu/asm_fmha.cu` |

## 6. Complete Examples

### Example 1: Top-K Softmax (Validated)

See [`examples/hsaco_topksoftmax.cpp`](../examples/hsaco_topksoftmax.cpp) — a standalone
program that loads `topksoftmax_4x128x4.co`, runs it on random data, and verifies against
a CPU reference. Validated on MI355X (gfx950) with perfect accuracy (0.0 max error).

```bash
# Build
hipcc -o hsaco_topksoftmax examples/hsaco_topksoftmax.cpp -std=c++17

# Run (point to the hsa/<arch> directory)
HIP_VISIBLE_DEVICES=0 ./hsaco_topksoftmax hsa/gfx950
```

Expected output:
```
=== HSACO Kernel Launch Test: topksoftmax_4x128x4 ===
GPU: AMD Instinct MI355X (arch: gfx950:sramecc+:xnack-)

Step 1: hipModuleLoad -> OK
Step 2: hipModuleGetFunction -> OK
Step 3: Allocated 32768 bytes to GPU
Step 4: KernelArgs struct size = 128 bytes
Step 5: Launching kernel  grid=(16,1,1)  block=(256,1,1)
  -> OK

=== Results ===
  Index match:  64/64 tokens
  Max weight error: 0.000000
  PASSED
```

### Example 2: BF16 GEMM (Argument Layout Reference)

The BF16 GEMM kernel uses the same pattern but with more arguments:

```cpp
struct __attribute__((packed)) GemmKernelArgs {
    void* ptr_D;             // Output matrix D [M, N]
    p2 _p0;
    void* ptr_C;             // Bias vector (optional)
    p2 _p1;
    void* ptr_A;             // Input matrix A [M, K]
    p2 _p2;
    void* ptr_B;             // Weight matrix B [K, N] or [N, K] depending on tn
    p2 _p3;
    float alpha;             // Scale factor for A*B (typically 1.0)
    p3 _p4;
    float beta;              // Scale factor for bias (typically 0.0)
    p3 _p5;
    unsigned int stride_D0;  // D row stride in elements
    p3 _p6;
    unsigned int stride_D1;  // D column stride (typically 1)
    p3 _p7;
    unsigned int stride_C0;  // Bias stride
    p3 _p8;
    unsigned int stride_C1;
    p3 _p9;
    unsigned int stride_A0;  // A row stride (= K)
    p3 _p10;
    unsigned int stride_A1;  // A column stride (= 1)
    p3 _p11;
    unsigned int stride_B0;  // B row stride (= N for TN layout)
    p3 _p12;
    unsigned int stride_B1;
    p3 _p13;
    unsigned int M;
    p3 _p14;
    unsigned int N;
    p3 _p15;
    unsigned int K;
    p3 _p16;
    unsigned int splitk;     // Split-K factor (0 or 1 = no split)
    p3 _p17;
    unsigned int is_out_b16; // 1 if output is BF16, 0 for FP32
    p3 _p18;
    void* ptr_Bias;          // Bias pointer (can be nullptr)
    p2 _p19;
    unsigned int add_bias;   // 1 to fuse bias addition
    p3 _p20;
    void* ptr_semaphore;     // Synchronization for split-K (nullptr if unused)
    p2 _p21;
};

// Grid dimensions for GEMM:
int gdx = (N + tileN - 1) / tileN;   // from CSV tileN column
int gdy = (M + tileM - 1) / tileM;   // from CSV tileM column
int gdz = splitK;                     // 1 = no split
```

## 7. AITER Internal Architecture

For reference, here is how AITER's build system and runtime connect the pieces:

```
Build Time:
  hsa/<arch>/<type>/*.csv          CSV config files
          │
          ▼
  hsa/codegen.py                   Reads CSV, generates C++ headers
          │
          ▼
  jit/build/<module>/blob/         Generated asm_*_configs.hpp
  asm_*_configs.hpp                (static unordered_map of kernel configs)

Runtime:
  Python (tuned_gemm.py)           Look up config, choose ASM vs Triton vs hipBLASLt
          │
          ▼
  C++ (asm_*.cu)                   Heuristic kernel selection from config map
          │
          ▼
  aiter_hip_common.h               AiterAsmKernel class:
    load_asm_kernel()                hipModuleLoad() + hipModuleGetFunction()
    launch_kernel()                  hipModuleLaunchKernel()
```

Key source files:

| File | Role |
|------|------|
| `csrc/include/aiter_hip_common.h` | HIP module load/launch helpers, padding structs |
| `csrc/py_itfs_cu/asm_*.cu` | Per-kernel-type argument packing and heuristic selection |
| `hsa/codegen.py` | CSV → C++ config header generator |
| `aiter/tuned_gemm.py` | Python entry point, config lookup from tuned CSV |

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `hipModuleLoad` fails | Wrong arch — loading gfx950 `.co` on gfx942 | Use `.co` files matching your GPU arch |
| `hipModuleGetFunction` fails | Wrong kernel name | Use exact `knl_name` from CSV (mangled C++ symbol) |
| Garbage output | Wrong argument struct layout | Verify struct matches `asm_*.cu` KernelArgs exactly |
| GPU fault / hang | Dimension not aligned to tile size | Check that N, K are divisible by kernel tile requirements |
| Wrong grid dimensions | Using wrong SUBM/tileM/tileN | Read tile sizes from CSV for the selected kernel |
| Split-K race condition | Missing semaphore buffer | Allocate `(gridX * gridY)` uint32 zeroed buffer |
