# FP4 (MXF4) GEMM Benchmark Guide

Step-by-step instructions for building AITER and benchmarking FP4 GEMM performance on AMD Instinct MI350X/MI355X (gfx950).

## 1. Build AITER

### From source

```bash
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
pip install pandas psutil pybind11>=3.0.1 ninja packaging
python3 setup.py develop
```

If you forgot `--recursive`:
```bash
git submodule sync && git submodule update --init --recursive
```

Build modes (via `PREBUILD_KERNELS` env var):
| Mode | Description |
|------|------------|
| `0` (default) | JIT-compile kernels on first use — fastest initial build |
| `1` | Precompile FMHA + GEMM modules — slower build, faster first run |
| `3` | Minimal (FMHA V3 only) — fastest build for quick iteration |

Example: `PREBUILD_KERNELS=1 GPU_ARCHS="gfx950" python3 setup.py develop`

### Using Docker

```bash
docker run -it --network host \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --shm-size=64g \
  rocm/pytorch:latest /bin/bash

# Inside the container:
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
pip install pandas psutil pybind11 ninja packaging
python3 setup.py develop
```

## 2. Prepare Shapes

Create a CSV with the GEMM shapes to benchmark. Format: `M,N,K` with a header row.

### Sample shapes (peak performance on MI355X)

Save as `sample_shapes.csv`:

```csv
M,N,K
16384,16384,32768
16384,8192,32768
16384,32768,16384
32768,16384,16384
32768,32768,16384
8192,32768,16384
8192,16384,32768
32768,16384,8192
65536,16384,8192
4736,44416,17920
4480,54400,36224
59520,7680,35200
48896,6144,18304
5248,25344,18816
6528,7040,5632
```

The first group (large power-of-2 shapes) achieves 5,200–5,800 TFLOPS on MI355X — approaching the ~5.8 PFLOPS theoretical peak. The second group covers real-world LLM inference shapes at 4,500–5,300 TFLOPS.

## 3. Tune Kernels

Tuning benchmarks all available kernel variants (CK-Tile + ASM) for each shape and records the best. This step is optional if you only want to run with the default configs, but is recommended for peak performance.

### Use pre-tuned configs

Pre-tuned configs for MI355X (gfx950) covering 977 FP4 shapes are available on the [`configs/mi355x-gemm-tuning`](https://github.com/sunway513/aiter/tree/configs/mi355x-gemm-tuning) branch:

```bash
# Fetch the pre-tuned config
git fetch https://github.com/sunway513/aiter.git configs/mi355x-gemm-tuning
git checkout FETCH_HEAD -- aiter/configs/a4w4_blockscale_tuned_gemm.csv
```

Or point AITER to it at runtime via environment variable:

```bash
export AITER_CONFIG_GEMM_A4W4=/path/to/a4w4_blockscale_tuned_gemm.csv
```

Multiple config files can be colon-separated: `path1.csv:path2.csv`

### Tune your own shapes

```bash
cd csrc/ck_gemm_a4w4_blockscale

python3 gemm_a4w4_blockscale_tune.py \
  -i /path/to/sample_shapes.csv \
  -o /path/to/tuned_output.csv
```

- Tests ~54 kernel variants per shape (20 CK-Tile + ~34 ASM tile configs)
- ~15 seconds per shape
- gfx950 only

After tuning, merge results into the AITER config:

```python
import pandas as pd

new = pd.read_csv('/path/to/tuned_output.csv')
existing = pd.read_csv('aiter/configs/a4w4_blockscale_tuned_gemm.csv')
merged = pd.concat([existing, new])

# Deduplicate — keep best TFLOPS per (cu_num, M, N, K)
merged = merged.sort_values('tflops', ascending=False) \
    .drop_duplicates(subset=['cu_num', 'M', 'N', 'K'], keep='first') \
    .sort_values(['cu_num', 'M', 'N', 'K'])
merged.to_csv('aiter/configs/a4w4_blockscale_tuned_gemm.csv', index=False)
```

## 4. Run Benchmark

Save as `bench_fp4.py`:

```python
#!/usr/bin/env python3
"""Benchmark AITER FP4 (MXF4) GEMM."""
import csv
import sys
import torch
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight

torch.manual_seed(42)
dtype = dtypes.bf16
device = "cuda:0"

# FP4 quantization: per-1x32 block scaling (matches ASM kernel expectations)
quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)

shapes = []
with open("sample_shapes.csv") as f:
    for row in csv.DictReader(f):
        shapes.append((int(row["M"]), int(row["N"]), int(row["K"])))

warmup, iters = 3, 10

print(f"{'M':>8s} {'N':>8s} {'K':>8s} {'us':>10s} {'TFLOPS':>10s} {'status':>8s}")
print("=" * 62)

for M, N, K in shapes:
    try:
        # Create BF16 tensors, then quantize to FP4
        x_fp = torch.randn((M, K), dtype=dtype, device=device)
        w_fp = torch.randn((N, K), dtype=dtype, device=device)

        x, x_scales = quant_func(x_fp, shuffle=True)
        w, w_scales = quant_func(w_fp, shuffle=True)
        w = shuffle_weight(w)  # Required for ASM preshuffle kernels

        del x_fp, w_fp
        torch.cuda.empty_cache()

        # Warmup
        for _ in range(warmup):
            out = aiter.gemm_a4w4(x, w, x_scales, w_scales, bpreshuffle=True)
        torch.cuda.synchronize()

        # Timed iterations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = aiter.gemm_a4w4(x, w, x_scales, w_scales, bpreshuffle=True)
        end.record()
        torch.cuda.synchronize()

        us = start.elapsed_time(end) * 1000 / iters  # microseconds
        tflops = 2 * M * N * K / us / 1e6
        print(f"{M:8d} {N:8d} {K:8d} {us:10.1f} {tflops:10.2f} {'ok':>8s}")

        del x, w, x_scales, w_scales, out
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"{M:8d} {N:8d} {K:8d} {'':>10s} {'':>10s} {'ERR':>8s}  {str(e)[:50]}")
        torch.cuda.empty_cache()

    sys.stdout.flush()
```

Run:
```bash
HIP_VISIBLE_DEVICES=0 python3 bench_fp4.py
```

### Key API notes

| Item | Detail |
|------|--------|
| **GEMM API** | `aiter.gemm_a4w4(x, w, x_scales, w_scales, bpreshuffle=True)` |
| **Quantization** | `aiter.get_triton_quant(aiter.QuantType.per_1x32)` → returns `(packed_fp4, e8m0_scales)` |
| **Weight shuffle** | `from aiter.ops.shuffle import shuffle_weight` — required for ASM kernels |
| **Constraint** | K must be divisible by 32 |
| **Backend** | ASM (hand-tuned assembly) + CK-Tile, auto-selected via tuned config |
| **Tuner** | `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` |
| **Config** | `aiter/configs/a4w4_blockscale_tuned_gemm.csv` |
| **Config env var** | `AITER_CONFIG_GEMM_A4W4` |

## 5. Reference: FP4 Performance

All results with BF16 output, tuned configs, single GPU.

### MI355X (gfx950, 256 CUs)

#### Peak shapes (large power-of-2)

| M | N | K | TFLOPS | us |
|---|---|---|--------|-----|
| 16384 | 16384 | 32768 | 5,775 | 3,047 |
| 16384 | 8192 | 32768 | 5,718 | 1,538 |
| 16384 | 32768 | 16384 | 5,594 | 3,145 |
| 32768 | 32768 | 16384 | 5,500 | 6,397 |
| 8192 | 32768 | 16384 | 5,500 | 1,599 |
| 32768 | 16384 | 16384 | 5,496 | 3,201 |
| 65536 | 16384 | 8192 | 5,226 | 3,366 |
| 8192 | 16384 | 32768 | 5,207 | 1,689 |
| 8192 | 8192 | 16384 | 5,124 | 429 |

#### Real-world LLM inference shapes

| M | N | K | TFLOPS | us |
|---|---|---|--------|-----|
| 4736 | 44416 | 17920 | 5,338 | 1,412 |
| 4480 | 54400 | 36224 | 5,113 | 3,453 |
| 59520 | 7680 | 35200 | 4,930 | 6,527 |
| 48896 | 6144 | 18304 | 4,820 | 2,282 |
| 5248 | 25344 | 18816 | 4,810 | 1,041 |
| 6528 | 7040 | 5632 | 4,654 | 111 |
| 16896 | 31104 | 7168 | 4,450 | 1,693 |
| 9984 | 15360 | 13824 | 4,326 | 980 |
| 12416 | 8960 | 15360 | 4,341 | 787 |

### MI350X (gfx950, 256 CUs)

Tested from `rocm/pytorch:latest` (ROCm 7.2), AITER built from source with `PREBUILD_KERNELS=0`.

#### Peak shapes (large power-of-2, tuned)

| M | N | K | TFLOPS | us |
|---|---|---|--------|-----|
| 16384 | 16384 | 32768 | 5,089 | 3,457 |
| 16384 | 8192 | 32768 | 5,094 | 1,727 |
| 16384 | 32768 | 16384 | 4,919 | 3,576 |
| 8192 | 16384 | 32768 | 4,752 | 1,851 |
| 16384 | 16384 | 16384 | 4,770 | 1,844 |
| 16384 | 8192 | 16384 | 4,840 | 909 |
| 8192 | 16384 | 16384 | 4,700 | 936 |
| 8192 | 32768 | 16384 | 4,695 | 1,874 |
| 8192 | 8192 | 16384 | 4,647 | 473 |
| 32768 | 8192 | 16384 | 4,553 | 1,932 |
| 32768 | 32768 | 16384 | 4,500 | 7,819 |
| 32768 | 16384 | 8192 | 4,378 | 2,009 |
| 32768 | 16384 | 16384 | 4,364 | 4,031 |
| 65536 | 16384 | 8192 | 4,246 | 4,143 |
| 32768 | 32768 | 8192 | 4,143 | 4,247 |

#### Real-world LLM inference shapes (tuned)

| M | N | K | TFLOPS | us |
|---|---|---|--------|-----|
| 12416 | 8960 | 15360 | 4,620 | 740 |
| 9984 | 15360 | 13824 | 4,575 | 927 |
| 4736 | 44416 | 17920 | 4,575 | 1,648 |
| 16896 | 31104 | 7168 | 4,469 | 1,686 |
| 48896 | 6144 | 18304 | 4,406 | 2,496 |
| 5248 | 25344 | 18816 | 4,331 | 1,156 |
| 4480 | 54400 | 36224 | 4,187 | 4,217 |
| 59520 | 7680 | 35200 | 4,142 | 7,769 |
| 6528 | 7040 | 5632 | 3,941 | 131 |
