---
name: gemm-tuning
description: Run AITER FP4 GEMM tuning and benchmarking on MI355X nodes. Handles shape splitting across nodes, Docker orchestration, ASM/CK-Tile/Triton/Gluon kernel profiling, and result analysis.
argument-hint: "<action> [options], e.g. 'tune shapes.csv', 'bench all', 'analyze', 'status'"
---

# AITER FP4 GEMM Tuning & Benchmarking

End-to-end workflow for tuning AITER FP4 GEMM kernels on MI355X nodes, benchmarking all backends, and analyzing results.

## Step 1: Parse Action from `$ARGUMENTS`

| Action | Description |
|--------|-------------|
| `tune <shapes.csv>` | Run AITER FP4 GEMM tuner across available MI355X nodes |
| `bench <shapes.csv>` | Benchmark Triton + Gluon backends (CK-Tile/ASM come from profile CSV) |
| `analyze [results-dir]` | Build 4-way comparison table from existing results |
| `status` | Check running tuning/benchmark jobs on remote nodes |

If `$ARGUMENTS` is empty or unrecognized, print this action table and ask which action to run.

---

## Step 2: Infrastructure Reference

### Nodes
- **Node 1**: `uswslocpm2m-106-881.amd.com`
- **Node 2**: `uswslocpm2m-106-1236.amd.com`

### Docker Image
`rocm/atom:rocm_7.2_preview_gfx950_latest`

### Paths

| Location | Host Path | Docker Path |
|----------|-----------|-------------|
| Scripts mount | `/mnt/m2m_nobackup/pensun/bench_serving/` | `/bench` |
| Results mount | `/mnt/m2m_nobackup/pensun/results/fp4_tune/` | `/results` |
| AITER (Docker) | — | `/root/aiter/` or `/app/aiter-test/` |
| AITER (local) | `/home/pensun/aiter/` | — |
| Local results | `/home/pensun/results/fp4_tune/` | — |
| Tuner script | — | `/root/aiter/csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py` |
| Existing tuned DB | `/home/pensun/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv` | — |

### Docker Run Template

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host --shm-size=16G \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/m2m_nobackup/pensun/bench_serving:/bench \
  -v /mnt/m2m_nobackup/pensun/results/fp4_tune:/results \
  -e UNTUNE_FILE=/bench/<shapes_file>.csv \
  rocm/atom:rocm_7.2_preview_gfx950_latest \
  bash /bench/run_fp4_tune.sh
```

---

## Step 3: Tuning Workflow (`tune` action)

### 3.1 Read and validate input shapes CSV

The input file must have header `M,N,K` followed by rows of integer triples:

```csv
M,N,K
16384,16384,32768
4736,44416,17920
```

Read the file specified in `$ARGUMENTS`. If it doesn't exist, ask the user for the path.

### 3.2 Check already-tuned shapes

Read the existing tuned DB at `aiter/configs/a4w4_blockscale_tuned_gemm.csv` to identify shapes that are already tuned. Report which shapes are new vs already-tuned. Ask user whether to retune all or only new shapes.

### 3.3 Check node availability

For each node, run:

```bash
ssh <node> 'docker ps --format "{{.Image}} {{.Status}}"'
```

Report which nodes are idle (no containers running) vs busy.

### 3.4 Split shapes across available nodes

Split the input shapes roughly evenly across idle nodes. Write per-node shape CSVs to `/tmp/fp4_tune_shapes_n<X>.csv`.

### 3.5 Create the tuning script (`/tmp/run_fp4_tune.sh`)

Generate a script with **all three critical fixes** baked in:

```bash
#!/bin/bash
set -e
echo "=== FP4 GEMM Tuning ==="
echo "Input: $UNTUNE_FILE"

# Detect AITER directory
if [ -d /root/aiter ]; then
    AITER_DIR=/root/aiter
elif [ -d /app/aiter-test ]; then
    AITER_DIR=/app/aiter-test
else
    echo "ERROR: Cannot find AITER directory"
    exit 1
fi
echo "Using AITER at: $AITER_DIR"

TUNE_DIR="$AITER_DIR/csrc/ck_gemm_a4w4_blockscale"
HSA_DIR="$AITER_DIR/hsa"
BASENAME=$(basename "$UNTUNE_FILE" .csv)
TUNE_FILE="/results/fp4_tuned_${BASENAME}.csv"
PROFILE_FILE="/results/fp4_profile_${BASENAME}.csv"

# --- FIX 1: ASM kernel symlink ---
# The tuner's get_asm_dir() returns {hsa}/ but ASM kernels live under {hsa}/gfx950/
if [ -d "$HSA_DIR/gfx950/f4gemm" ] && [ ! -e "$HSA_DIR/f4gemm" ]; then
    ln -sf gfx950/f4gemm "$HSA_DIR/f4gemm"
    echo "Created symlink: $HSA_DIR/f4gemm -> gfx950/f4gemm"
fi
# Also check installed location
for inst_hsa in /app/aiter-test/hsa /root/aiter/hsa; do
    if [ -d "$inst_hsa/gfx950/f4gemm" ] && [ ! -e "$inst_hsa/f4gemm" ]; then
        ln -sf gfx950/f4gemm "$inst_hsa/f4gemm" 2>/dev/null || true
        echo "Created symlink: $inst_hsa/f4gemm -> gfx950/f4gemm"
    fi
done

# --- FIX 2: Column name case (tile_M -> tile_m, tile_N -> tile_n) ---
for csv_file in "$HSA_DIR"/f4gemm/f4gemm_bf16_per1x32Fp4.csv "$HSA_DIR"/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv; do
    if [ -f "$csv_file" ]; then
        if head -1 "$csv_file" | grep -q "tile_M"; then
            sed -i '1s/tile_M/tile_m/g; 1s/tile_N/tile_n/g' "$csv_file"
            echo "Fixed column names in $csv_file"
        fi
    fi
done

# --- FIX 3: K-alignment patch for run_torch() ---
cd "$TUNE_DIR"
TUNE_PY="gemm_a4w4_blockscale_tune.py"
if grep -q "x_f32 = x_f32 \* x_scales_f32" "$TUNE_PY" 2>/dev/null; then
    echo "Patching run_torch() for K-alignment compatibility..."
    sed -i 's/    x_f32 = x_f32 \* x_scales_f32/    _k = min(x_f32.shape[1], x_scales_f32.shape[1]); x_f32 = x_f32[:, :_k] * x_scales_f32[:, :_k]/' "$TUNE_PY"
    sed -i 's/    w_f32 = w_f32 \* w_scales_f32/    _k = min(w_f32.shape[1], w_scales_f32.shape[1]); w_f32 = w_f32[:, :_k] * w_scales_f32[:, :_k]/' "$TUNE_PY"
    echo "Patched run_torch()"
fi

# --- Run tuner ---
echo "Starting tuning at $(date)"
python3 gemm_a4w4_blockscale_tune.py \
    -i "$UNTUNE_FILE" \
    -o "$TUNE_FILE" \
    -o2 "$PROFILE_FILE" \
    --mp 8 \
    --sort \
    --all \
    --batch 15 \
    -v 2>&1

echo "=== TUNING COMPLETE at $(date) ==="
echo "Results: $TUNE_FILE"
echo "Profile: $PROFILE_FILE"
```

### 3.6 Create per-node Docker launcher scripts

For each node N, generate `/tmp/launch_fp4_tune_n<N>.sh`:

```bash
#!/bin/bash
nohup docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host --shm-size=16G \
  --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /mnt/m2m_nobackup/pensun/bench_serving:/bench \
  -v /mnt/m2m_nobackup/pensun/results/fp4_tune:/results \
  -e UNTUNE_FILE=/bench/fp4_tune_shapes_n<N>.csv \
  rocm/atom:rocm_7.2_preview_gfx950_latest \
  bash /bench/run_fp4_tune.sh \
  > /mnt/m2m_nobackup/pensun/results/fp4_tune/run_n<N>.log 2>&1 &

echo "Node <N> FP4 tuning launched, PID=$!"
```

**IMPORTANT**: Use separate launcher `.sh` files, never inline complex Docker commands via SSH. Inline commands cause shell escaping issues (`Ambiguous output redirect`, `Illegal variable name` with `$!`).

### 3.7 Deploy scripts to nodes

```bash
# Deploy to Node 1
scp /tmp/run_fp4_tune.sh /tmp/fp4_tune_shapes_n1.csv /tmp/launch_fp4_tune_n1.sh \
    uswslocpm2m-106-881.amd.com:/mnt/m2m_nobackup/pensun/bench_serving/

# Deploy to Node 2 (ensure dirs exist first)
ssh uswslocpm2m-106-1236.amd.com 'mkdir -p /mnt/m2m_nobackup/pensun/results/fp4_tune /mnt/m2m_nobackup/pensun/bench_serving'
scp /tmp/run_fp4_tune.sh /tmp/fp4_tune_shapes_n2.csv /tmp/launch_fp4_tune_n2.sh \
    uswslocpm2m-106-1236.amd.com:/mnt/m2m_nobackup/pensun/bench_serving/
```

### 3.8 Launch tuning

```bash
# Node 1
ssh uswslocpm2m-106-881.amd.com 'bash /mnt/m2m_nobackup/pensun/bench_serving/launch_fp4_tune_n1.sh'

# Node 2
ssh uswslocpm2m-106-1236.amd.com 'bash /mnt/m2m_nobackup/pensun/bench_serving/launch_fp4_tune_n2.sh'
```

### 3.9 Monitor progress

Tail the log files remotely:

```bash
ssh <node> 'tail -50 /mnt/m2m_nobackup/pensun/results/fp4_tune/run_n<N>.log'
```

Look for:
- `=== TUNING COMPLETE ===` — success
- `ERROR` or `RuntimeError` — failure (check which fix was missed)
- `This GEMM is not supported!` — CK-Tile doesn't support that shape (ASM may still work, this is expected for some irregular shapes)
- Shape progress lines showing `M=... N=... K=...`

---

## Step 4: Benchmark Workflow (`bench` action)

### 4.1 Create benchmark script (`/tmp/bench_fp4_triton_gluon.py`)

Generate a Python script that benchmarks Triton and Gluon on the target shapes:

```python
#!/usr/bin/env python3
"""Benchmark FP4 GEMM: Triton vs Gluon."""
import sys, os, csv, importlib.util
import torch
import triton

RESULT_FILE = os.environ.get("RESULT_FILE", "/results/fp4_triton_gluon_bench.csv")

# Read shapes from env or default
shapes_file = os.environ.get("SHAPES_FILE")
if shapes_file:
    SHAPES = []
    with open(shapes_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            SHAPES.append((int(row["M"]), int(row["N"]), int(row["K"])))
else:
    # Fallback hardcoded shapes (the 15-shape FP4 benchmark suite)
    SHAPES = [
        (16384, 16384, 32768), (16384, 8192, 32768), (16384, 32768, 16384),
        (32768, 16384, 16384), (32768, 32768, 16384), (8192, 32768, 16384),
        (8192, 16384, 32768), (32768, 16384, 8192), (65536, 16384, 8192),
        (4736, 44416, 17920), (4480, 54400, 36224), (59520, 7680, 35200),
        (48896, 6144, 18304), (5248, 25344, 18816), (6528, 7040, 5632),
    ]

# Load Triton backend
triton_fn = None
try:
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4 as triton_fn
except ImportError:
    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4 as triton_fn
    except ImportError:
        print("WARNING: Triton FP4 GEMM not available")

# Load Gluon backend
gluon_fn = None
try:
    from aiter.ops.triton.gluon.gemm_afp4wfp4 import gemm_afp4wfp4 as gluon_fn
except ImportError:
    # Fallback: load from copied file
    if os.path.exists("/bench/gluon_gemm_afp4wfp4.py"):
        spec = importlib.util.spec_from_file_location(
            "gluon_gemm_afp4wfp4", "/bench/gluon_gemm_afp4wfp4.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        gluon_fn = mod.gemm_afp4wfp4
    else:
        print("WARNING: Gluon FP4 GEMM not available")

# Generate inputs via AITER test infra
sys.path.insert(0, "/root/aiter")
from op_tests.triton_tests.gemm.basic.test_gemm_afp4wfp4 import generate_gemm_afp4wfp4_inputs

def bench_one(M, N, K, impl_fn):
    x, _, w, _, _, x_scale, w_scale, _, y = generate_gemm_afp4wfp4_inputs(
        M, N, K, torch.bfloat16, layout="TN", output=True,
        shuffle_scales_fg=False, shuffle_weight_fg=False)
    ms = triton.testing.do_bench(
        lambda: impl_fn(x, w, x_scale, w_scale, torch.bfloat16, y),
        warmup=25, rep=100)
    tflops = 2.0 * M * N * K / ms * 1e-9
    return tflops, ms * 1000.0  # tflops, microseconds

with open(RESULT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["M", "N", "K", "triton_tflops", "triton_us", "gluon_tflops", "gluon_us"])
    for M, N, K in SHAPES:
        print(f"Benchmarking M={M} N={N} K={K}...")
        t_tflops, t_us = bench_one(M, N, K, triton_fn) if triton_fn else (0.0, 0.0)
        g_tflops, g_us = bench_one(M, N, K, gluon_fn) if gluon_fn else (0.0, 0.0)
        writer.writerow([M, N, K, f"{t_tflops:.1f}", f"{t_us:.1f}", f"{g_tflops:.1f}", f"{g_us:.1f}"])
        print(f"  Triton: {t_tflops:.1f} TFLOPS  Gluon: {g_tflops:.1f} TFLOPS")

print(f"Results saved to {RESULT_FILE}")
```

### 4.2 Copy Gluon kernel if Docker image is outdated

If the Docker image doesn't have Gluon support, copy these files from local AITER:

```bash
# Kernel
scp /home/pensun/aiter/aiter/ops/triton/gluon/gemm_afp4wfp4.py \
    <node>:/mnt/m2m_nobackup/pensun/bench_serving/gluon_gemm_afp4wfp4.py

# Config JSON (required by Gluon)
scp /home/pensun/aiter/aiter/ops/triton/configs/gemm/gluon/gfx950-GEMM-AFP4WFP4.json \
    <node>:/mnt/m2m_nobackup/pensun/bench_serving/
```

### 4.3 Create launcher and deploy

Generate `/tmp/run_fp4_triton_gluon_bench.sh` (Docker wrapper), deploy via `scp`, and launch on one node (single GPU suffices for benchmarking).

### 4.4 CK-Tile and ASM numbers

CK-Tile and ASM results come from the profile CSV (`fp4_profile_*.csv`) generated during the `tune` step. No separate benchmark needed — the profile contains all kernel candidate timings.

---

## Step 5: Analysis Workflow (`analyze` action)

### 5.1 Collect results

If results aren't already local, `scp` them from both nodes:

```bash
mkdir -p /home/pensun/results/fp4_tune
scp uswslocpm2m-106-881.amd.com:/mnt/m2m_nobackup/pensun/results/fp4_tune/fp4_*.csv /home/pensun/results/fp4_tune/
scp uswslocpm2m-106-1236.amd.com:/mnt/m2m_nobackup/pensun/results/fp4_tune/fp4_*.csv /home/pensun/results/fp4_tune/
```

If a `results-dir` argument is given, use that instead of the default path.

### 5.2 Parse profile CSVs for ASM and CK-Tile results

Read all `fp4_profile_*.csv` files. For each row:
- **ASM kernels**: `kernelName` contains `f4gemm_bf16` (e.g., `f4gemm_bf16_per1x32Fp4_BpreShuffle_...`)
- **CK-Tile kernels**: `kernelName` contains `a4w4_blockscale` (e.g., `a4w4_blockscale_256x64x128x128_...`)

For each shape (M,N,K), find the **best TFLOPS** per backend. Skip rows with `us <= 0` or `errRatio > 0.05`.

### 5.3 Parse Triton/Gluon CSV

Read `fp4_triton_gluon_bench.csv`. Columns: `M,N,K,triton_tflops,triton_us,gluon_tflops,gluon_us`.

### 5.4 Build 4-way comparison table

Print a formatted table:

```
| Shape (MxNxK)         | Triton  | Gluon   | CK-Tile | ASM     | Best    | ASM/Best-Alt |
|-----------------------|---------|---------|---------|---------|---------|--------------|
| 16384x16384x32768     | 3477    | 3384    | 5054    | **5913**| ASM     | 1.17x        |
| ...                   | ...     | ...     | ...     | ...     | ...     | ...          |
```

Values are TFLOPS. Bold the winner per row. "Best-Alt" = best non-ASM backend.

### 5.5 Print summary statistics

```
ASM vs CK-Tile: min Xa, max Xb, mean Xc speedup
ASM vs Gluon:   min Xd, max Xe, mean Xf speedup
ASM vs Triton:  min Xg, max Xh, mean Xi speedup
Gluon vs Triton: min Xj, max Xk, mean Xl speedup

Shapes where ASM loses to another backend: <list or "none">
Shapes with errRatio > 0.05: <list or "none">
```

---

## Step 6: Status Workflow (`status` action)

Check both nodes:

```bash
# For each node:
ssh <node> 'echo "=== $(hostname) ===" && docker ps --format "table {{.Image}}\t{{.Status}}\t{{.Command}}" && echo "---" && ls -lh /mnt/m2m_nobackup/pensun/results/fp4_tune/*.csv 2>/dev/null && echo "---" && tail -3 /mnt/m2m_nobackup/pensun/results/fp4_tune/run_n*.log 2>/dev/null'
```

Report:
- Whether each node has a running Docker container
- Latest log lines (is tuning still running? completed? errored?)
- Result files present and their sizes/timestamps

---

## Step 7: Known Issues & Workarounds

These are battle-tested fixes discovered during real tuning runs. **Always apply these proactively.**

| Issue | Symptom | Fix |
|-------|---------|-----|
| ASM kernel path | `ASM kernel list file not exist: .../hsa//f4gemm/...` | `ln -sf gfx950/f4gemm "$HSA_DIR/f4gemm"` — tuner's `get_asm_dir()` returns `{hsa}/` but kernels are under `{hsa}/gfx950/` |
| Column name case | `KeyError: 'tile_m'` in `sort_values()` | `sed -i '1s/tile_M/tile_m/g; 1s/tile_N/tile_n/g'` on the ASM kernel CSV `f4gemm_bf16_per1x32Fp4.csv`. **MUST fix in ALL AITER installs** — both `/root/aiter/hsa/` and `/app/aiter-test/hsa/` (the tuner imports from `/app/aiter-test/` even when running from `/root/aiter/`) |
| K-alignment crash | `RuntimeError: The size of tensor a (36224) must match the size of tensor b (36352)` in `run_torch()` | Patch: `_k = min(x_f32.shape[1], x_scales_f32.shape[1]); x_f32 = x_f32[:, :_k] * x_scales_f32[:, :_k]` — occurs when K is not a multiple of 128 (e.g., 36224, 35200, 18304, 18816) |
| SSH escaping | `Ambiguous output redirect` or `Illegal variable name` with `$!` | **Never inline complex Docker commands via SSH.** Write scripts to files, `scp` them, then `ssh <node> 'bash /path/to/script.sh'` |
| CK-Tile unsupported shapes | `This GEMM is not supported!` | Some irregular shapes are unsupported by CK-Tile. ASM often still works. This is expected — not an error to fix. |
| Gluon missing in Docker | `No module named 'gluon'` or `FileNotFoundError` for config JSON | Copy kernel from local AITER: `aiter/ops/triton/gluon/gemm_afp4wfp4.py` and config: `aiter/ops/triton/configs/gemm/gluon/gfx950-GEMM-AFP4WFP4.json` |
| Docker `$!` escaping | `Illegal variable name` when using `$!` in SSH command | Use separate launcher `.sh` files (the `$!` is in the launcher, not passed through SSH) |
| N=576 GPU faults | `Memory access fault by GPU node-X` for N=576 shapes | N=576 is only 64-aligned (not 128-aligned). ASM/CK-Tile kernels require N ≥ 128-aligned. Skip these shapes or file upstream bug. |
| K=29568 unsupported | All kernel candidates fail, 0 shapes tuned | K=29568 (128-aligned but 256-unaligned) has no supported kernel. No fix — inherent kernel limitation. |
| mp=8 GPU memory faults | `Memory access fault` after batch 1, multiprocessing pool hangs | Use batched approach: split shapes into per-N,K groups, run each group as `--mp 1 --batch <group_size>` with `timeout 600`. Crashes in one group don't affect others. |
| Node 1 (881) GPU health | Consistent GPU memory faults on all shapes (including shapes that work on Node 2) | Node 1 may have hardware issues. Prefer Node 2 (1236) for reliability. |

---

## Step 8: Result Files Reference

### Output from tuning (`tune` action)

| File | Schema | Purpose |
|------|--------|---------|
| `fp4_tuned_<basename>.csv` | `cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio` | Best kernel per shape |
| `fp4_profile_<basename>.csv` | `cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio` | ALL kernel candidates per shape (100+ per shape) |

### Output from benchmarking (`bench` action)

| File | Schema | Purpose |
|------|--------|---------|
| `fp4_triton_gluon_bench.csv` | `M,N,K,triton_tflops,triton_us,gluon_tflops,gluon_us` | Triton vs Gluon comparison |

### Backend detection in profile CSV

| `kernelName` pattern | Backend |
|---------------------|---------|
| Contains `f4gemm_bf16` | ASM |
| Contains `a4w4_blockscale` | CK-Tile |

---

## Critical Rules

### Always apply all three fixes
The tuning script (`run_fp4_tune.sh`) MUST include all three fixes (ASM symlink, column case, K-alignment). Never generate a tuning script without them.

### Never inline Docker commands via SSH
Always write scripts to files, `scp` them to the node, then launch via `ssh <node> 'bash /path/to/script.sh'`. Shell escaping of Docker env vars, redirects, and `$!` is unreliable over SSH.

### Profile CSV contains CK-Tile AND ASM results
The tuner evaluates ALL kernel candidates (CK-Tile + ASM) per shape. You do NOT need separate CK-Tile or ASM benchmarks — parse the profile CSV.

### Triton and Gluon need separate benchmarking
These backends are not included in the AITER tuner. They require the separate `bench_fp4_triton_gluon.py` script.

### Shapes CSV format is simple
Header `M,N,K` followed by integer rows. No `cu_num` column — the tuner adds that internally. No quotes, no spaces.

### Monitor for known error patterns
When checking logs, look for these specific errors and apply the corresponding fix from the Known Issues table. If an error doesn't match any known pattern, investigate before retrying.
