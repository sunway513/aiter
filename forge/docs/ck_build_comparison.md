# CK-on vs CK-off Build Comparison

**Snapshot**: 2026-04-22, AITER `origin/main` @ `6890159`, MI355X,
container `gemm-tune-1`.

**Companion docs**:
- `docs/ck_build_audit.md` — which csrc modules + Python dispatchers
  depend on `ENABLE_CK`.
- `docs/ck_removal_tracker.md` — API-level CK vs non-CK (PR #41).
- `learnings/tuning/ck_build_level_vs_api_level.md` — why this doc is
  different from PR #41.

**Question answered**: if a user rebuilds AITER with `ENABLE_CK=0`
and calls the same production dispatcher entries
(`aiter.fused_moe(...)`, `aiter.gemm_a8w8(...)`,
`aiter.rmsnorm2d_fwd(...)`, `aiter.batched_gemm_bf16(...)`,
`torch.matmul` for BF16 GEMM), what happens?

## Method

Two builds of AITER main @ `6890159`:

| Build | Location | Flag | `.so` cache |
|---|---|---|---|
| CK-on  | `/app/aiter-test` (container `gemm-tune-1`) | `ENABLE_CK=1` (default) | preserved from `2026-04-12` prebuild |
| CK-off | `/tmp/aiter-noCK` (copy of above) | `ENABLE_CK=0` | cleared for 27 CK-dependent modules; JIT re-compiles per-call |

For each of the 19 (family × model × shape) rows, we call the
**dispatcher entry** (not the internal `*_CK` symbol), measure TFLOPS
or GB/s with a **20-warmup + 100-iter** CUDA-event timer, and record
`status=OK` or `status=BROKEN` with the full exception string. The
big warmup is load-bearing: any first-call JIT compile must fold into
warmup rather than taint the timed window.

Reproducer: `scripts/bench_ck_build_comparison.py`.
Raw data: `results/ck_build_comparison/ck_on.jsonl` +
`results/ck_build_comparison/ck_off.jsonl`, merged into
`results/ck_build_comparison/bench_results.jsonl`.

Import-time check on the CK-off build:
```
ENABLE_CK: False
gemm_a8w8_CK attr:         <function>   (still registered at Python level)
ck_moe_stage1 attr:        <function>   (Python wrapper; JIT compiles on call)
rmsnorm2d_fwd_ck attr:     <function>   (same)
batched_gemm_bf16_CK attr: <function>   (same)
```
Every `ck_*` symbol is importable; breakage only manifests when the
backing JIT module fails to build on first invocation, AT WHICH POINT
the error is a `RuntimeError: [aiter] build [module_X] failed !!!`
from the JIT helper.

## End-to-end dispatcher outcome (build-level)

Status legend: 🟢 safe (CK-off ≥ 0.90× CK-on), 🟡 acceptable (0.70×
≤ CK-off < 0.90×), 🔴 broken or >30% slower. Values in TFLOPS for
GEMM/MoE, GB/s for RMSNorm. Gap = (CK-off − CK-on) / CK-on × 100.

### fused_moe_bf16 — `aiter.fused_moe(QuantType.No)`

| Model × shape | CK-on (TFLOPS) | CK-off | ΔE2E | Status |
|---|---|---|---|---|
| DSR1 M=1      | 5.68    | BROKEN — `module_moe_sorting` build fails | −100% | 🔴 |
| DSR1 M=128    | 24.97   | BROKEN — same | −100% | 🔴 |
| DSR1 M=1024   | 185.69  | BROKEN — same | −100% | 🔴 |
| MiniMax M=128 | 95.06   | BROKEN — same | −100% | 🔴 |
| Kimi M=128    | 48.12   | BROKEN — same | −100% | 🔴 |

Root error (CK-off, verbatim):
```
RuntimeError: [aiter] build [module_moe_sorting] under
/tmp/aiter-noCK/aiter/jit/build/module_moe_sorting/build failed !!!!!!
```
`module_moe_sorting` is a prerequisite of every MoE dispatch path; it
sits upstream of `module_moe_ck2stages`. Its CK-off build failure
kills fused_moe before reaching the BF16/BF16 stage1/2 selection. The
Python-level `if not ENABLE_CK` guard that exists for MHA is **not**
present here.

### gemm_bf16 — `torch.matmul`

| Model × shape | CK-on (TFLOPS) | CK-off | ΔE2E | Status |
|---|---|---|---|---|
| DSR1 M=1      N=36864 K=7168 | 6.02    | 6.04    | +0.3% | 🟢 |
| DSR1 M=1024   N=36864 K=7168 | 1281.3  | 1280.5  | −0.1% | 🟢 |
| DSR1 M=128    N=7168  K=2048 | 324.5   | 322.3   | −0.7% | 🟢 |
| DSR1 square M=4096           | 1555.8  | 1555.1  | 0.0%  | 🟢 |

CK-free baseline: `torch.matmul` dispatches through hipBLASLt entirely
outside AITER's `@compile_ops` plumbing. Noise-level variance across
builds.

### gemm_fp8 — `aiter.gemm_a8w8`

| Model × shape | CK-on (TFLOPS) | CK-off | ΔE2E | Status |
|---|---|---|---|---|
| DSR1 M=1    N=7168  K=2048 | 2.73   | 2.70   | −1.1% | 🟢 |
| DSR1 M=128  N=7168  K=2048 | 322.3  | 313.5  | −2.7% | 🟢 |
| DSR1 M=128  N=36864 K=7168 | 967.5  | 906.2  | −6.3% | 🟢 |
| MiniMax M=128 N=6144 K=4096 | 363.1 | 365.7  | +0.7% | 🟢 |

**Surprise**: `module_gemm_a8w8` builds successfully under
`ENABLE_CK=0` (~150s first JIT) and dispatched perf matches CK-on.
The `ck_gemm_a8w8/` sources include `composable_kernel/` headers
directly from the submodule's `include/` path — the `-DENABLE_CK=0`
macro doesn't gate those includes, so `a8w8_rowwise_*` CK kernels
still compile. Whether this is an intentional guarantee or an
accident of include-path layout is unclear; it is **not** what the
`ck_tile_shim.h` docstring suggests ("compiling without the full
Composable Kernel dependency"). Treat as green; flag the fragility
in the closure plan.

### batched_gemm_bf16 — `aiter.batched_gemm_bf16`

| Model × shape | CK-on (TFLOPS) | CK-off | ΔE2E | Status |
|---|---|---|---|---|
| DSR1 attn B=8  M=128 N=128 K=576 | 15.43 | 15.38 | −0.3% | 🟢 |
| MiniMax attn B=16 M=128 N=128 K=128 | 9.34 | 9.70 | +3.9% | 🟢 |

`module_batched_gemm_bf16` also builds under `ENABLE_CK=0` (~2 min
first JIT) and dispatches a CK kernel that matches CK-on perf once
warm. Initial bench runs with 3-warmup showed a large apparent gap
(−45 to −60%); this turned out to be the first-call JIT compile
being charged against the timed window. Re-ran with 20-warmup and
the gap disappears — kernels are byte-identical across builds.

### rmsnorm — `aiter.rmsnorm2d_fwd`

Dispatcher: `if input.shape[-1] > 8192: rmsnorm2d_fwd_ck(...) else
rms_norm_cu(...)`. So N≤8192 is the HIP path on **both** builds;
N>8192 is the CK path only reachable under `ENABLE_CK=1`. Reported
values are GB/s.

| Shape | CK-on (GB/s) | CK-off | ΔE2E | Status |
|---|---|---|---|---|
| H=4096  M=64    (HIP path)   | 217.2  | 208.0  | −4.2% | 🟢 |
| H=4096  M=1024  (HIP path)   | 3460.4 | 3176.2 | −8.2% | 🟢 |
| H=8192  M=1024  (HIP path)   | 6544.4 | 6619.4 | +1.1% | 🟢 |
| H=16384 M=128   (CK path)    | 2038.0 | BROKEN — `module_rmsnorm` build fails | −100% | 🔴 |

The dispatcher's N≤8192 guard already routes common production
shapes to the HIP kernel (which builds under ENABLE_CK=0 via
`module_rmsnorm_quant`), so three of the four rows are safe at
build-level. Only the N>8192 path trips on CK-off because
`module_rmsnorm` (which hosts `rms_norm`, `rmsnorm2d_fwd_ck`,
`rmsnorm2d_fwd_with_add_ck`, etc.) unconditionally includes CK
headers that don't survive the shim. CSR1/MiniMax/Kimi/GLM-5 all
have hidden sizes ≤ 8192 in the tracked dashboard, so in practice
the CK path is exercised by less-common configurations. Dashboard
shapes with N>8192 will fail.

## Summary counts

19 rows × 5 op families, 10% / 30% thresholds on ΔE2E:

| Bucket | Count | Families covered |
|---|---|---|
| 🟢 Safe (CK-off ≥ 0.90× CK-on) | 13 | gemm_bf16 (4/4), gemm_fp8 (4/4), batched_gemm_bf16 (2/2), rmsnorm N≤8192 (3/3) |
| 🟡 Acceptable (0.70× ≤ CK-off < 0.90×) | 0 | — |
| 🔴 Broken or >30% slower | 6 | fused_moe_bf16 (5/5 BROKEN), rmsnorm N>8192 (1/1 BROKEN) |

Out of **5 tracked op families**:
- **Fully ship-able today on CK-off**: 3 (gemm_bf16, gemm_fp8,
  batched_gemm_bf16).
- **Partially ship-able** (dispatcher works at common shapes, breaks
  at rare shapes): 1 (rmsnorm — N≤8192 OK, N>8192 broken).
- **Fully broken at build level**: 1 (fused_moe_bf16 — all 5 tested
  shapes fail on `module_moe_sorting`).

## Ship-able decision

**A CK-off AITER build is ship-able today for any workload that does
not need fused MoE or N>8192 RMSNorm**, which covers pure-GEMM
inference but not LLM serving with MoE experts.

### Gates for a complete CK-off ship

1. **`module_moe_sorting` must build under `ENABLE_CK=0`.** This
   alone unlocks every fused_moe flow (quantized and unquantized,
   all 5 MiniMax/DSR1/Kimi/MoE shapes). Either shim the CK includes
   in `csrc/py_itfs_ck/moe_sorting/` or factor sorting into a pure
   HIP/CU module.
2. **`module_rmsnorm` must build under `ENABLE_CK=0`** OR the
   dispatcher's `N>8192` branch must fallback to HIP. Either
   acceptable.
3. **Fused MoE BF16/BF16 kernel work** (FlyDSL BF16/BF16 MoE
   stage1/2, or Opus equivalent) stays on the plan — item (5) — but
   is now gated behind fixing #1 first; otherwise there's nothing
   for the new kernel to plug into.

### What already works

1. **BF16 GEMM** (`torch.matmul`) — already CK-free.
2. **FP8 a8w8 GEMM** (`aiter.gemm_a8w8`) — unexpectedly builds and
   runs on CK-off. Fragile; depends on `composable_kernel/` submodule
   headers still being present in the include path.
3. **Batched BF16 GEMM** (`aiter.batched_gemm_bf16`) — builds and
   matches CK-on perf.
4. **RMSNorm at hidden size ≤ 8192** (`aiter.rmsnorm2d_fwd`) — HIP
   path picked by dispatcher, builds cleanly.

## Paired-row rollup

See `results/ck_build_comparison/bench_results.jsonl` for the merged
JSONL (one `paired` record per row, plus raw `bench` records and
per-mode `provenance`). Regenerate with:

```bash
python3 scripts/merge_ck_build_comparison.py \
  --ck-on  results/ck_build_comparison/ck_on.jsonl \
  --ck-off results/ck_build_comparison/ck_off.jsonl \
  --out    results/ck_build_comparison/bench_results.jsonl
```

## Re-run instructions

```bash
docker cp scripts/bench_ck_build_comparison.py gemm-tune-1:/tmp/bench_ck_build_comparison.py
docker cp scripts/merge_ck_build_comparison.py gemm-tune-1:/tmp/merge_ck_build_comparison.py

# CK-on baseline (uses the pre-existing /app/aiter-test build).
docker exec gemm-tune-1 bash -c '
  cd /tmp && HIP_VISIBLE_DEVICES=0 python3 bench_ck_build_comparison.py \
    --mode ck_on --output /tmp/ck_build_comparison/ck_on.jsonl --iters 100'

# CK-off: copy aiter, clear prebuilt CK-dependent modules, export ENABLE_CK=0.
docker exec gemm-tune-1 bash -c '
  cp -a /app/aiter-test /tmp/aiter-noCK &&
  for m in module_gemm_a8w8 module_batched_gemm_bf16 module_rmsnorm \
           module_moe_ck2stages module_moe_sorting module_aiter_operator \
           module_aiter_unary module_aiter_core module_custom module_moe_asm; do
    rm -f /tmp/aiter-noCK/aiter/jit/${m}*.so
    rm -rf /tmp/aiter-noCK/aiter/jit/build/${m}
  done'

docker exec gemm-tune-1 bash -c '
  cd /tmp &&
  ENABLE_CK=0 PYTHONPATH=/tmp/aiter-noCK HIP_VISIBLE_DEVICES=0 \
  python3 bench_ck_build_comparison.py \
    --mode ck_off --output /tmp/ck_build_comparison/ck_off.jsonl --iters 100'

docker cp gemm-tune-1:/tmp/ck_build_comparison/ck_on.jsonl  results/ck_build_comparison/
docker cp gemm-tune-1:/tmp/ck_build_comparison/ck_off.jsonl results/ck_build_comparison/

python3 scripts/merge_ck_build_comparison.py \
  --ck-on  results/ck_build_comparison/ck_on.jsonl \
  --ck-off results/ck_build_comparison/ck_off.jsonl \
  --out    results/ck_build_comparison/bench_results.jsonl
```

Expect CK-on loop ~8s; CK-off loop ~3 minutes after the CK-dependent
modules have been JIT-rebuilt once (first run is ~30-40 minutes).
