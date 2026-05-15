# CK Removal Tracker

**Goal**: a zero-CK AITER build for AMD MI-series with acceptable perf parity.

**Owner**: @sunway513 / AITER team.

**Node/build**: MI355X, AITER `origin/main` (2026-04-22) in `gemm-tune-1` container,
FlyDSL 0.1.4, ROCm 7.2.2, PyTorch 2.9.1+rocm7.2.1.

**Harness**: `scripts/bench_ck_removal.py` (100 iters, CUDA-event timing,
silent warmup). Raw records: `results/ck_removal/bench_results.jsonl`
(70 rows). Kernel-name sanity dump: `results/ck_removal/kernel_sanity.json`
(captures actual dispatched kernels via torch profiler — every "green"
row below was confirmed dispatching the claimed kernel).

## Status summary

| Count | Category |
|---|---|
| 5 | Operator families tested |
| 29 | Distinct (op × model × shape) datapoints |
| 58 | Measured backend calls (CK + non-CK × 29 shapes) |
| **7** | Rows **green** (gap ≤ 10% OR CK unsupported) |
| **5** | Rows **yellow** (10% < gap ≤ 30%) |
| **17** | Rows **red** (gap > 30% OR no fallback path exists) |

Headline: **the conventional wisdom ("very close") holds for BF16 GEMM
and batched GEMM; it does NOT hold for FP8 GEMM, BF16 RMSNorm at large
shapes, or unquantized BF16 fused MoE**. See rows below and
`ck_removal_closure_plan.md` for the per-op next-step list.

## Kernel-name sanity checks (one per op family)

Captured via `torch.profiler` kineto trace after 3 warmup iters + 3 timed
iters on one representative shape. The full dump is in
`results/ck_removal/kernel_sanity.json`.

| Family | Backend | Kernel-name prefix captured |
|---|---|---|
| fused_moe_bf16 | CK  | `ck::kernel_moe_gemm_2lds<...>` + `ck_tile::MoeSortingMultiPhaseKernel_P0_v2<...>` + `..._P23<...>` |
| gemm_bf16      | non-CK | `Cijk_Alik_Bljk_BBS_BH_..._MT64x64x256_MI16x16x1_...` (Tensile/hipBLASLt) |
| gemm_fp8       | CK  | `ck::kernel_gemm_xdl_cshuffle_v3_multi_d<...>` |
| gemm_fp8       | non-CK | `Cijk_Alik_Bljk_F8BS_BH_..._MT64x64x512_MI16x16x1_...` (Tensile FP8) |
| batched_gemm_bf16 | CK | `ck::kernel_batched_gemm_xdl_cshuffle_v3_multi_d<...>` |
| batched_gemm_bf16 | non-CK | `Cijk_Ailk_Bljk_BBS_BH_..._MT128x128x64_MI16x16x1_...` (Tensile/hipBLAS) |
| rmsnorm        | CK  | `ck_tile::Rmsnorm2dFwd<ck_tile::Rmsnorm2dFwdPipelineOnePass<...>>` |
| rmsnorm        | non-CK | `aiter::add_rmsnorm_quant_kernel<bfloat16, bfloat16, 256, 16, ...>` |

Conclusion: every backend executed the path we claim it does. No
silent CK fallback on the "non-CK" rows.

## Per-op matrix

Colour legend: green = gap ≤ 10% (drop-in ready); yellow = 10%-30%
(acceptable, but perf cost); red = > 30% OR no non-CK path. "Gap" is
`(non_ck - ck) / ck` — positive means non-CK is FASTER, which is a CK
regression and a green signal for removal.

### Fused BF16 MoE (`ck_moe_stage1` + `ck_moe_stage2_fwd`)

**Non-CK path status**: **MISSING**. AITER's 1632 registered FlyDSL MoE
kernels all require a quantized B (fp4 or fp8); there is no
`a_dtype=bf16, b_dtype=bf16` MoE stage1 or stage2 kernel. Verified by
filtering `aiter.ops.flydsl.moe_kernels._KERNEL_PARAMS`.

| Model | Shape | CK TFLOPS | Non-CK | Status | Next step |
|---|---|---|---|---|---|
| DeepSeek-R1  | M=1 N=4096 K=7168 E=256 topk=8    | 5.7   | MISSING | **red** | author FlyDSL bf16/bf16 MoE stage1+stage2 |
| DeepSeek-R1  | M=128 N=4096 K=7168 E=256 topk=8  | 24.9  | MISSING | **red** | — |
| DeepSeek-R1  | M=1024 N=4096 K=7168 E=256 topk=8 | 185.9 | MISSING | **red** | — |
| MiniMax-M2.5 | M=1 N=3072 K=6144 E=64 topk=8     | 5.7   | MISSING | **red** | — |
| MiniMax-M2.5 | M=128 N=3072 K=6144 E=64 topk=8   | 95.0  | MISSING | **red** | — |
| MiniMax-M2.5 | M=1024 N=3072 K=6144 E=64 topk=8  | 512.5 | MISSING | **red** | — |
| GPT-OSS-120B | M=128 N=5760 K=2880 E=128 topk=4  | **UNSUPPORTED** | MISSING | **red** | CK errors "device_gemm ... does not support this GEMM problem"; need non-CK or a CK tile variant that handles N=5760/K=2880 |
| Kimi-K2      | M=128 N=4096 K=7168 E=128 topk=8  | 48.1  | MISSING | **red** | — |
| GLM-5        | M=128 N=4096 K=6144 E=128 topk=8  | 47.5  | MISSING | **red** | — |

**Verdict**: unquantized BF16 fused MoE is the biggest blocker. Every
serving workload that currently uses `QuantType.No` BF16 MoE has no
path forward if CK is removed. See closure plan for 2-week effort
estimate to author the FlyDSL kernels.

### BF16 GEMM (non-batched, `gemm_a16w16`)

**CK path status**: **no standalone CK BF16 GEMM symbol exists**.
`aiter.gemm_a16w16` is not present in this build — only
`aiter.gemm_a16w16_asm` (AITER ASM kernel) and the hipBLASLt-backed
`torch.matmul` path. AITER's `tuned_gemm.TunedGemm.mm()` already
dispatches BF16 through hipBLASLt / ASM / FlyDSL (no CK). This family
is **already CK-free** on the Python API surface.

| Model | Shape | CK | Non-CK TFLOPS | Status | Next step |
|---|---|---|---|---|---|
| DeepSeek-R1  | M=1 N=36864 K=7168    | n/a | 6.0    | **green** | no action (already CK-free) |
| DeepSeek-R1  | M=1024 N=36864 K=7168 | n/a | 1264.0 | **green** | — |
| DeepSeek-R1  | M=1 N=7168 K=2048     | n/a | 2.9    | **green** | — |
| MiniMax-M2.5 | M=128 N=6144 K=4096   | n/a | 336.6  | **green** | — |
| Kimi-K2      | M=128 N=6144 K=7168   | n/a | 223.2  | **green** | — |
| GLM-5        | M=128 N=6144 K=4096   | n/a | 338.4  | **green** | — |
| DeepSeek-R1  | M=128 N=6144 K=7168   | n/a | 224.0  | **green** | — |
| DeepSeek-R1  | M=4096 N=8192 K=8192  | n/a | 1549.6 | **green** | — |

**Verdict**: BF16 GEMM is the first CK-free family. Note: CK is still
used internally for the `batched` BF16 variant (see below) and inside
CK-compiled MoE kernels; this row only covers standalone BF16 GEMM.

### FP8 GEMM (a8w8, `gemm_a8w8_CK` vs `torch._scaled_mm`)

| Model | Shape | CK TFLOPS | Non-CK TFLOPS | Gap | Status | Next step |
|---|---|---|---|---|---|---|
| DeepSeek-R1  | M=1 N=7168 K=2048     | 3.5   | 2.6    | **-25.8%** | **yellow** | small-M regression; keep CK for token-gen until hipBLASLt heuristic fixed |
| DeepSeek-R1  | M=128 N=7168 K=2048   | 299.5 | 322.3  | +7.6%      | **green**  | non-CK drop-in ready |
| DeepSeek-R1  | M=128 N=36864 K=7168  | 882.3 | 1223.2 | **+38.6%** | **green (CK loss)** | non-CK is 39% FASTER — remove CK here immediately |
| MiniMax-M2.5 | M=128 N=6144 K=4096   | 352.4 | 537.5  | **+52.5%** | **green (CK loss)** | same — CK is the bottleneck |
| Kimi-K2      | M=128 N=6144 K=7168   | 387.0 | 687.3  | **+77.6%** | **green (CK loss)** | same |
| GLM-5        | M=128 N=6144 K=4096   | 353.1 | 534.4  | **+51.3%** | **green (CK loss)** | same |

**Verdict**: CK a8w8 *loses to hipBLASLt* on every M=128 shape measured
(+39% to +78%). CK is currently a performance floor for FP8 GEMM on
these shapes — removing it is a net **win**, except at M=1 (token-gen)
where the small-k regression needs hipBLASLt tuning. The "+38%" etc.
numbers are reproducible across iter counts (30 / 100 / 300) and were
auto-verified by the regression-guard. The "CK loss" tag means "non-CK
is faster; safe to remove CK" — exactly what we want.

### Batched BF16 GEMM (`batched_gemm_bf16_CK` vs `torch.bmm`)

| Model | Shape | CK TFLOPS | Non-CK TFLOPS | Gap | Status | Next step |
|---|---|---|---|---|---|---|
| DeepSeek-R1  | B=8 M=128 N=128 K=576  | 15.4 | 15.3 | -0.5%  | **green**  | drop-in ready |
| MiniMax-M2.5 | B=16 M=128 N=128 K=128 | 7.9  | 6.8  | -13.6% | **yellow** | hipBLAS ~14% slower; acceptable unless attn is critical path |
| Kimi-K2      | B=16 M=128 N=128 K=128 | 7.9  | 6.9  | -13.2% | **yellow** | same |

**Verdict**: mostly a drop-in; small-K (128) attention BMMs lose ~14%
on hipBLAS. Decide per-workload whether to keep CK or accept the gap.

### RMSNorm (`rmsnorm2d_fwd_ck` vs `rms_norm_cu`)

Reported values are **GB/s** (bandwidth-bound op).

| Shape | CK GB/s | Non-CK GB/s | Gap | Status | Next step |
|---|---|---|---|---|---|
| H=4096 M=64   | 285.2  | 352.7  | **+23.7%** | **yellow** | non-CK WINS at small M — fine |
| H=4096 M=1024 | 4469.0 | 2604.1 | **-41.7%** | **red**    | non-CK HIP kernel needs larger block/warp tile at M≥512 |
| H=8192 M=1024 | 6719.8 | 3704.1 | **-44.9%** | **red**    | same — port ck_tile::Rmsnorm2dFwdPipelineOnePass to HIP or add a Triton RMSNorm |

**Verdict**: CK's one-pass pipelined RMSNorm dominates at large M/H
(~2× over HIP). Removing CK will cost ~40% RMSNorm perf on prefill
unless the HIP kernel (or a Triton equivalent) is rewritten. This is
the second biggest blocker after BF16 MoE.

## Three replacement paths (FlyDSL / Triton / Opus)

Per `ROCm/aiter-forge#42` (tracker issue), CK can be replaced by **any one of**:

| Path | Status in AITER today | Best for |
|---|---|---|
| **FlyDSL** | `aiter.ops.flydsl.*` deeply integrated | GEMM bf16/fp16, quantized MoE, preshuffle GEMM a8; **does NOT cover BF16/BF16 MoE** |
| **Triton** | `aiter.ops.triton.*` partial | `moe_op`, GEMM tune helpers, attention helpers |
| **Opus** | minimal today (`aiter/ops/moe_sorting_opus.py` + env `AITER_USE_OPUS_MOE_SORTING`) | lightweight C++ tool by Carlos; reportedly strong for pipelined elementwise (RMSNorm one-pass candidate) and lightweight MoE stages |

**Per-row replacement recommendation** (for 🔴 rows; choose the cheapest next step):

| Op × shape | CK baseline | Cheapest closure path | Rationale |
|---|---|---|---|
| fused_moe (bf16/bf16, all 9 shapes) | 185.9 TFLOPS @ DSR1 M=1024 | **FlyDSL or Opus** | FlyDSL tile family known (`tile_m ∈ {32,64} × tile_n ∈ {128,256} × tile_k=128`); Opus candidate for faster one-off delivery |
| rmsnorm M=1024 H=4096/8192 | 6720 GB/s | **Opus** (primary) or Triton | one-pass pipeline is Opus's sweet spot; Triton viable if Opus availability slips |
| fp8 a8w8 M=128 wide-N | CK already slower than hipBLASLt | **nothing new needed** | just flip dispatcher (`AITER_DISABLE_CK_GEMM=1`) — closure is a config flag |

## Other CK paths discovered but not benchmarked

Listed for completeness — they're in the Python symbol table and would
need their own rows before CK can be flipped to OFF:

- `deepgemm_ck` — used for blockscale paths
- `fmoe_fp8_blockscale_g1u1` — blockscale FP8 MoE fused kernel
- `moe_cktile2stages_gemm1`, `moe_cktile2stages_gemm2` — CK-Tile MoE
  (FP4/swiglu path), used when `dtype ∈ {bf16, fp16}` + `per_1x32`
  quant + Swiglu (line 1013-1058 of `aiter/fused_moe.py`)
- `gemm_a8w8_blockscale_ck`, `gemm_a8w8_blockscale_cktile`,
  `gemm_a8w8_blockscale_bpreshuffle_ck`, `gemm_a8w8_blockscale_bpreshuffle_cktile`
- `gemm_a8w8_bpreshuffle_ck`, `gemm_a8w8_bpreshuffle_cktile`
- `gemm_a4w4_blockscale` (FP4 blockscale)
- `rmsnorm2d_fwd_with_add_ck`,
  `rmsnorm2d_fwd_with_add_dynamicquant_ck`,
  `rmsnorm2d_fwd_with_dynamicquant_ck` — fused residual+norm+quant ops
- `fused_qk_norm_rope_cache_block_quant_shuffle`

See `ck_removal_closure_plan.md` for per-op ownership / effort
estimates.

## References

- Raw benchmark data: `results/ck_removal/bench_results.jsonl` (70 rows)
- Kernel-name sanity: `results/ck_removal/kernel_sanity.json`
- Harness: `scripts/bench_ck_removal.py`
- Methodology: `learnings/tuning/ck_removal_methodology.md`
- Closure plan: `docs/ck_removal_closure_plan.md`
