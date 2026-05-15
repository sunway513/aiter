# CK Removal Closure Plan

**Companion to**: `docs/ck_removal_tracker.md` (measurements) and
`scripts/bench_ck_removal.py` (reproducer).

## Replacement paths considered

Three options per 🔴 op — pick the cheapest:

1. **FlyDSL** (`aiter.ops.flydsl.*`) — deeply integrated; known gaps: no BF16/BF16 MoE (all FlyDSL MoE kernels require quantized B).
2. **Triton** (`aiter.ops.triton.*`) — partial coverage (moe_op, GEMM tune helpers, attention helpers).
3. **Opus** — lightweight C++ tool authored by Carlos (AMD). Currently only `aiter/ops/moe_sorting_opus.py` + `AITER_USE_OPUS_MOE_SORTING` env var is integrated, but Opus reportedly produces high-perf kernels quickly and is a strong candidate for the RMSNorm one-pass gap and the BF16/BF16 fused-MoE gap. Engage Carlos when evaluating #4 and #5 below.

**Question being answered**: "what has to be true, by when, before AITER
can flip `ENABLE_CK=False` without regressing MI355X serving perf?"

**Status (2026-04-22)**: 2 of 5 measured operator families are already
CK-removal-ready (BF16 GEMM, FP8 a8w8 M=128). The remaining 3 need
kernel work. Biggest risk is BF16 fused MoE.

## Close-order ranking

Ordered by (impact × simplicity). Do the top of the list first — they
buy us CK-free builds for a meaningful slice of workloads while the
MoE work is in flight.

### 1. GREEN — remove CK for FP8 a8w8 GEMM at M≥64 (P0, 1-2 days)

**Evidence**: all five M=128 shapes run 39%-78% FASTER on hipBLASLt than
on CK (`gemm_a8w8_CK`). Regression-guard verified.

**Next step**: add a runtime env flag `AITER_DISABLE_CK_GEMM=1` in
`aiter/ops/gemm_op_a8w8.py` that skips the CK dispatch and routes to
`torch._scaled_mm`. Keep CK as the M<64 fallback until hipBLASLt ships
better small-M FP8 tiles (owner: hipBLASLt team, ~2 weeks out).

**Owner**: AITER ops team.

**Acceptance test**: replay the 5 shapes in `results/ck_removal/bench_results.jsonl`;
non-CK tflops must be ≥ 0.95 × CK tflops.

### 2. GREEN — formally declare BF16 GEMM CK-free (P0, 30 min)

**Evidence**: `aiter.gemm_a16w16` does not exist; `tuned_gemm.TunedGemm.mm()`
already dispatches through hipBLASLt/ASM/FlyDSL with no CK path.

**Next step**: add a CI tripwire `tests/test_ck_free_bf16_gemm.py` that
imports `aiter` and `assert not hasattr(aiter, "gemm_a16w16_CK")` + a
`grep -L "composable_kernel" csrc/gemm_bf16*.cpp` smoke. Keep it green
so future PRs can't re-introduce a CK BF16 GEMM path.

**Owner**: AITER build team.

### 3. YELLOW → GREEN — batched BF16 GEMM acceptable loss (P1, 1 week)

**Evidence**: hipBLAS `torch.bmm` is 14% slower than `batched_gemm_bf16_CK`
on B=16/M=N=128/K=128 (MiniMax and Kimi attn BMMs). One row (B=8 K=576)
is ≤ 1% green.

**Next step**: (a) measure if attn BMMs are critical-path in production
TTFT; if not, accept the 14% loss and remove CK now. (b) if critical,
port the `ck::kernel_batched_gemm_xdl_cshuffle_v3_multi_d` B=16 tile to
a HIP equivalent; rocBLAS 4.x tuning list should already contain it.

**Owner**: AITER attention team.

### 4. YELLOW → RED — RMSNorm large-M perf gap (P1, 2 weeks)

**Evidence**: `aiter.rms_norm_cu` (HIP) delivers 2.6-3.7 TB/s at M=1024
while `aiter.rmsnorm2d_fwd_ck` (ck_tile) hits 4.5-6.7 TB/s — a 41-45%
gap. Small-M (M=64) already favors the HIP kernel (+24%).

**Root cause**: the HIP `rms_norm_cu` uses a simple 256-thread block
with 16-element packs; the CK variant pipelines the `input → var →
normalize` passes into a single persistent kernel
(`Rmsnorm2dFwdPipelineOnePass`) so the hidden-dim reduction doesn't
re-read from HBM.

**Next step**: pick the cheapest of three options:
  (a) port the one-pass pattern to `rms_norm_cu` (est. 2 weeks, 1 kernel engineer);
  (b) add a Triton RMSNorm under `aiter/ops/triton/rmsnorm.py` tuned for MI355X (est. 1 week);
  (c) **ask Carlos for an Opus RMSNorm one-pass kernel** — this is Opus's stated sweet spot (pipelined elementwise with persistent-kernel variance reduction). Could land in days rather than weeks if Carlos has bandwidth.

**Owner**: AITER norm-ops team + Triton lead.

**Acceptance test**: non-CK GB/s ≥ 0.9 × CK GB/s for all 3
`results/ck_removal/bench_results.jsonl` rmsnorm rows.

### 5. RED — author FlyDSL BF16/BF16 fused MoE (P0, 2-3 weeks)

**Evidence**: 9 of 9 `fused_moe_bf16` rows have **no** non-CK path
because FlyDSL's MoE kernel registry (1632 entries) requires quantized
B (fp4 or fp8). Unquantized BF16 serving — still common for some
pre-quant workloads — depends entirely on `ck_moe_stage1` +
`ck_moe_stage2_fwd`.

**Next step**: author two FlyDSL kernels following the pattern of the
existing `flydsl_moe1_afp8_wfp4_bf16_t32x32x256_bnt0` but with
`a_dtype=bf16, b_dtype=bf16, out_dtype=bf16`. The handoff learning
(`learnings/moe/coderfeli_small_tile_feedback.md`) says tile_m ∈
{32, 64} × tile_n ∈ {128, 256} × tile_k=128 is the preferred config
family at M≥512. Start with one tile variant each for stage1 and stage2;
tune in a follow-up.

**Owner**: FlyDSL MoE team (coderfeli) + AITER MoE. **Opus alternative**: if FlyDSL 2-3 week timeline blocks the CK-off target date, ask Carlos whether Opus can produce BF16/BF16 MoE stage1+2 kernels faster. Opus is already integrated for `moe_sorting`, so extending to the GEMM stages is a known-viable path rather than greenfield.

**Acceptance test**: non-CK tflops ≥ 0.85 × CK tflops for the 8
measurable DSR1 / MiniMax / Kimi / GLM rows
(`results/ck_removal/bench_results.jsonl`, family=fused_moe_bf16).

**Related**: also patch the GPT-OSS M=128 N=5760 K=2880 CK
`UNSUPPORTED` case — either add a compatible CK tile or ensure the new
FlyDSL kernel covers it.

### 6. RED — audit the "not-yet-benchmarked" CK paths (P1, 3 days)

**List** (`ck_removal_tracker.md` §"Other CK paths"):
`deepgemm_ck`, `fmoe_fp8_blockscale_g1u1`, `moe_cktile2stages_gemm1/2`,
`gemm_a8w8_blockscale*_ck`/`*_cktile*`, `gemm_a8w8_bpreshuffle_ck`/
`_cktile`, `gemm_a4w4_blockscale`, `rmsnorm2d_fwd_with_add_ck` family,
`fused_qk_norm_rope_cache_block_quant_shuffle`.

**Next step**: extend `scripts/bench_ck_removal.py` BENCHES table with
one row per family (add `gemm_a8w8_blockscale` FP8-blockscale,
`moe_cktile2stages` FP4-Swiglu path, `deepgemm_ck` batched blockscale);
run the same bench loop; update the tracker. For `fused_qk_norm_rope`
it's likely fine because non-CK FlyDSL RoPE exists
(see `learnings/` RoPE entries), but verify.

**Owner**: aiter-forge (this repo).

## Proposed close dates

| Item | Owner | ETA |
|---|---|---|
| (1) FP8 a8w8 M≥64: runtime disable | AITER ops | 2026-04-25 (3 days) |
| (2) BF16 GEMM CK-free tripwire | AITER build | 2026-04-23 (1 day) |
| (3) Batched BF16 GEMM decision | AITER attn | 2026-04-29 (1 wk) |
| (4) RMSNorm non-CK parity | Norm/Triton | 2026-05-06 (2 wk) |
| (5) FlyDSL BF16/BF16 MoE | FlyDSL + AITER MoE | 2026-05-13 (3 wk) |
| (6) Audit remaining CK paths | aiter-forge | 2026-04-25 (3 days) |

**Earliest "CK fully off" date**: 2026-05-13 (gated by (5) only).
BF16-GEMM-only + FP8-a8w8-M≥64 builds can go CK-free a full 3 weeks
earlier, which is worth a partial release.

## Ownership table

| Family | On-call | Escalation |
|---|---|---|
| fused_moe_bf16 | coderfeli (FlyDSL) | AITER MoE lead |
| gemm_fp8 | @sunway513 (AITER ops) | hipBLASLt contact |
| rmsnorm | AITER norm-ops | Triton lead |
| batched_gemm_bf16 | AITER attn | rocBLAS |
| gemm_bf16 | AITER build | — |

## Re-run instructions

```bash
docker cp /home/pensun/aiter-forge/scripts/bench_ck_removal.py \
    gemm-tune-1:/tmp/bench_ck_removal.py
docker exec gemm-tune-1 bash -c '
  cd /tmp &&
  HIP_VISIBLE_DEVICES=0 python3 bench_ck_removal.py \
    --output /tmp/ck_removal_results/bench_results.jsonl \
    --iters 100'
docker cp gemm-tune-1:/tmp/ck_removal_results/bench_results.jsonl \
    /home/pensun/aiter-forge/results/ck_removal/bench_results.jsonl
```

Full loop takes ~12 seconds on one MI355X. Re-run when AITER main moves
by more than 3 days (dashboard-staleness rule,
`learnings/tuning/dashboard_staleness_detection.md`).

## Build-level impact — if CK were removed today

Separate companion audit (PR `feat/ck-build-comparison`, 2026-04-22)
rebuilt AITER with `ENABLE_CK=0` and called the **user-facing
dispatcher entries** (`aiter.fused_moe`, `aiter.gemm_a8w8`,
`aiter.rmsnorm2d_fwd`, `aiter.batched_gemm_bf16`, `torch.matmul`)
rather than internal `*_CK` symbols. Full methodology in
`docs/ck_build_comparison.md` + `docs/ck_build_audit.md`; 4-section
learning at `learnings/tuning/ck_build_level_vs_api_level.md`.

**Headline**: of 5 op families × 19 production shapes tested at the
dispatcher level, **13 rows are safe, 0 are acceptable, 6 are red**.
CK-off is **not ship-able today** for fused MoE or RMSNorm N>8192.

### 🔴 rows with no FlyDSL/Triton/Opus alternative yet

Each row below fails under `ENABLE_CK=0`. The "blocker path" names
what the closure plan item actually is; the "today?" column answers
"is there a working non-CK kernel right now that the dispatcher
could route to if patched?".

| Family × row | Root error | Blocker path | Today? |
|---|---|---|---|
| `fused_moe_bf16` DSR1 M=1 | `module_moe_sorting` JIT build fails under ENABLE_CK=0 | **MoE sorting must build under ENABLE_CK=0** (shim its CK includes); prerequisite for (5) | **no** |
| `fused_moe_bf16` DSR1 M=128 | same | same | **no** |
| `fused_moe_bf16` DSR1 M=1024 | same | same | **no** |
| `fused_moe_bf16` MiniMax M=128 | same | same | **no** |
| `fused_moe_bf16` Kimi M=128 | same | same | **no** |
| `rmsnorm` H=16384 M=128 | dispatcher hard-routes to CK at N>8192; `module_rmsnorm` JIT build fails | (4) replace CK one-pass with non-CK one-pass OR patch dispatcher to avoid CK at N>8192 | **no** (dispatcher lacks fallback) |

### Rows that DO work at build-level (corrected from first audit)

An earlier pass with 3-warmup bench saw `batched_gemm_bf16` rows at
~45-60% of CK-on and the 3 N≤8192 `rmsnorm` rows ~60% slow. Those
results were a **warmup artifact**: the first call in the timed loop
pulled the JIT compile into the measurement. Re-run with 20-warmup
+ 100 iters:

- `batched_gemm_bf16` (both rows): within ±4% of CK-on — 🟢 safe.
- `rmsnorm` N≤8192 (H=4096 M=64, H=4096 M=1024, H=8192 M=1024):
  within ±9% of CK-on — 🟢 safe. The HIP path
  (`module_rmsnorm_quant`) builds cleanly under ENABLE_CK=0 and
  the dispatcher already routes there for N≤8192.

### Delta vs API-level conclusions (PR #41)

Build-level work exposes two gaps the API-level comparator missed:

1. **`module_moe_sorting` is an upstream CK dependency of ALL fused
   MoE paths** (quantized OR unquantized). Item (5) of the close-order
   was framed as "author FlyDSL BF16/BF16 MoE kernels" — that closes
   the *kernel* gap, but unless `module_moe_sorting` also gets a
   CK-free build path, no fused_moe call will even reach those
   kernels. Adjust (5) to cover both.
2. **`module_rmsnorm` fails to build under `ENABLE_CK=0`** while
   `module_rmsnorm_quant` (the HIP path host) builds fine. Item (4)'s
   non-CK parity effort additionally needs to make `module_rmsnorm`
   either build as a shim (empty CK path) or the dispatcher needs to
   stop calling `rmsnorm2d_fwd_ck` at N>8192.

It also contradicts what the API-level tracker implied for
`batched_gemm_bf16` (tracker showed "drop-in mostly" at 14% loss):
at build level the dispatcher **does not route to `torch.bmm`**; it
calls the CK-backed module which builds fine on CK-off and matches
perf. So the real state is better than PR #41 predicted for this
family.

### Earliest "build-level CK-off is ship-able" date

| Gate | Owner | ETA |
|---|---|---|
| Existing items (1)-(6) | as listed above | 2026-05-13 |
| NEW: `module_moe_sorting` ENABLE_CK=0 shim | AITER MoE | 2026-04-29 (1 wk) — prerequisite for (5) |
| NEW: `module_rmsnorm` ENABLE_CK=0 shim OR dispatcher guard | AITER norm-ops | merged into (4) |

A partial ship (BF16-GEMM + FP8-a8w8 + batched_gemm_bf16 + rmsnorm
N≤8192 workloads) is possible at the current date as long as it is
documented that `ENABLE_CK=0` does not yet cover fused MoE or
RMSNorm at N>8192.
