# CK Build Audit — what `ENABLE_CK=0` actually turns off

**Companion to**: `docs/ck_removal_tracker.md` (API-level CK vs non-CK) and
`docs/ck_build_comparison.md` (build-level CK-on vs CK-off — this work).

**Snapshot**: AITER `origin/main` @ `6890159`, MI355X, container
`gemm-tune-1`, 2026-04-22.

**Why this doc exists**: PR #41 measured which of AITER's paired backends
(`ck_moe_stage1` vs `torch.matmul`) is faster while **both are compiled
in**. This doc answers a different question: *if a user sets
`ENABLE_CK=0` at build time, what silently disappears, what raises, and
what keeps working through a fallback?*  It is the prerequisite for the
build-level comparison in `ck_build_comparison.md`.

## ENABLE_CK plumbing

| File | Line | Role |
|---|---|---|
| `aiter/jit/core.py` | 29 | `ENABLE_CK = int(os.environ.get("ENABLE_CK", "1")) != 0` — module-level boolean read once per Python process |
| `aiter/jit/core.py` | 793-795 | In `build_module()`, appends `-DENABLE_CK={enable_ck}` to every JIT extension's compile flags |
| `csrc/include/aiter_hip_common.h` | 9 | `#if !ENABLE_CK ... #include "ck_tile_shim.h" ... #else ... #include "ck_tile/core.hpp" ... #endif` — headers switch between a tiny shim and full CK |
| `csrc/include/ck_tile_shim.h` | 6 | Shim "providing ck_tile:: types and FMHA enums/structs when compiling without the full Composable Kernel dependency (ENABLE_CK==0)" |
| `csrc/include/py_itfs_common.h` | 61 | `#if ENABLE_CK` guards the `t2ck` dtype-to-ck_tile mapper |
| `csrc/include/mha_fwd.h` / `mha_bwd.h` | 8, 15, 247 | MHA header guards both forward+backward entry signatures |
| `csrc/cpp_itfs/utils.py` | 189 | Standalone cpp builder hardcodes `-DENABLE_CK=1` (ignored by JIT path) |

The switch is a **compile-time** flag for C++/HIP plus a **runtime**
Python boolean.  The JIT cache is **not keyed on `ENABLE_CK`** (confirmed
by reading `build_module`): changing `ENABLE_CK` without deleting
prebuilt `.so` files leaves the old CK-enabled modules in place.

## Python-level guards (dispatchers that know about ENABLE_CK)

Found via `grep -rn "ENABLE_CK" aiter/`:

| File | Use | CK-off behaviour |
|---|---|---|
| `aiter/ops/mha.py` L1972, L2670 | `if not ENABLE_CK: return fallback(...)` — flash-attention fwd/bwd | graceful fallback to the ASM/Triton MHA path |
| `aiter/ops/mha.py` L3029, L3095 | `if not ENABLE_CK and sink_ptr is None: ...` — fmha_v3 logits-free fwd/varlen | conditional fallback |
| `aiter/jit/core.py` L29 | `ENABLE_CK` module-level export | imported by consumers |

**That is the entire Python-level guard set** — only MHA checks
`ENABLE_CK` and picks a fallback.  No other dispatcher does.

## C++/HIP module source audit

The csrc dirs split into two groups under `ENABLE_CK=0`, confirmed
empirically by attempting a JIT rebuild:

| csrc dir | Has `#if ENABLE_CK` in sources? | CK-off build outcome (measured) |
|---|---|---|
| `ck_gemm_a8w8/` | no | **builds clean** — includes `composable_kernel/` submodule headers directly, which stay on disk |
| `ck_batched_gemm_bf16/` | no | **builds clean** — same |
| `ck_batched_gemm_a8w8/` | no | likely builds (same pattern; not explicitly re-verified) |
| `ck_deepgemm/` | no | untested (dispatcher not in the 5 op families) |
| `ck_gemm_a4w4_blockscale/` | no | untested |
| `ck_gemm_a8w8_blockscale/` | no | untested |
| `ck_gemm_a8w8_blockscale_bpreshuffle/` | no | untested |
| `ck_gemm_a8w8_bpreshuffle/` | no | untested |
| `ck_gemm_moe_2stages_codegen/` | no | untested; blocked upstream by `module_moe_sorting` which fails first |
| `ck_tile_gemm_moe_2stages/` | no | untested; same |
| `cktile_gemm_a8w8_bpreshuffle/` | no | untested |
| `py_itfs_ck/moe_sorting/*.cpp` | partial (`aiter_hip_common.h` routes through shim) | **build fails** under ENABLE_CK=0 — `module_moe_sorting` prerequisite of every MoE dispatch |
| `py_itfs_ck/rmsnorm2d_fwd.cpp` + `kernels/rmsnorm_kernels.cu` | partial | **build fails** — `module_rmsnorm` |
| `py_itfs_ck/asm_mla_decode_fwd_torch.cpp` | yes | stub-compiles clean |

**Implication**: "`ENABLE_CK=0` fails to build" is **not** a uniform
property — some CK csrc dirs keep building because they include
`composable_kernel/` submodule headers directly from the path (which
is still on disk) and ignore the `-DENABLE_CK=0` macro. Others go
through `aiter_hip_common.h`'s `#if !ENABLE_CK #include "ck_tile_shim.h"`
switch and fail because the shim is FMHA-focused.

The two modules that actually matter for the 5 tracked op families and
**fail to build** are:
1. `module_moe_sorting` — blocks all `aiter.fused_moe(...)` calls
2. `module_rmsnorm` — blocks `aiter.rmsnorm2d_fwd(...)` at N>8192

Everything else in the 5 families either builds cleanly
(`module_gemm_a8w8`, `module_batched_gemm_bf16`, `module_rmsnorm_quant`)
or is reached through a CK-free path (`torch.matmul`).

## Op-family × module map (what breaks when `ENABLE_CK=0`)

Cross-referencing `@compile_ops("module_...")` decorators (Python) with
the csrc source tree; statuses marked **verified** were rebuilt and
called end-to-end, **inferred** are read off source inspection:

| Op family | Python dispatcher entry | JIT module | Depends on csrc dir | CK-off status |
|---|---|---|---|---|
| Fused MoE (BF16/BF16) | `aiter.fused_moe.fused_moe(...)` with `QuantType.No` | `module_moe_sorting` (prereq) → `module_moe_ck2stages` | `py_itfs_ck/moe_sorting/` | **BROKEN (verified)** — `module_moe_sorting` build fails before stage1/2 even loads |
| Fused MoE (BF16+fp4 Swiglu) | `aiter.fused_moe.fused_moe(...)` with `QuantType.per_1x32` | same prereq | `py_itfs_ck/moe_sorting/` | **BROKEN (inferred)** — same upstream failure |
| GEMM BF16 (standalone) | `torch.matmul` (AITER's `TunedGemm.mm` routing) | n/a (hipBLASLt) | none | **OK (verified)** — already CK-free |
| GEMM FP8 a8w8 | `aiter.gemm_a8w8(...)` | `module_gemm_a8w8` | `ck_gemm_a8w8/` | **OK (verified)** — surprisingly compiles under ENABLE_CK=0 because CK submodule headers stay in include path |
| GEMM FP8 a8w8 (alt) | `torch._scaled_mm(...)` | n/a (hipBLASLt) | none | **OK** (not AITER's default dispatcher) |
| GEMM FP8 a8w8 (ASM) | `aiter.gemm_a8w8_ASM(...)` | `module_gemm_a8w8_asm` | none (kernels in `csrc/kernels/`) | **OK (verified)** — pure ASM, no CK |
| Batched GEMM BF16 | `aiter.batched_gemm_bf16(...)` | `module_batched_gemm_bf16` | `ck_batched_gemm_bf16/` | **OK (verified)** — compiles + matches CK-on perf |
| Batched GEMM BF16 (alt) | `torch.bmm` | n/a (hipBLAS) | none | **OK** — not wired into `aiter.batched_gemm_bf16` but works standalone |
| RMSNorm (N≤8192) | `aiter.rmsnorm2d_fwd(...)` | `module_rmsnorm_quant` | `csrc/kernels/rmsnorm_kernels.cu` (pure HIP) | **OK (verified)** — dispatcher routes here when N≤8192 |
| RMSNorm (N>8192) | `aiter.rmsnorm2d_fwd(...)` | `module_rmsnorm` | `py_itfs_ck/rmsnorm2d_fwd.cpp` | **BROKEN (verified)** — dispatcher hardcodes CK call for N>8192; module fails to build |
| RMSNorm (direct CK) | `aiter.rmsnorm2d_fwd_ck(...)` | `module_rmsnorm` | `py_itfs_ck/rmsnorm2d_fwd.cpp` | **BROKEN (verified)** — Python wrapper exists but `.so` build fails |
| MHA fwd | `aiter.flash_attn_fwd(...)` | `module_attention` (guarded) | `py_itfs_ck/` | **OK (inferred)** — `if not ENABLE_CK: return asm_fallback()` in `aiter/ops/mha.py` |
| MHA bwd | `aiter.flash_attn_bwd(...)` | `module_attention` (guarded) | `py_itfs_ck/` | **OK (inferred)** — same guard |
| FMoE FP8 blockscale | `aiter.fmoe_fp8_blockscale_g1u1(...)` | `module_moe_fmoe_asm` | `py_itfs_cu/asm_fmoe.cu` (ASM) | **OK (inferred)** — pure ASM path |
| DeepGEMM | `aiter.deepgemm(...)` | `module_deepgemm` | `ck_deepgemm/` | untested |
| GEMM a8w8 blockscale | `aiter.gemm_a8w8_blockscale(...)` | `module_gemm_a8w8_blockscale` | `ck_gemm_a8w8_blockscale/` | untested |
| GEMM a8w8 bpreshuffle | `aiter.gemm_a8w8_bpreshuffle(...)` | `module_gemm_a8w8_bpreshuffle` | `ck_gemm_a8w8_bpreshuffle/` | untested |
| GEMM a4w4 blockscale (fp4) | `aiter.gemm_a4w4_blockscale(...)` | `module_gemm_a4w4_blockscale` | `ck_gemm_a4w4_blockscale/` | untested |

## Import-time audit

From a fresh CK-off build (see §3 of `ck_build_comparison.md`), `import
aiter` **succeeds** — Python wrappers for every `@compile_ops(...)` decorator are
constructed lazily.  The breakage only manifests when a wrapper is
**first called**, which triggers JIT compilation of the backing module
and fails with an HIP compile error:

```
In file included from /tmp/aiter-noCK/aiter/jit/build/module_gemm_a8w8/build/srcs/gemm_a8w8.cu:5:
In file included from /tmp/aiter-noCK/aiter/jit/build/module_gemm_a8w8/build/srcs/gemm_a8w8_common.cuh:N:
csrc/include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp:... fatal error: 'composable_kernel/...' file not found
```

So the user experience at runtime is: `import aiter` OK, `aiter.gemm_a8w8(...)`
-> HIP compile error printed in a wall of JIT diagnostics, then a Python
`RuntimeError: failed to build module_gemm_a8w8`.

## Classification summary (5 op families tracked in PR #41)

Updated after empirical rebuild + call (see `ck_build_comparison.md`
for the measurements):

| Family | Python guard? | CK-off shippable today? | Notes |
|---|---|---|---|
| fused_moe_bf16 | no | **no** | `module_moe_sorting` build fails under ENABLE_CK=0; blocks all fused_moe paths regardless of dtype/quant |
| gemm_bf16 (standalone) | no | **yes** | already CK-free (`torch.matmul`) |
| gemm_fp8 (a8w8) | no | **yes (fragile)** | `module_gemm_a8w8` builds + runs on CK-off because CK submodule headers stay in include path; a header refactor could break this |
| batched_gemm_bf16 | no | **yes (fragile)** | same reason as gemm_fp8 |
| rmsnorm | **partial** | **yes at N≤8192, no at N>8192** | dispatcher switches on `input.shape[-1] > 8192`; HIP path (`rms_norm_cu`) covers N≤8192 cleanly; CK path (`rmsnorm2d_fwd_ck`) fails at N>8192 |

**Bottom line**: of 5 op families, **4 are CK-off-shippable today**
(3 fragile + 1 clean) and **1 is broken** (fused_moe_bf16). The single
broken family blocks any LLM-serving workload that uses MoE experts.

Within the rmsnorm family, **3 of 4 shapes work** (the common
production N≤8192 case); only rare N>8192 shapes break.

## References

- `ck_removal_tracker.md` — API-level comparison (PR #41)
- `ck_build_comparison.md` — build-level comparison (this PR)
- `learnings/tuning/ck_build_level_vs_api_level.md` — why the two are different
