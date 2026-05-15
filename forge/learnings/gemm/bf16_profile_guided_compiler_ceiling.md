# Profile evidence — FlyDSL's large-shape gap to asm is compiler-level, not tile-level

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` vs `aiter.ops.gemm_op_a16w16.gemm_a16w16_asm`
- **Shape**: M=N=K=8192 (profile target; same pattern at M=4096 and 16384)
- **Date**: 2026-04-22
- **Confidence**: verified via rocprofv3

## Hypothesis
After ~6200 tuning configs, FlyDSL tuned topped out at 1131 TFLOPS at M=8192 (vs asm's 1591 after we fixed the asm baseline to use `bpreshuffle=True` + `splitK`). We suspected this was a *tile-size* limitation — asm uses 256×256 output tiles while our FlyDSL winners used 128×128. We tested the hypothesis by forcing FlyDSL to 256×256 with every valid warp layout.

## Result

**At M=4096, FlyDSL with tile 256×256 gets 1020 TFLOPS — LOWER than FlyDSL with 128×128 (1203).**

| Variant | Tile | TFLOPS at M=4096 |
|---|---|---|
| FlyDSL 128×128 w2×2 + hidden knobs | 128×128×64 | **1203** |
| FlyDSL 256×256 w2×2 | 256×256×32 | 1022 |
| FlyDSL 256×128 w4×1 + hidden knobs | 256×128×64 | 1035 |
| aiter_asm 256×256_bpreshuffle splitK=8 | 256×256×?  | **1453** |

rocprofv3 counters at M=8192 (20-run avg, 23 counter rows each):

| Metric | FlyDSL tuned (128×128) | aiter_asm (256×256_bpre+splitK) |
|---|---|---|
| Kernel name | `hgemm_bf16_128x128x64_S2TN_AS_BS_0` | `bf16gemm_bf16_tn_256x256_bpreshuffle` |
| **VGPR per wave** | **96** | **256** |
| **LDS per WG** | **65 KB** | **164 KB** |
| **SQ_WAVES** | 16,384 | **4,096** (4× fewer) |
| SQ_INSTS_VALU | 101M | 70M (1.4× fewer) |
| Avg kernel duration | 1068 μs | **769 μs** |
| Effective TFLOPS | 1030 | **1430** |

## Root cause
The asm kernel **deliberately uses 256 VGPRs per wave** to hold large accumulators for a 256×256 output tile. This gives it:
- 1 wave/SIMD occupancy (the maximum for VGPR ≥ ~200)
- 4× larger per-WG workload → 4× fewer workgroups → 4× less kernel-launch + sync overhead
- Per-tile arithmetic intensity 4× higher

FlyDSL's compiler refuses to use more than ~96 VGPRs for the same tile shape because its register allocator targets 2 waves/SIMD occupancy. Attempting to force 256×256 tile in FlyDSL:
- Register pressure exceeds the 2w/SIMD budget → VGPR spill or occupancy drop
- Result: 1022 TFLOPS — **slower** than the 128×128 config despite the larger tile

This is the compiler-vs-asm gap the HipKittens paper (arxiv 2511.08083) documents: "AMD kernels resort to raw assembly to finely interleave instruction issues." Their fix is an entirely new DSL with 8-wave ping-pong patterns. FlyDSL's current dispatch layer does not expose this.

## Reusable rule
**At M ≤ 2048 (hgemm) or M ≤ 1024 (small_m), FlyDSL's compiler-generated code can match or beat asm** — because the output tile is small enough that the 96-VGPR 2w/SIMD path gives higher occupancy than asm's 256-VGPR 1w/SIMD path.

**At M ≥ 4096, FlyDSL's default tuning surface cannot close the gap without a compiler change** that:
1. Allows explicit 1w/SIMD VGPR target (≥200 VGPR per wave), OR
2. Emits 8-wave ping-pong schedules (HipKittens pattern), OR
3. Generates deeper software pipelines (asm has `pf3_splitk` = depth-3 prefetch + split-K).

**Corollary for tuning agents**: when your FlyDSL (or any compiler DSL) config saturates ~30% below the hand-tuned asm reference on BF16 GEMM at large shapes, don't waste further sweeps on tile geometry — profile once, confirm the VGPR gap, and either accept the small/mid-shape wins or switch to a different codegen path (HipKittens, CK, raw asm).

## References
- Raw rocprofv3 CSVs: `results/gemm_bf16/rocprof_flydsl_csv_counter_collection.csv`, `rocprof_asm_csv_counter_collection.csv`
- Profile parser: `scripts/parse_rocprof.py`
- HipKittens paper: arxiv 2511.08083 (Mai et al.) — proposes 8-wave ping-pong as the AMD-native analogue of NVIDIA wave specialization.
- Related learning: `learnings/gemm/bf16_small_m_kernel_family.md` (where FlyDSL *does* win at M=1024).
