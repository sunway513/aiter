# b_to_lds=True is the dominant BF16 HGEMM knob on MI355X

- **Area**: gemm
- **Kernel**: `aiter.ops.flydsl.flydsl_hgemm` (BF16 square matmul)
- **Shape**: M=N=K ∈ {1024, 2048, 8192, 16384}
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
The ARGUS paper (arxiv 2604.18616) reported AITER BF16 GEMM at 0.62× geomean vs ARGUS on MI300X, suggesting significant tuning headroom. We suspected the default b-matrix staging path (VGPR, `b_preshuffle=True`) is suboptimal and that routing B through LDS (`b_to_lds=True` + `b_preshuffle=False`) would increase reuse, reduce GMEM pressure, and improve throughput on MI355X.

## Result
| Shape | flydsl_default | tuned w/o b_lds | **tuned w/ b_lds** | Δ vs default |
|---|---|---|---|---|
| 1024  | 100.6 | 105.4  | **176.4** | **+75%** |
| 2048  | 469.8 | 556.5  | **643.8** | **+37%** |
| 8192  | 981.4 | 976.3  | **1116.6**| **+14%** |
| 16384 | 894.9 | 887.7  | **997.1** | **+11%** |

Only M=4096 preferred a different winner (128×256×64 w1×4 async, no b_lds → 1116 TFLOPS).

## Root cause
`b_to_lds=True` stages B tiles through LDS instead of registers. When paired with `async_copy=True`, the GMEM → LDS load overlaps with MFMA issue and the A tile already in LDS. This:
1. Cuts VGPR pressure (B no longer needs register staging), freeing space for larger accumulators.
2. Doubles B-tile reuse because the same LDS-staged tile is re-read across multiple A rows in the same workgroup.
3. Enables `async_copy` to hide the GMEM latency behind MFMA issue on the previous iteration.

`b_preshuffle=False` is required because shuffle is a store-side permutation that only applies when B is register-resident; when B goes through LDS, the shuffle path is a no-op but currently conflicts with the kernel config check.

## Reusable rule
**On gfx950 BF16 HGEMM, default to `b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False, async_copy=True`** unless the tile shape spills LDS (tile_k ≥ 128 with tile_m×tile_n ≥ 256×128 overflows). For shapes where this combo fits, the resulting config beats the `b_preshuffle=True` default by 11-75% across the token range.

Corollary for MoE work: the same principle likely applies to FP8/MXFP4 stage1 GEMMs in `moe_gemm_2stage.py` — future sweeps should include `b_to_lds=True` as a first-class candidate, not just a fallback.

## References
- Raw data: `results/gemm_bf16/results_round1.jsonl`, `results_round3.jsonl`, `results_round4.jsonl`
- Summary: `results/gemm_bf16/summary.md`
- Bench script: `scripts/bench_gemm_bf16_v3.py`
- Paper: Mai et al., "ARGUS: Agentic GPU Optimization Guided by Data-Flow Invariants", arxiv 2604.18616
- Task spec: `tasks/gemm_bf16_argus_replica.yaml`
