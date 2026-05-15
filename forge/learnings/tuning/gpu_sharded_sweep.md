# Shard the search space across idle GPUs, don't serialize

- **Area**: tuning
- **Kernel**: N/A (applies to any sweep harness)
- **Shape**: N/A (applies to any shape)
- **Date**: 2026-04-22
- **Confidence**: verified

## Hypothesis
The obvious parallelization of a tuning sweep is one shape per GPU. That leaves you GPU-limited at the *slowest* shape — in BF16 GEMM on MI355X, M=16384 takes ~50× longer per config than M=1024 because of the 256× larger matmul work. On an 8-GPU node, running 5 shape-parallel workers leaves 3 GPUs idle while the slowest shape bottlenecks the whole sweep.

## Result
For M=16384 on MI355X, 250,560 config enumeration would take ~5 hours on one GPU. Sharded across 5 GPUs by `tile_m`:

| GPU | tile_m shard | Configs | Observed early best |
|---|---|---|---|
| 4 (exhaustive, pre-existing) | all | 250,560 | 419 TFLOPS |
| 2 | {16, 32} | 73,728 | 185 TFLOPS |
| 5 | {48, 64} | 67,584 | (just starting) |
| 6 | {80, 96, 112} | — | 315 TFLOPS |
| 7 | **{128, 160, 256}** | 107,520 | **616 TFLOPS @ 10s** |

The large-tile shard on GPU 7 hit 616 TFLOPS in **10 seconds** — faster than the exhaustive sweep reached in its first 25 minutes on GPU 4. The `tile_m=128` region is where prior winners live; sharding exposed it immediately.

## Root cause
Sweep iteration order matters. `itertools.product(TILE_M, TILE_N, TILE_K)` iterates `tile_m=16` over all (tile_n, tile_k) first, then `tile_m=32`, and so on. If prior knowledge says the winners are at `tile_m>=128`, the first ~60% of the sweep produces no useful new information.

Sharding by `tile_m` into disjoint subsets:
1. Assigns the "likely winner" shard to its own GPU — it finishes quickly, gives signal fast.
2. Assigns "exhaustive coverage" shards (small tiles) to other GPUs — run in parallel, don't block.
3. Consumes all available GPUs, even when the workload doesn't have N independent shapes to fill them.

## Reusable rule
**Never leave GPUs idle during a sweep.** If the natural parallelism (e.g. shape) is narrower than the GPU count, shard the config space along an orthogonal axis:
- For GEMM/HGEMM tuning: shard by `tile_m`.
- For MoE tuning: shard by `(token_range, inter_dim)` or by `tile_k`.
- For attention tuning: shard by `(head_dim, seq_len)`.

Pattern:
```python
# Shard M=16384 across 5 GPUs by tile_m
for gpu, tm_list in enumerate([[16,32], [48,64], [80,96,112], [128,160], [256]]):
    launch_worker(gpu, shape=16384, tile_m_in=tm_list)
```

**Corollary — prioritize the "likely winner" shard on a fast GPU**: if prior learnings say `tile_m=128` dominates at this shape, put that shard first so the best-so-far metric moves quickly and you can early-abort dominated branches.

## Implementation in aiter-forge
This pattern is operationalized as `aiter_forge.dispatcher.dispatch_sharded(task_path, shard_key, n_shards, agent_runner, ...)` — a dispatcher-side helper that reads a task YAML, partitions its candidates by `shard_key` (e.g. `tile_m`), builds one sub-spec per shard, and runs all shards concurrently through the same `dispatch` pipeline (so the hypothesis hard-rule applies per shard). Shard failures are isolated; each shard's summary includes `shard_index`, `shard_candidates`, `status`, `best_tflops`, and `best_candidate_name`.

```python
from pathlib import Path
from aiter_forge.dispatcher import dispatch_sharded

results = dispatch_sharded(
    Path("tasks/gemm_bf16_m16384.yaml"),
    shard_key="tile_m",
    n_shards=5,
    agent_runner=my_runner,
)
# results[i] -> {shard_index, shard_candidates, status, best_tflops, ...}
```

Partitions are distributed across shards largest-first / round-robin on current shard size, so a skewed `shard_key` distribution still yields roughly balanced shards.

`scripts/bench_gemm_bf16_fullday_sharded.py` remains the shape-specific CLI variant that accepts `--tile-m-in=128,160,256`; `dispatch_sharded` is the generic library API any task YAML can use.

## References
- Implementation: `src/aiter_forge/dispatcher.py` (function `dispatch_sharded`).
- Tests: `tests/test_dispatcher.py::test_dispatch_sharded_*`.
- Shape-specific CLI variant: `scripts/bench_gemm_bf16_fullday_sharded.py`.
- Applied to M=16384 sweep on 2026-04-22 — gave ~5x wall-time speedup over single-GPU exhaustive.
