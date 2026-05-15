"""Task dispatcher: the single entry point for running tasks-as-YAML.

A P9 orchestrator writes a ``tasks/<id>.yaml``. The dispatcher:
  1. Parses + validates the spec via predict_verify (rejects if hypothesis missing).
  2. Renders an agent-ready prompt that includes the hypothesis.
  3. Spawns the agent via an injectable ``agent_runner`` (production: Claude
     Agent SDK; tests: a mock).
  4. Invokes an optional ``extractor`` callback on success so learning-extractor
     can append the learnings entry synchronously.

``dispatch_all`` runs many tasks concurrently on a thread pool. Task failures
are isolated: one RuntimeError does not poison the batch.

``dispatch_sharded`` splits a single task's candidate list across N worker
shards along an orthogonal axis (e.g. ``tile_m``), operationalizing the
"never leave GPUs idle during a sweep" rule from
``learnings/tuning/gpu_sharded_sweep.md``.
"""
from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import yaml

from .predict_verify import TaskSpec, parse

AgentRunner = Callable[..., dict[str, Any]]
Extractor = Callable[[TaskSpec, dict[str, Any]], None]


def render_prompt(spec: TaskSpec) -> str:
    """Build the agent prompt from a task spec.

    Must include the hypothesis verbatim — tests check that an agent runner
    sees the hypothesis string, which prevents a buggy renderer from silently
    stripping it.
    """
    lines = [
        f"# Task: {spec.id}",
        "",
        "## Hypothesis",
        spec.hypothesis,
        "",
        "## Candidates",
    ]
    for c in spec.candidates:
        params = c.get("params", {})
        params_str = ", ".join(f"{k}={v}" for k, v in params.items()) or "(none)"
        lines.append(f"- {c['name']}: {params_str}")
    lines += [
        "",
        "## Verify",
        f"- cmd: {spec.verify.get('cmd', '')}",
    ]
    if spec.verify.get("parse"):
        lines.append(f"- parse: {spec.verify['parse']}")
    lines += [
        "",
        "## Finalize",
        f"- learning_path: {spec.finalize.get('learning_path', '')}",
    ]
    return "\n".join(lines)


def dispatch(
    task_path: Path,
    *,
    agent_runner: AgentRunner,
    extractor: Extractor | None = None,
) -> dict[str, Any]:
    spec, errors = parse(Path(task_path))
    if spec is None:
        raise ValueError(
            "task spec invalid: " + "; ".join(f"{e.field}: {e.detail}" for e in errors)
        )
    prompt = render_prompt(spec)
    result = agent_runner(prompt, task_id=spec.id)
    if not isinstance(result, dict):
        result = {"status": "completed", "raw": result}
    result.setdefault("task_id", spec.id)
    if extractor is not None and result.get("status") == "completed":
        extractor(spec, result)
    return result


def dispatch_all(
    task_paths: list[Path],
    *,
    agent_runner: AgentRunner,
    max_workers: int = 4,
    extractor: Extractor | None = None,
) -> list[dict[str, Any]]:
    results_by_path: dict[Path, dict[str, Any]] = {}

    def worker(path: Path) -> dict[str, Any]:
        try:
            return dispatch(path, agent_runner=agent_runner, extractor=extractor)
        except Exception as exc:  # noqa: BLE001  # isolation boundary
            return {
                "task_id": Path(path).stem,
                "status": "failed",
                "error": repr(exc),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_path = {pool.submit(worker, Path(p)): Path(p) for p in task_paths}
        for fut in as_completed(future_to_path):
            path = future_to_path[fut]
            results_by_path[path] = fut.result()

    return [results_by_path[Path(p)] for p in task_paths]


def _partition_by_shard_key(
    candidates: list[dict[str, Any]], shard_key: str
) -> dict[Any, list[dict[str, Any]]]:
    """Group candidates by the value of ``params[shard_key]``.

    Raises ValueError with an actionable message if no candidate carries the
    requested key — the caller can't shard a sweep along an axis that isn't
    there.
    """
    buckets: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    seen_any = False
    for cand in candidates:
        params = cand.get("params", {}) or {}
        if shard_key in params:
            seen_any = True
            buckets[params[shard_key]].append(cand)
    if not seen_any:
        available = sorted({
            k for c in candidates for k in (c.get("params", {}) or {}).keys()
        })
        raise ValueError(
            f"shard_key {shard_key!r} not found in any candidate's params; "
            f"available params: {available or '(none)'}"
        )
    return dict(buckets)


def _assign_partitions_to_shards(
    buckets: dict[Any, list[dict[str, Any]]], n_shards: int
) -> list[list[dict[str, Any]]]:
    """Round-robin partitions (largest first) across ``n_shards`` shards.

    Largest-first keeps shard sizes roughly balanced even when partition sizes
    vary, a common case when ``shard_key`` has a skewed value distribution.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n_shards)]
    ordered = sorted(
        buckets.items(),
        key=lambda kv: (-len(kv[1]), str(kv[0])),
    )
    sizes = Counter({i: 0 for i in range(n_shards)})
    for _key, cands in ordered:
        target, _ = min(sizes.items(), key=lambda kv: (kv[1], kv[0]))
        shards[target].extend(cands)
        sizes[target] += len(cands)
    return shards


def _write_shard_spec(
    spec: TaskSpec, shard_index: int, shard_cands: list[dict[str, Any]], tmp_dir: Path
) -> Path:
    """Serialize a per-shard sub-spec to a YAML file inside ``tmp_dir``.

    We go through YAML (rather than constructing a TaskSpec in memory and
    skipping parse) so every shard re-enters the same
    ``dispatch`` -> ``parse`` pipeline — the hypothesis hard-rule keeps
    applying even when the caller only specified it once at the top level.
    """
    doc: dict[str, Any] = {
        "id": f"{spec.id}__shard{shard_index:02d}",
        "hypothesis": spec.hypothesis,
        "candidates": shard_cands,
        "verify": dict(spec.verify),
        "finalize": dict(spec.finalize),
    }
    path = tmp_dir / f"{doc['id']}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False))
    return path


def _summarize_shard(
    shard_index: int,
    shard_cands: list[dict[str, Any]],
    result: dict[str, Any] | None,
    error: BaseException | None,
) -> dict[str, Any]:
    """Build the per-shard summary row returned by ``dispatch_sharded``."""
    summary: dict[str, Any] = {
        "shard_index": shard_index,
        "shard_candidates": [c.get("name") for c in shard_cands],
    }
    if error is not None:
        summary["status"] = "failed"
        summary["error"] = repr(error)
        summary["best_tflops"] = None
        summary["best_candidate_name"] = None
        summary["extractor_state"] = None
        return summary
    assert result is not None  # noqa: S101  # control-flow contract
    summary["status"] = result.get("status", "completed")
    summary["best_tflops"] = result.get("best_tflops")
    summary["best_candidate_name"] = result.get("best_candidate_name")
    summary["extractor_state"] = result.get("extractor_state")
    summary["raw"] = result
    return summary


def dispatch_sharded(
    task_path: Path,
    *,
    shard_key: str,
    n_shards: int,
    agent_runner: AgentRunner,
    extractor: Extractor | None = None,
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Split the task's ``shard_key`` candidates across ``n_shards`` workers.

    Reads the task YAML, partitions the candidates by ``shard_key``, builds an
    in-memory sub-spec per shard, and runs all shards concurrently via the
    same ``dispatch`` pipeline (so the hypothesis hard-rule applies per
    shard). One shard erroring does not poison the batch.

    Returns one result dict per shard:
        ``[{shard_index, shard_candidates, status, best_tflops,
           best_candidate_name, extractor_state}, ...]``
    """
    spec, errors = parse(Path(task_path))
    if spec is None:
        raise ValueError(
            "task spec invalid: " + "; ".join(f"{e.field}: {e.detail}" for e in errors)
        )
    buckets = _partition_by_shard_key(spec.candidates, shard_key)
    shards = _assign_partitions_to_shards(buckets, n_shards)
    workers = max_workers if max_workers is not None else n_shards

    with tempfile.TemporaryDirectory(prefix=f"{spec.id}-shards-") as tmp:
        tmp_dir = Path(tmp)
        shard_paths = [
            _write_shard_spec(spec, idx, cands, tmp_dir)
            for idx, cands in enumerate(shards)
            if cands
        ]
        shard_indices = [idx for idx, cands in enumerate(shards) if cands]
        summaries_by_idx: dict[int, dict[str, Any]] = {}

        def worker(idx: int, path: Path) -> dict[str, Any]:
            try:
                res = dispatch(path, agent_runner=agent_runner, extractor=extractor)
                return _summarize_shard(idx, shards[idx], res, None)
            except Exception as exc:  # noqa: BLE001  # isolation boundary
                return _summarize_shard(idx, shards[idx], None, exc)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {
                pool.submit(worker, idx, path): idx
                for idx, path in zip(shard_indices, shard_paths)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                summaries_by_idx[idx] = fut.result()

    return [summaries_by_idx[idx] for idx in shard_indices]
