"""Minimal AVO optimization loop: target -> baseline -> optimize -> commit best."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .evolution.controller import EvolutionController
from .evolution.scoring import BenchmarkResult, ScoringFunction
from .knowledge.base import KnowledgeBase
from .lineage.store import LineageStore
from .remote import RemoteRunner
from .validation import (
    validate_target_schema,
    validate_gpu_env,
    validate_benchmark_result,
    check_all,
)


def load_local_env(target_dir: Path) -> None:
    """Load targets/local.env (KEY=VALUE lines) into os.environ.

    Searches for local.env in target_dir, then in target_dir's parent
    (targets/local.env). Skips blank lines and comments (#).
    Does NOT override variables already set in the environment.
    """
    candidates = [target_dir / "local.env", target_dir.parent / "local.env"]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value
            return


def load_target(target_dir: Path) -> dict:
    """Load target.yaml and resolve env vars."""
    cfg = yaml.safe_load((target_dir / "target.yaml").read_text())
    return cfg


def resolve_env(s: str) -> str:
    """Expand $VAR references in a string."""
    return os.path.expandvars(s)


def run_command(cmd: str, label: str) -> tuple[int, str]:
    """Run a shell command, return (returncode, stdout+stderr)."""
    print(f"[{label}] Running: {cmd}", file=sys.stderr)
    result = subprocess.run(
        resolve_env(cmd), shell=True, capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    print(f"[{label}] Exit code: {result.returncode}", file=sys.stderr)
    return result.returncode, output


def run_correctness(target: dict) -> bool:
    """Run correctness harness. Returns True if passed."""
    corr = target.get("correctness", {})
    cmd = corr.get("command")
    if not cmd:
        print("[correctness] No correctness command, skipping", file=sys.stderr)
        return True
    rc, output = run_command(cmd, "correctness")
    pass_pattern = corr.get("pass_pattern", "test passed")
    passed = pass_pattern.lower() in output.lower() and rc == 0
    if not passed:
        print(f"[correctness] FAILED. Output:\n{output[:500]}", file=sys.stderr)
    return passed


def run_benchmark(target: dict, scoring: ScoringFunction) -> list:
    """Run benchmark for each shape, return list of BenchmarkResults."""
    bench = target.get("benchmark", {})
    cmd_template = bench.get("command", "")
    shapes = bench.get("shapes", [{}])
    results = []
    for shape in shapes:
        cmd = cmd_template.format(**shape) if shape else cmd_template
        rc, output = run_command(cmd, f"benchmark shape={shape}")
        if rc != 0:
            print(f"[benchmark] Command failed: {output[:300]}", file=sys.stderr)
            continue
        parsed = scoring.parse(output)
        if parsed.valid:
            results.append(parsed)
        else:
            print(f"[benchmark] Could not parse metrics from output", file=sys.stderr)
    return results


def main(
    target_dir: str = "targets/aiter_mha",
    output_dir: str = "optimization_logs",
    max_rounds: int = 1,
    resume: bool = False,
    auto: bool = False,
    gpu_id: int | None = None,
    seed: int = 0,
    config: "ForgeConfig | None" = None,
):
    """Run the mini AVO loop."""
    target_path = Path(target_dir)
    out_path = Path(output_dir)

    # Load config: forge.yaml (new) or local.env (deprecated fallback)
    if config is not None:
        if config.gpu.aiter_root and "AITER_ROOT" not in os.environ:
            os.environ["AITER_ROOT"] = config.gpu.aiter_root
    else:
        load_local_env(target_path)

    # Fresh run by default: wipe old lineage to prevent stale data mixing in.
    # Use --resume to continue a previous run.
    lineage_dir = out_path / "lineage"
    if not resume and lineage_dir.exists():
        shutil.rmtree(lineage_dir)

    out_path.mkdir(parents=True, exist_ok=True)

    target = load_target(target_path)

    # Validate required fields
    required = [
        ("name", target.get("name")),
        ("kernel.path", target.get("kernel", {}).get("path")),
        ("correctness.command", target.get("correctness", {}).get("command")),
        ("benchmark.command", target.get("benchmark", {}).get("command")),
        ("benchmark.shapes", target.get("benchmark", {}).get("shapes")),
        ("scoring.primary_metric", target.get("scoring", {}).get("primary_metric")),
        ("benchmark.aggregate", target.get("benchmark", {}).get("aggregate")),
    ]
    for field_name, value in required:
        if not value:
            print(f"ERROR: target.yaml missing required field: {field_name}", file=sys.stderr)
            sys.exit(1)

    # Validate target schema (validation block)
    schema_results = validate_target_schema(target)
    if not check_all(schema_results):
        sys.exit(1)

    print(f"Target: {target['name']}", file=sys.stderr)

    # Initialize components
    scoring_cfg = target.get("scoring", {})
    scoring = ScoringFunction(
        primary_metric=scoring_cfg.get("primary_metric", "tflops"),
        higher_is_better=scoring_cfg.get("higher_is_better", True),
    )
    knowledge_dir = Path(__file__).parent / "knowledge" / "patterns"
    kb = KnowledgeBase(knowledge_dir)
    store = LineageStore(out_path / "lineage")
    store.load()

    kernel_path = target.get("kernel", {}).get("path", "")
    ctrl = EvolutionController(store, kb, scoring, kernel_path=kernel_path)

    # Auto-mode components
    llm_editor = None
    patch_manager = None
    if auto:
        from .llm_editor import LLMEditor
        from .patch_manager import PatchManager

        if config and config.llm:
            model = config.llm.model
        else:
            model = os.environ.get("AITER_FORGE_MODEL", "claude-sonnet-4-20250514")
        llm_editor = LLMEditor(model=model)

        if config:
            host = config.gpu.host
            user = config.gpu.user
            runner = RemoteRunner(
                host=host, user=user, gpu_id=gpu_id,
                key_filename=config.gpu.ssh_key or None,
                jump_host=config.gpu.jump_host or None,
            )
        else:
            hw = target.get("hardware", {})
            host = hw.get("host") or os.environ.get("REMOTE_HOST", "localhost")
            user = hw.get("user") or os.environ.get("REMOTE_USER", "")
            runner = RemoteRunner(host=host, user=user, gpu_id=gpu_id)

        remote_kernel_path = resolve_env(kernel_path)
        patch_manager = PatchManager(runner, remote_kernel_path=remote_kernel_path)

    # Validate GPU environment
    gpu_results = validate_gpu_env(target, parallel_mode=(gpu_id is not None))
    if not check_all(gpu_results):
        sys.exit(1)

    # Step 1: Correctness check
    if not run_correctness(target):
        print("ERROR: Baseline correctness check failed. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Baseline benchmark
    print("\n=== Baseline Benchmark ===", file=sys.stderr)
    baseline_results = run_benchmark(target, scoring)
    if not baseline_results:
        print("ERROR: No valid baseline results. Aborting.", file=sys.stderr)
        sys.exit(1)

    aggregate = target.get("benchmark", {}).get("aggregate", "last")
    if aggregate == "geomean" and len(baseline_results) > 1:
        baseline = scoring.aggregate_geomean(baseline_results)
    else:
        baseline = baseline_results[-1]

    baseline_variant = ctrl.commit_variant(
        round_num=0, parent_id=None,
        patch_path="baseline", result=baseline,
        description="original kernel", strategy="baseline",
    )
    metric_val = baseline.metrics.get(scoring.primary_metric, "N/A")
    print(f"\nBaseline: {scoring.primary_metric}={metric_val}", file=sys.stderr)

    # Step 3: Optimization rounds
    current_best_id = baseline_variant.variant_id
    for round_num in range(1, max_rounds + 1):
        print(f"\n=== Round {round_num}/{max_rounds} ===", file=sys.stderr)

        # Generate optimization prompt
        prompt = ctrl.generate_optimization_prompt(round_num, current_best_id)
        if seed > 0:
            prompt += f"\n\n## Strategy Seed: {seed}\nExplore a different optimization direction than seed 0. Vary your approach based on this seed number."
        prompt_path = out_path / f"round_{round_num}_prompt.md"
        prompt_path.write_text(prompt)
        print(f"Optimization prompt saved to: {prompt_path}", file=sys.stderr)

        if auto:
            # Read current kernel from remote
            try:
                kernel_source = patch_manager._read_remote()
            except Exception as exc:
                print(f"Round {round_num}: Failed to read remote kernel: {exc}. Skipping.", file=sys.stderr)
                continue
            # Generate edit via LLM
            edit_result = llm_editor.generate_edit(kernel_source, prompt)
            if not edit_result.success or not edit_result.has_changes:
                print(f"Round {round_num}: LLM produced no changes. Skipping.", file=sys.stderr)
                continue
            # Apply patch to remote
            patch_manager.apply(edit_result.modified)
            # Save patch file
            patch_manager.save_patch(str(out_path / f"round_{round_num}.patch"))
        else:
            print(">>> Human-in-the-loop: apply the edit (manually or feed prompt to LLM), then press Enter <<<", file=sys.stderr)
            input()  # Phase 1: human gate

        # Correctness gate
        if not run_correctness(target):
            if auto:
                patch_manager.rollback()
            print(f"Round {round_num}: FAILED correctness. Reverting.", file=sys.stderr)
            continue

        # Benchmark
        round_results = run_benchmark(target, scoring)
        if not round_results:
            print(f"Round {round_num}: No valid results.", file=sys.stderr)
            continue

        if aggregate == "geomean" and len(round_results) > 1:
            round_result = scoring.aggregate_geomean(round_results)
        else:
            round_result = round_results[-1]

        # Post-benchmark validation
        val_block = target.get("validation", {})
        compute_prec = val_block.get("precision", {}).get("compute")
        result_metrics = dict(round_result.metrics)
        if compute_prec:
            result_metrics["precision"] = compute_prec

        baseline_metrics = None
        if baseline:
            baseline_metrics = dict(baseline.metrics)
            if compute_prec:
                baseline_metrics["precision"] = compute_prec

        bench_val_results = validate_benchmark_result(
            result_metrics, target, baseline_result=baseline_metrics,
        )
        if not check_all(bench_val_results):
            print("[VALIDATION] Benchmark result failed validation — skipping this round", file=sys.stderr)
            continue

        # Try commit
        committed = ctrl.try_commit(
            round_num=round_num,
            parent_id=current_best_id,
            patch_path=str(out_path / f"round_{round_num}.patch"),
            result=round_result,
            description=f"round {round_num} optimization",
            strategy="llm_suggested",
            correct=True,
        )

        rv = round_result.metrics.get(scoring.primary_metric, 0)
        if committed:
            if auto and patch_manager.has_backup:
                # Accept this kernel as new baseline for future rollbacks.
                patch_manager.accept()
            speedup = scoring.speedup(round_result, baseline)
            print(f"Round {round_num}: COMMITTED. {scoring.primary_metric}={rv:.2f} "
                  f"(speedup={speedup:.3f}x vs baseline)", file=sys.stderr)
            current_best_id = committed.variant_id
        else:
            if auto:
                patch_manager.rollback()
            print(f"Round {round_num}: REJECTED (no improvement). {scoring.primary_metric}={rv:.2f}", file=sys.stderr)

    # Summary
    best = store.best(scoring.primary_metric, scoring.higher_is_better)
    if best:
        best_result = BenchmarkResult(metrics=best.metrics, valid=True)
        speedup = scoring.speedup(best_result, baseline) if baseline.valid else 0
        print(f"\n=== Final Result ===", file=sys.stderr)
        print(f"Best: {best.variant_id} ({scoring.primary_metric}={best.metrics.get(scoring.primary_metric)})", file=sys.stderr)
        print(f"Total variants committed: {len(store.all_variants())}", file=sys.stderr)

    # Save report
    report = {
        "target": target["name"],
        "baseline": baseline.metrics,
        "best": best.to_dict() if best else None,
        "total_rounds": max_rounds,
        "committed_variants": len(store.all_variants()),
    }
    val_block = target.get("validation")
    if val_block:
        report["validation"] = {
            "precision": val_block.get("precision", {}),
            "comparison": val_block.get("comparison", {}),
        }
    (out_path / "report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AITER-Forge mini optimization loop")
    parser.add_argument("--target", default="targets/aiter_mha", help="Target directory")
    parser.add_argument("--output", default="optimization_logs", help="Output directory")
    parser.add_argument("--rounds", type=int, default=1, help="Max optimization rounds")
    parser.add_argument("--resume", action="store_true", help="Resume from existing lineage instead of fresh run")
    parser.add_argument("--auto", action="store_true", help="Autonomous mode: LLM generates edits, auto apply/rollback")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID for HIP_VISIBLE_DEVICES isolation")
    parser.add_argument("--seed", type=int, default=0, help="Strategy seed for prompt diversity in parallel runs")
    args = parser.parse_args()
    main(target_dir=args.target, output_dir=args.output, max_rounds=args.rounds,
         resume=args.resume, auto=args.auto, gpu_id=args.gpu_id, seed=args.seed)
