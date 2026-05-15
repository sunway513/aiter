"""CLI entry point: aiter-forge run."""
from __future__ import annotations

import argparse
import sys

from .config import load


def main():
    parser = argparse.ArgumentParser(
        prog="aiter-forge",
        description="AITER-Forge: kernel optimization harness for MI355X",
    )
    sub = parser.add_subparsers(dest="command")

    # aiter-forge run
    run_p = sub.add_parser("run", help="Run optimization loop")
    run_p.add_argument("--target", default="targets/aiter_mha", help="Target directory")
    run_p.add_argument("--output", default="optimization_logs", help="Output directory")
    run_p.add_argument("--rounds", type=int, default=1, help="Max optimization rounds")
    run_p.add_argument("--resume", action="store_true", help="Resume from existing lineage")
    run_p.add_argument(
        "--mode", choices=["benchmark", "manual", "auto"],
        help="Override auto-detected mode",
    )
    run_p.add_argument("--gpu-id", type=int, default=None, help="GPU ID")
    run_p.add_argument("--seed", type=int, default=0, help="Strategy seed")
    run_p.add_argument("--config", default="forge.yaml", help="Config file path")
    run_p.add_argument("--host", help="Override gpu.host")
    run_p.add_argument("--user", help="Override gpu.user")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _handle_run(args)


def _handle_run(args):
    """Handle 'aiter-forge run' command."""
    cli_overrides = {}
    if args.host:
        cli_overrides["host"] = args.host
    if args.user:
        cli_overrides["user"] = args.user

    cfg = load(config_path=args.config, cli_overrides=cli_overrides or None)

    # Determine mode
    mode = args.mode or cfg.mode
    auto = mode == "auto"

    if mode == "auto" and cfg.mode != "auto":
        print(
            f"ERROR: --mode auto requires LLM API key. "
            f"Set {cfg.llm.api_key_env if cfg.llm else 'ANTHROPIC_API_KEY'} env var.",
            file=sys.stderr,
        )
        sys.exit(1)

    if mode == "benchmark":
        max_rounds = 0
    else:
        max_rounds = args.rounds

    from .mini_loop import main as run_loop

    run_loop(
        target_dir=args.target,
        output_dir=args.output,
        max_rounds=max_rounds,
        resume=args.resume,
        auto=auto,
        gpu_id=args.gpu_id,
        seed=args.seed,
        config=cfg,
    )
