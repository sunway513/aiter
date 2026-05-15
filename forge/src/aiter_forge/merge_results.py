"""Merge lineage results from parallel GPU optimization runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aiter_forge.validation import validate_merge_consistency, check_all


def merge_lineages(
    output_dirs: list[str],
    metric: str = "tflops",
    higher_is_better: bool = True,
) -> dict:
    """Merge lineage from N parallel runs, return best variant and merged report.

    Each output_dir should contain report.json and lineage/lineage.json.
    """
    all_variants = []
    reports = []

    for d in output_dirs:
        d = Path(d)
        report_path = d / "report.json"
        lineage_path = d / "lineage" / "lineage.json"

        if not report_path.exists():
            print(f"[merge] Skipping {d}: no report.json", file=sys.stderr)
            continue

        report = json.loads(report_path.read_text())
        report["source_dir"] = str(d)
        reports.append(report)

        if lineage_path.exists():
            variants = json.loads(lineage_path.read_text())
            for v in variants:
                v["source_dir"] = str(d)
            all_variants.extend(variants)

    if not reports:
        return {"error": "No valid reports found", "best": None, "total_variants": 0}

    # Validate merge consistency
    merge_val_results = validate_merge_consistency(reports)
    if not check_all(merge_val_results):
        return {"error": "Merge validation failed", "best": None, "total_variants": 0}

    # Find best variant across all runs
    best = None
    for v in all_variants:
        val = v.get("metrics", {}).get(metric)
        if val is None:
            continue
        if best is None:
            best = v
        else:
            best_val = best.get("metrics", {}).get(metric, 0)
            if (higher_is_better and val > best_val) or (not higher_is_better and val < best_val):
                best = v

    merged = {
        "total_runs": len(reports),
        "total_variants": len(all_variants),
        "metric": metric,
        "higher_is_better": higher_is_better,
        "best": best,
        "per_run": [
            {
                "source_dir": r["source_dir"],
                "target": r.get("target"),
                "committed_variants": r.get("committed_variants", 0),
                "baseline": r.get("baseline"),
                "best": r.get("best"),
            }
            for r in reports
        ],
    }
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge parallel optimization results")
    parser.add_argument("--dirs", nargs="+", required=True, help="Output directories from parallel runs")
    parser.add_argument("--metric", default="tflops", help="Metric to rank by")
    parser.add_argument("--output", default="merged_report.json", help="Output file path")
    parser.add_argument("--lower-is-better", action="store_true", help="Lower metric is better")
    args = parser.parse_args()

    result = merge_lineages(args.dirs, metric=args.metric, higher_is_better=not args.lower_is_better)
    Path(args.output).write_text(json.dumps(result, indent=2, default=str))
    print(f"Merged {result['total_runs']} runs, {result['total_variants']} variants", file=sys.stderr)
    if result.get("best"):
        best_val = result["best"].get("metrics", {}).get(args.metric)
        print(f"Best: {result['best'].get('variant_id')} ({args.metric}={best_val})", file=sys.stderr)
