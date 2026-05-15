"""Merge-time precision consistency validation."""
from __future__ import annotations

from .result import ValidationResult


def validate_merge_consistency(reports: list[dict]) -> list[ValidationResult]:
    """Validate that all merged reports use consistent precision and metric."""
    results: list[ValidationResult] = []

    with_val = [r for r in reports if r.get("validation")]
    without_val = [r for r in reports if not r.get("validation")]

    if not with_val:
        return results

    if without_val:
        results.append(ValidationResult(
            level="warning",
            rule="merge_consistency",
            message=f"{len(without_val)} of {len(reports)} reports missing validation metadata",
            suggestion="Add validation block to all targets for consistent merge checks",
        ))

    # Check metric consistency
    metrics = set()
    precisions = set()
    for r in with_val:
        val = r["validation"]
        comp = val.get("comparison", {})
        prec = val.get("precision", {})
        if comp.get("metric"):
            metrics.add(comp["metric"])
        if prec.get("compute"):
            precisions.add(prec["compute"])

    if len(metrics) > 1:
        results.append(ValidationResult(
            level="error",
            rule="merge_consistency",
            message=f"Inconsistent comparison.metric across runs: {sorted(metrics)}",
            suggestion="All merged runs must use the same comparison metric",
        ))

    if len(precisions) > 1 and "tflops" in metrics:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"Mixed precision.compute ({sorted(precisions)}) with TFLOPS metric",
            suggestion="Cannot merge TFLOPS results across different precisions",
        ))

    return results
