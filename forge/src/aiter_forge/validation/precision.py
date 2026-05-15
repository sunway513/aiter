# src/aiter_forge/validation/precision.py
"""Post-benchmark precision and FLOPS validation.

Checks: cross-precision comparison, SOL%, TFLOPS sanity.
"""
from __future__ import annotations

from .result import ValidationResult

LATENCY_METRICS = frozenset(("latency_us", "latency_ms"))


def validate_benchmark_result(
    result: dict,
    target: dict,
    baseline_result: dict | None = None,
) -> list[ValidationResult]:
    """Validate a benchmark result against target validation rules."""
    results: list[ValidationResult] = []
    val = target.get("validation")
    if not val:
        return results

    precision = val.get("precision", {})
    comparison = val.get("comparison", {})
    peak = precision.get("peak_tflops")
    comp_metric = comparison.get("metric", "tflops")

    # SOL% calculation (info-level)
    tflops = result.get("tflops")
    if peak and tflops is not None:
        sol_pct = tflops / peak * 100
        results.append(ValidationResult(
            level="info",
            rule="rule1_sol",
            message=f"SOL = {sol_pct:.1f}% ({tflops:.1f} / {peak} TFLOPS)",
        ))
        if sol_pct > 100:
            results.append(ValidationResult(
                level="warning",
                rule="rule2_flops_formula",
                message=f"TFLOPS ({tflops:.1f}) exceeds peak ({peak}), SOL={sol_pct:.1f}%",
                suggestion="Check FLOPS formula for overcounting (e.g. gate+up double-counted)",
            ))

    # Cross-precision comparison check
    if baseline_result and comp_metric not in LATENCY_METRICS:
        base_prec = baseline_result.get("precision")
        curr_prec = result.get("precision")
        if base_prec and curr_prec and base_prec != curr_prec:
            results.append(ValidationResult(
                level="error",
                rule="rule1_apple_to_apple",
                message=f"Cross-precision TFLOPS comparison: {curr_prec} vs {base_prec}",
                suggestion="Use latency_us for cross-precision comparison",
            ))

    return results
