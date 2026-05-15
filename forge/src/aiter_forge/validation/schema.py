# src/aiter_forge/validation/schema.py
"""Target-load-time schema validation.

Validates the `validation` block in target.yaml.
"""
from __future__ import annotations

from .result import ValidationResult

VALID_PRECISIONS = frozenset(("fp4", "fp8", "bf16", "fp16", "int8"))
VALID_FLOPS_KINDS = frozenset(("moe_stage1_gate_up", "moe_stage2_down", "custom"))
VALID_METRICS = frozenset(("tflops", "latency_us", "latency_ms", "bandwidth_gbps", "time_ms"))
LATENCY_METRICS = frozenset(("latency_us", "latency_ms"))


def validate_target_schema(target: dict) -> list[ValidationResult]:
    """Validate target.yaml schema. Called from mini_loop.load_target()."""
    results: list[ValidationResult] = []
    val = target.get("validation")

    if not val:
        results.append(ValidationResult(
            level="warning",
            rule="schema",
            message="No validation block in target.yaml",
            suggestion="Add validation block for precision/FLOPS enforcement",
        ))
        return results

    # --- precision ---
    precision = val.get("precision", {})
    compute = precision.get("compute")
    if compute and compute not in VALID_PRECISIONS:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"Invalid precision.compute: '{compute}'. Valid: {sorted(VALID_PRECISIONS)}",
        ))

    peak = precision.get("peak_tflops")
    if peak is not None and peak <= 0:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"precision.peak_tflops must be > 0, got {peak}",
        ))

    # --- comparison ---
    comparison = val.get("comparison", {})
    comp_metric = comparison.get("metric")
    if comp_metric and comp_metric not in VALID_METRICS:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"Invalid comparison.metric: '{comp_metric}'. Valid: {sorted(VALID_METRICS)}",
        ))

    comp_precision = comparison.get("precision")
    if compute and comp_precision and comp_precision != compute:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"comparison.precision ('{comp_precision}') != precision.compute ('{compute}')",
            suggestion="comparison.precision must match precision.compute for TFLOPS comparison",
        ))

    cross_metric = comparison.get("cross_precision_metric")
    if cross_metric and cross_metric not in LATENCY_METRICS:
        results.append(ValidationResult(
            level="error",
            rule="rule1_apple_to_apple",
            message=f"cross_precision_metric must be latency-based, got '{cross_metric}'",
            suggestion="Use 'latency_us' or 'latency_ms' for cross-precision comparison",
        ))

    # --- flops ---
    flops = val.get("flops", {})
    kind = flops.get("kind")
    if kind and kind not in VALID_FLOPS_KINDS:
        results.append(ValidationResult(
            level="error",
            rule="rule2_flops_formula",
            message=f"Invalid flops.kind: '{kind}'. Valid: {sorted(VALID_FLOPS_KINDS)}",
        ))

    # --- gpu ---
    gpu = val.get("gpu", {})
    isolation = gpu.get("isolation")
    valid_isolation = ("rocr_visible_devices", "hip_visible_devices")
    if isolation and isolation not in valid_isolation:
        results.append(ValidationResult(
            level="warning",
            rule="rule4_gpu_isolation",
            message=f"Unknown gpu.isolation method: '{isolation}'",
        ))

    return results
