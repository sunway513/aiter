"""Pre-run GPU environment validation.

Checks: ROCR_VISIBLE_DEVICES isolation, single-GPU visibility in parallel mode.
"""
from __future__ import annotations

import os

from .result import ValidationResult


def validate_gpu_env(
    target: dict,
    parallel_mode: bool = False,
) -> list[ValidationResult]:
    """Validate GPU environment before running benchmark."""
    results: list[ValidationResult] = []
    val = target.get("validation")
    if not val:
        return results

    gpu = val.get("gpu", {})
    isolation = gpu.get("isolation")

    if not isolation:
        return results

    # Check the correct env var based on isolation type
    if isolation == "hip_visible_devices":
        env_var = "HIP_VISIBLE_DEVICES"
    else:
        env_var = "ROCR_VISIBLE_DEVICES"

    gpu_vis = os.environ.get(env_var)

    if isolation in ("rocr_visible_devices", "hip_visible_devices"):
        if not gpu_vis:
            level = "error" if parallel_mode else "warning"
            results.append(ValidationResult(
                level=level,
                rule="rule4_gpu_isolation",
                message=f"{env_var} not set",
                suggestion=f"Set {env_var}=N to isolate to single GPU",
            ))
        elif parallel_mode and "," in gpu_vis:
            results.append(ValidationResult(
                level="error",
                rule="rule4_gpu_isolation",
                message=f"Multiple GPUs visible ({gpu_vis}) in parallel mode — expected single GPU",
                suggestion=f"Set {env_var} to a single GPU ID",
            ))

    return results
