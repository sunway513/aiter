from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal


@dataclass
class ValidationResult:
    level: Literal["error", "warning", "info"]
    rule: str
    message: str
    suggestion: str = ""


def check_all(results: list[ValidationResult], verbose: bool = False) -> bool:
    """Print all results, return False if any errors.

    INFO-level results are only printed when verbose=True.
    """
    errors = [r for r in results if r.level == "error"]
    for r in results:
        if r.level == "info" and not verbose:
            continue
        prefix = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[r.level]
        print(f"[{prefix}] [{r.rule}] {r.message}", file=sys.stderr)
        if r.suggestion:
            print(f"  → {r.suggestion}", file=sys.stderr)
    return len(errors) == 0
