"""Predict-verify task runner.

Operationalizes expert principle #3: every experiment declares a
hypothesis before running. The YAML schema makes trial-and-error
impossible — you can't submit a task without a non-empty hypothesis
and a concrete verify command + parse regex.

Produces a ``learnings/*.md`` draft after successful runs, so the
knowledge-sedimentation loop closes automatically.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable

import yaml

REQUIRED_FIELDS: tuple[str, ...] = ("id", "hypothesis", "candidates", "verify", "finalize")


@dataclass
class TaskSpec:
    id: str
    hypothesis: str
    candidates: list[dict[str, Any]]
    verify: dict[str, Any]
    finalize: dict[str, Any]
    source: Path | None = None


@dataclass
class SpecError:
    field: str
    detail: str


@dataclass
class RunResult:
    candidate: str
    exit_code: int
    stdout: str
    stderr: str
    parsed: dict[str, float]


def validate_spec(data: Any) -> list[SpecError]:
    if not isinstance(data, dict):
        return [SpecError("<root>", "task must be a mapping")]
    errors: list[SpecError] = []
    for f in REQUIRED_FIELDS:
        if f not in data:
            errors.append(SpecError(f, "missing required field"))
    if "hypothesis" in data and not str(data["hypothesis"]).strip():
        errors.append(SpecError("hypothesis", "must be non-empty"))
    if "candidates" in data:
        candidates = data["candidates"]
        if not isinstance(candidates, list) or not candidates:
            errors.append(SpecError("candidates", "must be a non-empty list"))
        else:
            for i, candidate in enumerate(candidates):
                if not isinstance(candidate, dict) or "name" not in candidate:
                    errors.append(SpecError(f"candidates[{i}]", "must be mapping with 'name'"))
    if "verify" in data:
        verify = data["verify"]
        if not isinstance(verify, dict) or not verify.get("cmd"):
            errors.append(SpecError("verify", "must be mapping with non-empty 'cmd'"))
    if "finalize" in data:
        finalize = data["finalize"]
        if not isinstance(finalize, dict) or not finalize.get("learning_path"):
            errors.append(SpecError("finalize", "must be mapping with 'learning_path'"))
    return errors


def parse(path: Path) -> tuple[TaskSpec | None, list[SpecError]]:
    data = yaml.safe_load(Path(path).read_text())
    errors = validate_spec(data)
    if errors:
        return None, errors
    return TaskSpec(
        id=data["id"],
        hypothesis=str(data["hypothesis"]).strip(),
        candidates=list(data["candidates"]),
        verify=dict(data["verify"]),
        finalize=dict(data["finalize"]),
        source=Path(path),
    ), []


Runner = Callable[..., subprocess.CompletedProcess]


def run_candidate(spec: TaskSpec, candidate: dict[str, Any],
                  *, runner: Runner = subprocess.run) -> RunResult:
    cmd_tmpl: str = spec.verify["cmd"]
    params: dict[str, Any] = dict(candidate.get("params", {}))
    cmd = cmd_tmpl.format(**params)
    proc = runner(shlex.split(cmd), capture_output=True, text=True, check=False)
    parsed: dict[str, float] = {}
    pattern = spec.verify.get("parse")
    if pattern and proc.stdout:
        regex = re.compile(pattern)
        for match in regex.finditer(proc.stdout):
            if match.groups():
                key = match.group(0)
                try:
                    parsed[key] = float(match.group(1))
                except ValueError:
                    continue
    return RunResult(
        candidate=candidate["name"],
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        parsed=parsed,
    )


def learning_draft(spec: TaskSpec, results: list[RunResult]) -> str:
    lines: list[str] = [
        f"# {spec.id}",
        "",
        "- **Area**: TBD",
        "- **Kernel**: TBD",
        "- **Shape**: TBD",
        f"- **Date**: {date.today().isoformat()}",
        "- **Confidence**: verified",
        "",
        "## Hypothesis",
        spec.hypothesis,
        "",
        "## Result",
    ]
    for r in results:
        parsed_view = ", ".join(f"{k}={v}" for k, v in r.parsed.items()) or "no values parsed"
        lines.append(f"- {r.candidate}: exit={r.exit_code}, {parsed_view}")
    lines += [
        "",
        "## Root cause",
        "TODO: explain why the result matches (or contradicts) the hypothesis.",
        "",
        "## Reusable rule",
        "TODO: one-sentence generalization for future shapes.",
        "",
        "## References",
        f"- Task: {spec.source or spec.id}",
    ]
    return "\n".join(lines)
