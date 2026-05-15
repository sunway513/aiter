"""Complexity gate: quantified "simple enough" per expert principle #1.

"Anything fast, correct, and complicated must be thrown away." The gate
makes that objective: a file exceeding any limit fails CI. Limits are
configurable per-repo so policy can evolve without code changes.

Metrics:
  - file_loc            : non-blank / non-comment lines in the file
  - env_flags           : distinct ``os.environ`` / ``os.getenv`` names
  - funcs[name].loc     : function body length
  - funcs[name].branches: if / for / while / try count per function
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Limits:
    max_func_loc: int = 120
    max_env_flags: int = 5
    max_file_loc: int = 800
    max_branches_per_func: int = 15


@dataclass
class FileReport:
    path: Path
    file_loc: int
    env_flags: set[str] = field(default_factory=set)
    funcs: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class Violation:
    path: Path
    rule: str
    detail: str


# Matches all three common forms of env-var access used in our kernel code:
#   os.environ.get("X")   os.getenv("X")   os.environ["X"]
_ENV_RE = re.compile(
    r'os\.(?:(?:environ\.get|getenv)\s*\(|environ\s*\[)\s*[\'"]([A-Z_][A-Z0-9_]*)[\'"]'
)


class _BranchCounter(ast.NodeVisitor):
    def __init__(self) -> None:
        self.count = 0

    def visit_If(self, node: ast.If) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.count += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.count += 1
        self.generic_visit(node)


def _func_loc(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if not node.body:
        return 0
    last = node.body[-1]
    end_line = getattr(last, "end_lineno", last.lineno)
    return end_line - node.body[0].lineno + 1


def analyze(path: Path) -> FileReport:
    src = Path(path).read_text()
    tree = ast.parse(src, filename=str(path))
    funcs: dict[str, dict[str, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counter = _BranchCounter()
            counter.visit(node)
            funcs[node.name] = {"loc": _func_loc(node), "branches": counter.count}
    env_flags = set(_ENV_RE.findall(src))
    file_loc = sum(
        1 for line in src.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    return FileReport(path=Path(path), file_loc=file_loc, env_flags=env_flags, funcs=funcs)


def check(report: FileReport, limits: Limits) -> list[Violation]:
    out: list[Violation] = []
    if report.file_loc > limits.max_file_loc:
        out.append(Violation(report.path, "file_loc",
                             f"{report.file_loc} > {limits.max_file_loc}"))
    if len(report.env_flags) > limits.max_env_flags:
        flag_list = ", ".join(sorted(report.env_flags))
        out.append(Violation(report.path, "env_flags",
                             f"{len(report.env_flags)} flags ({flag_list}) > {limits.max_env_flags}"))
    for name, metrics in report.funcs.items():
        if metrics["loc"] > limits.max_func_loc:
            out.append(Violation(report.path, "func_loc",
                                 f"{name}: {metrics['loc']} > {limits.max_func_loc}"))
        if metrics["branches"] > limits.max_branches_per_func:
            out.append(Violation(report.path, "branches",
                                 f"{name}: {metrics['branches']} > {limits.max_branches_per_func}"))
    return out
