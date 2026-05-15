"""Learnings store: load, validate, and query experiment-derived knowledge.

Operationalizes expert principle #3 ("evidence-based, learning-driven"):
every experiment must append a structured entry under ``learnings/``, and
every entry must have the same four sections so a downstream agent can
skim it without a schema-guessing game.

Complements ``aiter_forge.knowledge.KnowledgeBase`` (static architectural
priors, hand-written). ``LearningsStore`` holds dynamic project-specific
experience derived from runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS: tuple[str, ...] = ("Hypothesis", "Result", "Root cause", "Reusable rule")
REQUIRED_META: tuple[str, ...] = ("Area", "Kernel", "Shape", "Date", "Confidence")

_META_RE = re.compile(r"^- \*\*([^*:]+)\*\*:\s*(.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^## (.+?)\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)


@dataclass
class Learning:
    path: Path
    title: str
    meta: dict[str, str]
    sections: dict[str, str]


@dataclass
class ValidationIssue:
    path: Path
    kind: str  # "missing_section" | "missing_meta"
    detail: str


def parse_learning(path: Path) -> Learning:
    text = Path(path).read_text()
    title_match = _TITLE_RE.search(text)
    title = title_match.group(1).strip() if title_match else Path(path).stem
    meta = {k.strip(): v.strip() for k, v in _META_RE.findall(text)}
    sections = {m.group(1).strip(): m.group(2).strip() for m in _SECTION_RE.finditer(text)}
    return Learning(path=Path(path), title=title, meta=meta, sections=sections)


def validate(learning: Learning) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for section in REQUIRED_SECTIONS:
        if section not in learning.sections:
            issues.append(ValidationIssue(learning.path, "missing_section", section))
    for key in REQUIRED_META:
        if key not in learning.meta:
            issues.append(ValidationIssue(learning.path, "missing_meta", key))
    return issues


class LearningsStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._entries: list[Learning] = []
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("**/*.md")):
            if path.name == "README.md":
                continue
            self._entries.append(parse_learning(path))

    def entries(self) -> list[Learning]:
        return list(self._entries)

    def validate_all(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for entry in self._entries:
            issues.extend(validate(entry))
        return issues

    def query(self, terms: str) -> list[Learning]:
        words = [w for w in terms.lower().split() if w]
        scored: list[tuple[int, Learning]] = []
        for entry in self._entries:
            haystack = entry.title.lower() + " " + " ".join(s.lower() for s in entry.sections.values())
            score = sum(1 for w in words if w in haystack)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored]
