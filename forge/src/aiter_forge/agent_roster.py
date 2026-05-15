"""Agent roster: the five standard P8 agents used in aiter-forge.

Each agent is a markdown file under ``agents/<name>.md`` with five required
sections (Goal / Inputs / Outputs / Tools / Completion). A ``P9`` orchestrator
(see ``aiter_forge.dispatcher``, pending) renders these into agent prompts
programmatically, so humans write intent once and agents execute many times.

Mirrors ``aiter_forge.learnings`` — markdown-as-schema with a validator,
per-entry parser, and directory-level store.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS: tuple[str, ...] = ("Goal", "Inputs", "Outputs", "Tools", "Completion")
REQUIRED_AGENTS: tuple[str, ...] = (
    "kernel-writer",
    "bench-runner",
    "ir-inspector",
    "perf-analyzer",
    "learning-extractor",
)

_TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^## (.+?)\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)


@dataclass
class Agent:
    path: Path
    name: str
    sections: dict[str, str]


@dataclass
class ValidationIssue:
    path: Path
    kind: str  # "missing_section"
    detail: str


def parse_agent(path: Path) -> Agent:
    text = Path(path).read_text()
    title_match = _TITLE_RE.search(text)
    name = title_match.group(1).strip() if title_match else Path(path).stem
    sections = {m.group(1).strip(): m.group(2).strip() for m in _SECTION_RE.finditer(text)}
    return Agent(path=Path(path), name=name, sections=sections)


def validate(agent: Agent) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for section in REQUIRED_SECTIONS:
        if section not in agent.sections:
            issues.append(ValidationIssue(agent.path, "missing_section", section))
    return issues


class AgentRoster:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._agents: list[Agent] = []
        self._load()

    def _load(self) -> None:
        if not self.root.exists():
            return
        for path in sorted(self.root.glob("*.md")):
            if path.name == "README.md":
                continue
            self._agents.append(parse_agent(path))

    def agents(self) -> list[Agent]:
        return list(self._agents)

    def by_name(self, name: str) -> Agent | None:
        for agent in self._agents:
            if agent.name == name:
                return agent
        return None

    def validate_all(self) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for agent in self._agents:
            issues.extend(validate(agent))
        return issues

    def missing_standard(self) -> set[str]:
        present = {a.name for a in self._agents}
        return set(REQUIRED_AGENTS) - present
