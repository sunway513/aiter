from __future__ import annotations
from pathlib import Path


class KnowledgeBase:
    def __init__(self, patterns_dir: Path):
        self._dir = patterns_dir
        self._patterns: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.exists():
            return
        for f in self._dir.glob("*.md"):
            self._patterns[f.stem] = f.read_text()

    def list_patterns(self) -> list[str]:
        return list(self._patterns.keys())

    def get_pattern(self, name: str) -> str:
        return self._patterns.get(name, "")

    def query(self, query: str) -> list[tuple[str, str]]:
        """Simple keyword-based relevance search."""
        words = query.lower().split()
        scored: list[tuple[int, str, str]] = []
        for name, content in self._patterns.items():
            lower = content.lower()
            score = sum(1 for w in words if w in lower)
            if score > 0:
                scored.append((score, name, content))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(name, content) for _, name, content in scored]

    def format_prompt(self, pattern_names: list[str]) -> str:
        lines = ["## Knowledge Base\n"]
        for name in pattern_names:
            content = self._patterns.get(name, "")
            if content:
                lines.append(f"### {name}\n")
                lines.append(content)
                lines.append("")
        return "\n".join(lines)
