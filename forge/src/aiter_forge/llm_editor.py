# src/aiter_forge/llm_editor.py
"""LLM-based kernel editor using Claude API."""
from __future__ import annotations

import difflib
import os
import re
import sys
from dataclasses import dataclass, field

try:
    import anthropic
except ImportError:
    anthropic = None


SYSTEM_PROMPT = """\
You are an expert GPU kernel optimizer specializing in AMD MI355X (CDNA4) with Triton.
You receive a kernel source file and an optimization prompt with lineage context.

Rules:
1. Return the COMPLETE modified kernel file inside a single ```python code fence.
2. Preserve all function signatures and public APIs.
3. Only modify performance-relevant code (tiling, block sizes, memory access patterns).
4. Do NOT add new dependencies or imports beyond what exists.
5. Explain your changes briefly before the code fence.
"""


@dataclass
class EditResult:
    """Result of an LLM edit attempt."""
    original: str
    modified: str
    explanation: str
    success: bool
    error: str | None = None

    @property
    def has_changes(self) -> bool:
        return self.original != self.modified

    @property
    def diff(self) -> str | None:
        if not self.has_changes:
            return None
        return "\n".join(difflib.unified_diff(
            self.original.splitlines(keepends=True),
            self.modified.splitlines(keepends=True),
            fromfile="original",
            tofile="modified",
        ))


class LLMEditor:
    """Generate kernel edits using Claude API."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 8192):
        if anthropic is None:
            raise ImportError(
                "anthropic package is required for auto mode. "
                "Install it with: pip install aiter-forge[llm]"
            )
        self.model = model
        self.max_tokens = max_tokens

    def generate_edit(self, kernel_source: str, optimization_prompt: str) -> EditResult:
        """Send kernel + prompt to Claude, extract modified kernel."""
        user_msg = (
            f"## Current Kernel\n\n```python\n{kernel_source}\n```\n\n"
            f"## Optimization Task\n\n{optimization_prompt}"
        )

        try:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw_text = response.content[0].text
        except Exception as exc:
            print(f"[llm_editor] API error: {exc}", file=sys.stderr)
            return EditResult(
                original=kernel_source,
                modified=kernel_source,
                explanation=f"API error: {exc}",
                success=False,
                error=str(exc),
            )

        # Extract code from response
        modified = self._extract_code(raw_text)
        if modified is None:
            return EditResult(
                original=kernel_source,
                modified=kernel_source,
                explanation="No code fence found in LLM response",
                success=False,
            )

        # Extract explanation (text before the code fence)
        explanation = raw_text.split("```")[0].strip()

        return EditResult(
            original=kernel_source,
            modified=modified,
            explanation=explanation,
            success=True,
        )

    @staticmethod
    def _extract_code(text: str) -> str | None:
        """Extract the content of the first ```python code fence."""
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip() + "\n"
        return None
