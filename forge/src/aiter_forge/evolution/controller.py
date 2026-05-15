from __future__ import annotations
import uuid
from ..lineage.store import LineageStore
from ..lineage.types import KernelVariant
from ..knowledge.base import KnowledgeBase
from .scoring import ScoringFunction, BenchmarkResult


class EvolutionController:
    """AVO Agent(Pt, K, f) - generates optimization prompts using lineage + knowledge."""

    def __init__(
        self,
        lineage: LineageStore,
        knowledge: KnowledgeBase,
        scoring: ScoringFunction,
        kernel_path: str,
    ):
        self.lineage = lineage
        self.knowledge = knowledge
        self.scoring = scoring
        self.kernel_path = kernel_path

    def generate_optimization_prompt(self, round_num: int, parent_id: str | None) -> str:
        sections: list[str] = []

        # Header
        sections.append(f"# Optimization Round {round_num}\n")
        sections.append(f"Target kernel: `{self.kernel_path}`\n")

        # Lineage context (Pt) — key off store contents, not just parent_id
        has_lineage = len(self.lineage.all_variants()) > 0
        if parent_id:
            lineage_prompt = self.lineage.format_lineage_prompt(parent_id)
            sections.append(lineage_prompt)
        if has_lineage:
            best = self.lineage.best(self.scoring.primary_metric, self.scoring.higher_is_better)
            if best:
                sections.append(f"\nCurrent best: **{best.variant_id}** "
                              f"({self.scoring.primary_metric}={best.metrics.get(self.scoring.primary_metric, 'N/A')})\n")
        else:
            sections.append("This is the first optimization round. No prior lineage available.\n")

        # Knowledge base (K)
        all_patterns = self.knowledge.list_patterns()
        if all_patterns:
            sections.append(self.knowledge.format_prompt(all_patterns))

        # Scoring objective (f)
        direction = "maximize" if self.scoring.higher_is_better else "minimize"
        sections.append(f"\n## Objective\n{direction.capitalize()} **{self.scoring.primary_metric}**.\n")

        # Instructions
        sections.append(
            "## Instructions\n"
            "1. Analyze the current kernel and the optimization lineage above.\n"
            "2. Choose a strategy that has NOT been tried in previous rounds.\n"
            "3. Apply the optimization, ensuring correctness is preserved.\n"
            "4. The kernel must pass all existing tests.\n"
        )

        return "\n".join(sections)

    def commit_variant(
        self,
        round_num: int,
        parent_id: str | None,
        patch_path: str,
        result: BenchmarkResult,
        description: str,
        strategy: str,
    ) -> KernelVariant:
        """Unconditionally commit a variant (used for baseline).

        Raises ValueError if result is invalid or missing the primary metric.
        """
        if not result.valid:
            raise ValueError("Cannot commit variant with invalid benchmark result")
        if self.scoring.primary_metric not in result.metrics:
            raise ValueError(
                f"Cannot commit variant: missing primary metric '{self.scoring.primary_metric}'"
            )
        variant_id = f"v{uuid.uuid4().hex[:6]}"
        variant = KernelVariant(
            variant_id=variant_id,
            parent_id=parent_id,
            round_num=round_num,
            patch_path=patch_path,
            metrics=result.metrics,
            description=description,
            strategy=strategy,
        )
        self.lineage.add(variant)
        self.lineage.save()
        return variant

    def try_commit(
        self,
        round_num: int,
        parent_id: str | None,
        patch_path: str,
        result: BenchmarkResult,
        description: str,
        strategy: str,
        correct: bool,
    ) -> KernelVariant | None:
        """Commit only if correct AND improved over current best. Returns None if rejected."""
        if not correct:
            return None
        best = self.lineage.best(self.scoring.primary_metric, self.scoring.higher_is_better)
        if best and not self.scoring.is_better(result, BenchmarkResult(metrics=best.metrics)):
            return None
        return self.commit_variant(round_num, parent_id, patch_path, result, description, strategy)
