# src/aiter_forge/lineage/store.py
from __future__ import annotations
import json
from pathlib import Path
from .types import KernelVariant


class LineageStore:
    def __init__(self, store_dir: Path):
        self._dir = store_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._variants: dict[str, KernelVariant] = {}

    def add(self, variant: KernelVariant) -> None:
        if variant.variant_id in self._variants:
            raise ValueError(f"Duplicate variant_id: {variant.variant_id}")
        if variant.parent_id is not None and variant.parent_id not in self._variants:
            raise ValueError(
                f"Parent {variant.parent_id} not found for variant {variant.variant_id}"
            )
        self._variants[variant.variant_id] = variant

    def get(self, variant_id: str) -> KernelVariant | None:
        return self._variants.get(variant_id)

    def get_lineage(self, variant_id: str) -> list[KernelVariant]:
        chain: list[KernelVariant] = []
        cur = self._variants.get(variant_id)
        while cur:
            chain.append(cur)
            cur = self._variants.get(cur.parent_id) if cur.parent_id else None
        return list(reversed(chain))

    def best(self, metric: str, higher_is_better: bool = True) -> KernelVariant | None:
        valid = [v for v in self._variants.values() if metric in v.metrics]
        if not valid:
            return None
        return max(valid, key=lambda v: v.metrics[metric] * (1 if higher_is_better else -1))

    def all_variants(self) -> list[KernelVariant]:
        return list(self._variants.values())

    def format_lineage_prompt(self, variant_id: str) -> str:
        chain = self.get_lineage(variant_id)
        if not chain:
            return "No lineage available."
        lines = ["## Optimization Lineage\n"]
        for v in chain:
            metrics_str = ", ".join(f"{mk}={mv}" for mk, mv in v.metrics.items())
            lines.append(f"- **{v.variant_id}** (round {v.round_num}, strategy: {v.strategy}): {v.description}")
            lines.append(f"  Metrics: {metrics_str}")
            if v.parent_id:
                lines.append(f"  Parent: {v.parent_id}")
        return "\n".join(lines)

    def save(self) -> None:
        path = self._dir / "lineage.json"
        data = [v.to_dict() for v in self._variants.values()]
        path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        path = self._dir / "lineage.json"
        if path.exists():
            self._variants.clear()
            data = json.loads(path.read_text())
            variants = [KernelVariant.from_dict(d) for d in data]
            # Sort: roots first (parent_id=None), then by round_num
            variants.sort(key=lambda v: (v.parent_id is not None, v.round_num))
            for v in variants:
                self.add(v)
