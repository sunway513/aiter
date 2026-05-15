# src/aiter_forge/lineage/types.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class KernelVariant:
    variant_id: str
    parent_id: str | None
    round_num: int
    patch_path: str
    metrics: dict[str, float]
    description: str
    strategy: str
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelVariant:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
