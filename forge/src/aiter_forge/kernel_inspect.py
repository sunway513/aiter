"""Kernel inspector: DSL-agnostic source↔hardware bundle builder.

Operationalizes expert principle #3's demand for "mapping between source and
hardware architecture." Produces four artifacts per kernel × shape:

  - MLIR attributes (kernel name, tile sizes, VGPR budget, LDS bytes)
  - ISA listing counters (MFMA / ds_read / ds_write / gmem_load; sgpr, vgpr, lds)
  - rocprof counters (MFMA busy cycles, LDS bank conflict, wave count)
  - predicted-vs-actual delta (when a hypothesis is provided)

The three parsers are pure-Python regex / JSON — unit-testable without a GPU.
The raw artifact generation (running rocprofv3, invoking FlyDSL, extracting
ISA from the binary) is the job of the ``ir-inspector`` P8 agent at runtime.
"""
from __future__ import annotations

import json
import re
from typing import Any

# --- MLIR parsing --------------------------------------------------------

_MLIR_NAME_RE = re.compile(r"rocdl\.kernel\s+@(\w+)")
_MLIR_INT_ATTR = {
    "vgpr": re.compile(r'"amdgpu\.num_vgpr"\s*=\s*(\d+)'),
    "lds_bytes": re.compile(r'"amdgpu\.lds_bytes"\s*=\s*(\d+)'),
}
_MLIR_TILE_RE = re.compile(r"\b([mnk])\s*=\s*(\d+)\s*:\s*i32")


def parse_mlir(text: str) -> dict[str, Any]:
    name_match = _MLIR_NAME_RE.search(text)
    out: dict[str, Any] = {"kernel_name": name_match.group(1) if name_match else None}
    for key, pattern in _MLIR_INT_ATTR.items():
        m = pattern.search(text)
        out[key] = int(m.group(1)) if m else None
    tile = {"m": None, "n": None, "k": None}
    for m in _MLIR_TILE_RE.finditer(text):
        tile[m.group(1)] = int(m.group(2))
    out["tile_m"] = tile["m"]
    out["tile_n"] = tile["n"]
    out["tile_k"] = tile["k"]
    return out


# --- ISA parsing ---------------------------------------------------------

_ISA_HEADER = {
    "vgpr": re.compile(r";\s*NumVgprs:\s*(\d+)", re.IGNORECASE),
    "sgpr": re.compile(r";\s*NumSgprs:\s*(\d+)", re.IGNORECASE),
    "lds_bytes": re.compile(r";\s*LdsSize:\s*(\d+)", re.IGNORECASE),
}
_ISA_COUNTERS = {
    "mfma_count": re.compile(r"\bv_mfma_\w+", re.MULTILINE),
    "ds_read_count": re.compile(r"\bds_read_\w+", re.MULTILINE),
    "ds_write_count": re.compile(r"\bds_write_\w+", re.MULTILINE),
    "gmem_load_count": re.compile(r"\bglobal_load_\w+", re.MULTILINE),
}


def parse_isa(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in _ISA_HEADER.items():
        m = pattern.search(text)
        out[key] = int(m.group(1)) if m else None
    for key, pattern in _ISA_COUNTERS.items():
        out[key] = len(pattern.findall(text))
    return out


# --- rocprof parsing -----------------------------------------------------

_ROCPROF_FIELD_MAP = {
    "KernelName": "kernel_name",
    "SQ_WAVES": "sq_waves",
    "TA_BUSY_CYCLES": "ta_busy_cycles",
    "MFMA_BUSY_CYCLES": "mfma_busy_cycles",
    "LDS_BANK_CONFLICT": "lds_bank_conflict",
    "SQ_INSTS_VMEM_RD": "vmem_reads",
}


def parse_rocprof(json_text: str) -> dict[str, Any] | None:
    data = json.loads(json_text)
    if not data:
        return None
    entry = data[0] if isinstance(data, list) else data
    out: dict[str, Any] = {}
    for src, dst in _ROCPROF_FIELD_MAP.items():
        if src in entry:
            out[dst] = entry[src]
    return out


# --- bundle --------------------------------------------------------------

def _resolve_actual(metric: str, mlir: dict | None, isa: dict | None, rocprof: dict | None) -> Any:
    """First non-None value across sources, in priority order: rocprof → isa → mlir."""
    for src in (rocprof, isa, mlir):
        if src and src.get(metric) is not None:
            return src[metric]
    return None


def build_bundle(
    *,
    mlir_text: str = "",
    isa_text: str = "",
    rocprof_json: str = "[]",
    hypothesis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mlir = parse_mlir(mlir_text) if mlir_text else None
    isa = parse_isa(isa_text) if isa_text else None
    rocprof = parse_rocprof(rocprof_json) if rocprof_json else None

    bundle: dict[str, Any] = {}
    if mlir is not None:
        bundle["mlir"] = mlir
    if isa is not None:
        bundle["isa"] = isa
    if rocprof is not None:
        bundle["rocprof"] = rocprof

    if hypothesis is not None:
        pva: dict[str, dict[str, Any]] = {}
        for metric, predicted in hypothesis.items():
            actual = _resolve_actual(metric, mlir, isa, rocprof)
            delta = None
            if isinstance(predicted, (int, float)) and isinstance(actual, (int, float)):
                delta = actual - predicted
            pva[metric] = {"predicted": predicted, "actual": actual, "delta": delta}
        bundle["predicted_vs_actual"] = pva

    return bundle
