# AITER-Forge Phase 1: Triton-First Mini AVO for AMD Kernels

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a minimal but runnable human-in-the-loop optimization harness for AITER Triton attention kernels on MI355X: load target, run correctness gate, run baseline benchmark across multiple shapes, generate an optimization prompt (with lineage + knowledge + objective), wait for human or external LLM to apply the edit, re-benchmark, and commit only if correct AND improved. Phase 2 replaces the human gate with autonomous LLM-driven edits. FlyDSL is the eventual authoring target; Triton is used first because AITER has ready-made benchmark harnesses.

**Architecture:** A standalone mini AVO sidecar (not a GEAK extension in Phase 1). Core modules: LineageStore (committed variant history), KnowledgeBase (CDNA4/Triton patterns), ScoringFunction (parses real Triton `perf_report` output), EvolutionController (prompt generation with attempt/commit semantics), and `mini_loop.py` (the runnable optimization loop). Optional GEAK integration is Phase 2.

**Tech Stack:** Python 3.10+, Triton on ROCm, AITER kernels as targets. No GEAK or LiteLLM dependency in Phase 1 (LiteLLM is Phase 2 for autonomous edits).

---

## File Structure

```
aiter-forge/
├── README.md                           # Project overview
├── pyproject.toml                      # Package config
├── .gitignore
├── src/aiter_forge/
│   ├── __init__.py
│   ├── mini_loop.py                    # THE RUNNABLE LOOP: load target → baseline → optimize → commit
│   ├── lineage/
│   │   ├── __init__.py
│   │   ├── store.py                    # LineageStore: committed variant history only
│   │   └── types.py                    # KernelVariant dataclass
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── base.py                     # KnowledgeBase: loads and queries patterns
│   │   └── patterns/
│   │       ├── cdna4_isa.md            # CDNA4/MI355X ISA optimization patterns
│   │       ├── triton_on_rocm.md       # Triton-specific patterns for ROCm
│   │       └── attention_kernels.md    # Attention kernel optimization patterns
│   └── evolution/
│       ├── __init__.py
│       ├── controller.py              # EvolutionController: attempt/commit semantics
│       └── scoring.py                 # ScoringFunction: parses Triton perf_report tables
├── scripts/
│   └── verify_mi355.sh                # MI355X environment verification
├── targets/
│   ├── local.env.example              # Template for machine-specific paths
│   └── aiter_mha/
│       └── target.yaml                 # Target kernel config (env-var driven)
├── tests/
│   ├── test_lineage_store.py
│   ├── test_knowledge_base.py
│   ├── test_scoring.py
│   ├── test_evolution_controller.py
│   ├── test_integration.py            # End-to-end smoke test (Layer 2)
│   └── test_mini_loop.py              # Local harness tests (Layer 2.5)
└── docs/
    ├── PLAN.md
    └── REVIEW.md
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/aiter_forge/__init__.py`
- Create: `README.md` (update existing)

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "aiter-forge"
version = "0.1.0"
description = "Human-in-the-loop kernel optimization harness for AMD MI355X (AVO methodology)"
requires-python = ">=3.10"
dependencies = [
    "pyyaml",
    "jinja2",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
```

- [ ] **Step 2: Create src/aiter_forge/__init__.py**

```python
"""AITER-Forge: Human-in-the-loop GPU kernel optimization harness for MI355X."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create directory structure**

```bash
mkdir -p src/aiter_forge/{lineage,knowledge/patterns,evolution}
mkdir -p targets/aiter_mha
mkdir -p tests
touch src/aiter_forge/{lineage,knowledge,evolution}/__init__.py
```

- [ ] **Step 4: Update README.md**

Update the existing README with project description, architecture diagram reference, and quickstart.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding for aiter-forge"
```

---

### Task 2: Lineage Store (AVO's Pt - Variant History)

**Files:**
- Create: `src/aiter_forge/lineage/types.py`
- Create: `src/aiter_forge/lineage/store.py`
- Test: `tests/test_lineage_store.py`

This implements AVO's core concept: the agent has access to the full lineage of kernel variants (Pt) when generating the next variation.

- [ ] **Step 1: Write failing tests for LineageStore**

```python
# tests/test_lineage_store.py
import pytest
import tempfile
from pathlib import Path
from aiter_forge.lineage.types import KernelVariant
from aiter_forge.lineage.store import LineageStore


def test_add_and_retrieve_variant():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LineageStore(Path(tmpdir))
        v = KernelVariant(
            variant_id="v001",
            parent_id=None,
            round_num=1,
            patch_path="patches/v001.patch",
            metrics={"throughput_tflops": 45.2, "duration_us": 120.0},
            description="Baseline kernel",
            strategy="baseline",
        )
        store.add(v)
        assert store.get("v001") == v


def test_lineage_chain():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LineageStore(Path(tmpdir))
        store.add(KernelVariant("v001", None, 1, "p1", {"tflops": 40}, "base", "baseline"))
        store.add(KernelVariant("v002", "v001", 2, "p2", {"tflops": 45}, "tile opt", "tiling"))
        store.add(KernelVariant("v003", "v002", 3, "p3", {"tflops": 50}, "vec opt", "vectorize"))
        chain = store.get_lineage("v003")
        assert [v.variant_id for v in chain] == ["v001", "v002", "v003"]


def test_best_variant():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LineageStore(Path(tmpdir))
        store.add(KernelVariant("v001", None, 1, "p1", {"tflops": 40}, "base", "baseline"))
        store.add(KernelVariant("v002", "v001", 2, "p2", {"tflops": 50}, "opt", "tiling"))
        store.add(KernelVariant("v003", "v001", 2, "p3", {"tflops": 45}, "opt2", "unroll"))
        best = store.best(metric="tflops", higher_is_better=True)
        assert best.variant_id == "v002"


def test_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        store1 = LineageStore(Path(tmpdir))
        store1.add(KernelVariant("v001", None, 1, "p1", {"tflops": 40}, "base", "baseline"))
        store1.save()
        store2 = LineageStore(Path(tmpdir))
        store2.load()
        assert store2.get("v001").metrics["tflops"] == 40


def test_format_lineage_prompt():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LineageStore(Path(tmpdir))
        store.add(KernelVariant("v001", None, 1, "p1", {"tflops": 40}, "base", "baseline"))
        store.add(KernelVariant("v002", "v001", 2, "p2", {"tflops": 50}, "tiled", "tiling"))
        prompt = store.format_lineage_prompt("v002")
        assert "v001" in prompt
        assert "tflops" in prompt
        assert "40" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_lineage_store.py -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement KernelVariant dataclass**

```python
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
```

- [ ] **Step 4: Implement LineageStore**

```python
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
            data = json.loads(path.read_text())
            for d in data:
                v = KernelVariant.from_dict(d)
                self._variants[v.variant_id] = v
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_lineage_store.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/aiter_forge/lineage/ tests/test_lineage_store.py
git commit -m "feat: lineage store for tracking kernel variant history (AVO Pt)"
```

---

### Task 3: Knowledge Base (AVO's K - Domain Knowledge)

**Files:**
- Create: `src/aiter_forge/knowledge/base.py`
- Create: `src/aiter_forge/knowledge/patterns/cdna4_isa.md`
- Create: `src/aiter_forge/knowledge/patterns/triton_on_rocm.md`
- Create: `src/aiter_forge/knowledge/patterns/attention_kernels.md`
- Test: `tests/test_knowledge_base.py`

The Knowledge Base stores AMD-specific optimization patterns that guide the agent. This maps to AVO's K parameter.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_knowledge_base.py
import pytest
import tempfile
from pathlib import Path
from aiter_forge.knowledge.base import KnowledgeBase


def test_load_patterns():
    kb = KnowledgeBase(Path(__file__).parent.parent / "src" / "aiter_forge" / "knowledge" / "patterns")
    patterns = kb.list_patterns()
    assert len(patterns) >= 3
    assert "cdna4_isa" in patterns


def test_get_pattern():
    kb = KnowledgeBase(Path(__file__).parent.parent / "src" / "aiter_forge" / "knowledge" / "patterns")
    content = kb.get_pattern("cdna4_isa")
    assert "MFMA" in content or "mfma" in content.lower()


def test_query_relevant():
    kb = KnowledgeBase(Path(__file__).parent.parent / "src" / "aiter_forge" / "knowledge" / "patterns")
    results = kb.query("attention tiling block size")
    assert len(results) > 0


def test_format_knowledge_prompt():
    kb = KnowledgeBase(Path(__file__).parent.parent / "src" / "aiter_forge" / "knowledge" / "patterns")
    prompt = kb.format_prompt(["cdna4_isa", "attention_kernels"])
    assert "## Knowledge Base" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_knowledge_base.py -v`
Expected: FAIL

- [ ] **Step 3: Create pattern files**

Create three markdown files with AMD-specific optimization knowledge:

`src/aiter_forge/knowledge/patterns/cdna4_isa.md`:

```markdown
# CDNA4 ISA Optimization Patterns (MI355X)

## Architecture Overview
- MI355X: CDNA4 architecture, 256 Compute Units (CUs)
- Wavefront size: 64 threads (always)
- Max wavefronts per CU: 8 (occupancy target)
- VGPR per CU: 512 (each 64-wide, so 32KB per wavefront if using 256 VGPRs)
- LDS per CU: 64 KB shared across wavefronts on that CU
- HBM3E: ~8 TB/s aggregate bandwidth

## MFMA Instructions
- Matrix Fused Multiply-Add: the primary compute primitive
- FP16: `v_mfma_f32_32x32x16_f16` (32x32 output, K=16, throughput ~1024 FLOPs/cycle/CU)
- FP8: `v_mfma_f32_32x32x32_fp8` (32x32 output, K=32, ~2048 FLOPs/cycle/CU)
- BF16: `v_mfma_f32_32x32x16_bf16` (same shape as FP16)
- Smaller tiles: 16x16 variants exist but lower throughput per instruction

## Key Optimization Patterns
1. **Tile to MFMA size**: Block dimensions should be multiples of 32 (MFMA native size)
2. **Minimize VGPR usage**: Each VGPR above 128 per wavefront reduces max occupancy
3. **LDS bank conflicts**: 32 banks, 4-byte stride. Pad shared memory to avoid conflicts
4. **Async global loads**: Use `buffer_load` with `s_waitcnt` for latency hiding
5. **Register rebalancing across warp groups**: Distribute accumulator registers to avoid spills
6. **Occupancy sweet spot**: 4-5 wavefronts/CU often better than max 8 (more registers available)

## Memory Hierarchy
- Registers: fastest, but limited (256 VGPRs max per wavefront for full occupancy)
- LDS: 64KB/CU, ~16 TB/s bandwidth, 1-2 cycle latency
- L2 Cache: shared across all CUs, ~4 TB/s
- HBM3E: ~8 TB/s, 100+ cycle latency
```

`src/aiter_forge/knowledge/patterns/triton_on_rocm.md`:

```markdown
# Triton on ROCm Optimization Patterns

## Block Size Recommendations for CDNA4
- BLOCK_M, BLOCK_N: use multiples of 32 (maps to MFMA tile size)
- Common effective sizes: 64, 128, 256
- BLOCK_K: 32 for FP16, 64 for FP8 (matches MFMA K dimension)
- Avoid BLOCK < 32 on any dimension (underutilizes MFMA units)

## tl.dot → MFMA Mapping
- `tl.dot(a, b)` compiles to MFMA instructions
- a shape: (BLOCK_M, BLOCK_K), b shape: (BLOCK_K, BLOCK_N)
- For best throughput: BLOCK_M=BLOCK_N=128, BLOCK_K=32 (FP16) or 64 (FP8)
- Accumulator is always FP32

## LDS (Shared Memory) Usage
- Triton automatically uses LDS for tl.load with block pointers
- LDS budget: 64KB per CU, shared across wavefronts
- Large blocks (256x128) may exceed LDS → reduce block size or num_warps
- `num_warps`: 4 or 8 typical. More warps = more LDS pressure

## Known Performance Pitfalls on ROCm
1. **num_stages**: Software pipelining stages. ROCm Triton supports 1-2 stages (not 4+ like CUDA)
2. **Atomic operations**: `tl.atomic_add` on FP16 can be slow; accumulate in FP32 then convert
3. **Predicated loads**: Masked loads generate scalar predicates; avoid complex mask expressions
4. **Compilation time**: Large kernels can take 60s+ to compile; use persistent kernel cache

## Autotune Parameter Ranges for MI355X
```python
@triton.autotune(configs=[
    triton.Config({'BLOCK_M': m, 'BLOCK_N': n, 'BLOCK_K': k, 'num_warps': w, 'num_stages': s})
    for m in [64, 128, 256]
    for n in [64, 128, 256]
    for k in [32, 64]
    for w in [4, 8]
    for s in [1, 2]
], key=['M', 'N', 'K'])
```
```

`src/aiter_forge/knowledge/patterns/attention_kernels.md`:

```markdown
# Attention Kernel Optimization Patterns

## FlashAttention Tiling Strategy
- Tile Q over BLOCK_M (rows of Q), iterate over KV in BLOCK_N chunks
- Each tile computes partial softmax; rescale accumulators when max changes
- Memory complexity: O(N) instead of O(N^2) by never materializing full attention matrix
- On CDNA4: BLOCK_M=128, BLOCK_N=64 is often optimal for head_dim=128

## Online Softmax with Rescaling
- Track running max `m_i` and sum `l_i` per row
- When new block has larger max: rescale accumulator by `exp(m_old - m_new)`
- AVO insight: **branchless accumulator rescaling** (+8.1% over branched version)
- Always use FP32 for max/sum tracking even with FP16 inputs

## Key Optimization Strategies (from AVO paper)
1. **Branchless accumulator rescaling**: Remove if/else on max comparison, always rescale
2. **Correction/MMA pipeline overlap**: Start next MFMA while correction factor is being applied
3. **Register rebalancing across warp groups**: Distribute Q/K/V/O accumulators to avoid spills

## Head Dimension Considerations
- head_dim=64: BLOCK_K=64 (single MFMA pass), very efficient
- head_dim=128: BLOCK_K=32 or 64 (2-4 MFMA passes per dot product)
- head_dim=256: High register pressure, may need to reduce BLOCK_M

## Multi-Head Attention Variants
- **MHA**: nheads_q == nheads_k, standard case
- **MQA**: nheads_k == 1, broadcast K/V across Q heads (memory bandwidth bound)
- **GQA**: nheads_q = n * nheads_k, group broadcast (intermediate case)
- For MQA/GQA: K/V reuse across heads → higher arithmetic intensity

## FP8 Attention
- Use `v_mfma_f32_32x32x32_fp8` (double the K dimension throughput)
- Quantize Q, K to FP8 before MFMA; keep accumulator in FP32
- Softmax probabilities: quantize to FP8 before V multiplication
- Watch for accuracy: FP8 E4M3 range is limited, may need per-head scaling
```

- [ ] **Step 4: Implement KnowledgeBase**

```python
# src/aiter_forge/knowledge/base.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_knowledge_base.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/aiter_forge/knowledge/ tests/test_knowledge_base.py
git commit -m "feat: knowledge base with CDNA4/Triton/attention patterns (AVO K)"
```

---

### Task 4: Scoring Function (AVO's f - Benchmark Evaluation)

**Files:**
- Create: `src/aiter_forge/evolution/scoring.py`
- Test: `tests/test_scoring.py`

The scoring function extracts performance metrics from benchmark output. Maps to AVO's f(x).

- [ ] **Step 1: Write failing tests**

```python
# tests/test_scoring.py
import pytest
from aiter_forge.evolution.scoring import ScoringFunction, BenchmarkResult


def test_parse_triton_perf_report():
    """Parse real Triton perf_report table output from bench_mha.py."""
    output = """FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          4  16  16     4096     4096      45.200"""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    result = sf.parse(output)
    assert result.metrics["tflops"] == pytest.approx(45.2)
    assert result.valid


def test_parse_multi_row_perf_report():
    """Parse multi-row Triton output and extract last value."""
    output = """FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          1  32   8        1     8192       12.500
1          4  16  16     4096     4096       45.200
2          1  48  48     1024     1024       38.700"""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = sf.parse_all_rows(output)
    assert len(results) == 3
    assert results[0].metrics["tflops"] == pytest.approx(12.5)
    assert results[1].metrics["tflops"] == pytest.approx(45.2)


def test_geomean_aggregate():
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = [
        BenchmarkResult(metrics={"tflops": 10.0}, valid=True),
        BenchmarkResult(metrics={"tflops": 40.0}, valid=True),
    ]
    agg = sf.aggregate_geomean(results)
    assert agg.metrics["tflops"] == pytest.approx(20.0)  # sqrt(10*40)


def test_compare():
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    a = BenchmarkResult(metrics={"tflops": 50.0}, valid=True)
    b = BenchmarkResult(metrics={"tflops": 45.0}, valid=True)
    assert sf.is_better(a, b)
    assert not sf.is_better(b, a)


def test_speedup():
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    baseline = BenchmarkResult(metrics={"tflops": 40.0}, valid=True)
    current = BenchmarkResult(metrics={"tflops": 50.0}, valid=True)
    assert sf.speedup(current, baseline) == pytest.approx(1.25)


# --- Defensive / edge-case tests ---

def test_parse_no_header():
    """Output with no recognizable table header → empty results."""
    output = "some random log output\nno table here\n"
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = sf.parse_all_rows(output)
    assert results == []
    single = sf.parse(output)
    assert not single.valid


def test_parse_header_but_missing_target_column():
    """Table header exists but target metric column is missing → empty results."""
    output = """FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    time_ms
0          4  16  16     4096     4096      2.350"""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = sf.parse_all_rows(output)
    assert results == []


def test_parse_non_numeric_value():
    """Row with non-numeric value in metric column → skip that row."""
    output = """FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          4  16  16     4096     4096           N/A
1          4  16  16     4096     4096          45.2"""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = sf.parse_all_rows(output)
    assert len(results) == 1
    assert results[0].metrics["tflops"] == pytest.approx(45.2)


def test_geomean_empty_input():
    """geomean of empty list → invalid result."""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    agg = sf.aggregate_geomean([])
    assert not agg.valid


def test_geomean_with_zero():
    """geomean with a zero value → result is 0 (not NaN or error)."""
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    results = [
        BenchmarkResult(metrics={"tflops": 0.0}, valid=True),
        BenchmarkResult(metrics={"tflops": 40.0}, valid=True),
    ]
    agg = sf.aggregate_geomean(results)
    assert agg.valid
    assert agg.metrics["tflops"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement ScoringFunction**

```python
# src/aiter_forge/evolution/scoring.py
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    metrics: dict[str, float] = field(default_factory=dict)
    valid: bool = True
    raw_output: str = ""


# Maps metric names to Triton perf_report column patterns
COLUMN_PATTERNS: dict[str, list[str]] = {
    "tflops": [r"fwd\(TFLOPS\)", r"bwd\(TFLOPS\)", r"TFLOPS"],
    "bandwidth_gbps": [r"fwd\(GB/s\)", r"bwd\(GB/s\)", r"GB/s"],
    "time_ms": [r"fwd\(ms\)", r"bwd\(ms\)", r"ms"],
}


class ScoringFunction:
    def __init__(self, primary_metric: str, higher_is_better: bool = True):
        self.primary_metric = primary_metric
        self.higher_is_better = higher_is_better

    def _find_metric_column(self, header_line: str) -> tuple[str, int] | None:
        """Find the column index for primary_metric in a Triton perf_report header."""
        patterns = COLUMN_PATTERNS.get(self.primary_metric, [])
        cols = re.split(r"\s{2,}", header_line.strip())
        for pat in patterns:
            for i, col in enumerate(cols):
                if re.search(pat, col, re.IGNORECASE):
                    return col, i
        return None

    def _parse_table_row(self, row: str, col_idx: int) -> float | None:
        """Extract float value from a specific column in a table row."""
        cols = re.split(r"\s{2,}", row.strip())
        # Account for possible row index prefix (e.g., "0  4  16  ...")
        if cols and re.match(r"^\d+$", cols[0]):
            cols = cols[1:]  # strip row index
        if col_idx < len(cols):
            try:
                return float(cols[col_idx])
            except ValueError:
                pass
        return None

    def parse(self, output: str) -> BenchmarkResult:
        """Parse the LAST row of a Triton perf_report table."""
        rows = self.parse_all_rows(output)
        if rows:
            return rows[-1]
        return BenchmarkResult(valid=False, raw_output=output)

    def parse_all_rows(self, output: str) -> list[BenchmarkResult]:
        """Parse ALL rows of a Triton perf_report table."""
        lines = output.strip().split("\n")
        header_idx = None
        col_idx = None
        for i, line in enumerate(lines):
            result = self._find_metric_column(line)
            if result:
                _, col_idx = result
                header_idx = i
                break
        if header_idx is None or col_idx is None:
            return []
        results: list[BenchmarkResult] = []
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            val = self._parse_table_row(line, col_idx)
            if val is not None:
                results.append(BenchmarkResult(
                    metrics={self.primary_metric: val},
                    valid=True,
                    raw_output=line,
                ))
        return results

    def aggregate_geomean(self, results: list[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate multiple results using geometric mean."""
        values = [r.metrics[self.primary_metric] for r in results
                  if self.primary_metric in r.metrics and r.metrics[self.primary_metric] > 0]
        if not values:
            return BenchmarkResult(valid=False)
        geomean = math.exp(sum(math.log(v) for v in values) / len(values))
        return BenchmarkResult(metrics={self.primary_metric: geomean}, valid=True)

    def is_better(self, a: BenchmarkResult, b: BenchmarkResult) -> bool:
        av = a.metrics.get(self.primary_metric, float("-inf"))
        bv = b.metrics.get(self.primary_metric, float("-inf"))
        return (av > bv) if self.higher_is_better else (av < bv)

    def speedup(self, current: BenchmarkResult, baseline: BenchmarkResult) -> float:
        cv = current.metrics.get(self.primary_metric, 0)
        bv = baseline.metrics.get(self.primary_metric, 0)
        if bv == 0:
            return 0.0
        return cv / bv if self.higher_is_better else bv / cv
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_scoring.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiter_forge/evolution/ tests/test_scoring.py
git commit -m "feat: scoring function for benchmark metric extraction (AVO f)"
```

---

### Task 5: Evolution Controller (AVO's Agent(Pt, K, f))

**Files:**
- Create: `src/aiter_forge/evolution/controller.py`
- Test: `tests/test_evolution_controller.py`

The Evolution Controller is the core of AVO methodology: it combines lineage (Pt), knowledge (K), and scoring (f) to generate the next optimization prompt, and manages committed vs rejected variants.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evolution_controller.py
import pytest
import tempfile
from pathlib import Path
from aiter_forge.evolution.controller import EvolutionController
from aiter_forge.evolution.scoring import ScoringFunction, BenchmarkResult
from aiter_forge.lineage.store import LineageStore
from aiter_forge.lineage.types import KernelVariant
from aiter_forge.knowledge.base import KnowledgeBase


@pytest.fixture
def tmp_env(tmp_path):
    # Create minimal knowledge patterns
    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    (patterns_dir / "cdna4_isa.md").write_text("# CDNA4\nMFMA instructions, wavefront64, LDS 64KB per CU")
    (patterns_dir / "attention_kernels.md").write_text("# Attention\nFlash attention tiling, online softmax")

    store = LineageStore(tmp_path / "lineage")
    kb = KnowledgeBase(patterns_dir)
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    return store, kb, sf, tmp_path


def test_generate_initial_prompt(tmp_env):
    store, kb, sf, _ = tmp_env
    ctrl = EvolutionController(store, kb, sf, kernel_path="/path/to/kernel.py")
    prompt = ctrl.generate_optimization_prompt(round_num=1, parent_id=None)
    assert "kernel" in prompt.lower()
    assert "CDNA4" in prompt or "cdna4" in prompt.lower()


def test_generate_prompt_with_lineage(tmp_env):
    store, kb, sf, _ = tmp_env
    store.add(KernelVariant("v001", None, 1, "p1", {"tflops": 40}, "baseline", "baseline"))
    store.add(KernelVariant("v002", "v001", 2, "p2", {"tflops": 45}, "tiled", "tiling"))
    ctrl = EvolutionController(store, kb, sf, kernel_path="/path/to/kernel.py")
    prompt = ctrl.generate_optimization_prompt(round_num=3, parent_id="v002")
    assert "v001" in prompt
    assert "v002" in prompt
    assert "45" in prompt or "tflops" in prompt.lower()


def test_commit_variant_on_improvement(tmp_env):
    store, kb, sf, tmp_path = tmp_env
    ctrl = EvolutionController(store, kb, sf, kernel_path="/path/to/kernel.py")
    # Baseline
    baseline = BenchmarkResult(metrics={"tflops": 40.0}, valid=True)
    v1 = ctrl.commit_variant(1, None, "p1", baseline, "baseline", "baseline")
    assert store.get(v1.variant_id) is not None
    # Improved attempt → committed
    better = BenchmarkResult(metrics={"tflops": 50.0}, valid=True)
    v2 = ctrl.try_commit(2, v1.variant_id, "p2", better, "tiled", "tiling", correct=True)
    assert v2 is not None
    assert store.get(v2.variant_id) is not None


def test_reject_variant_on_regression(tmp_env):
    store, kb, sf, tmp_path = tmp_env
    ctrl = EvolutionController(store, kb, sf, kernel_path="/path/to/kernel.py")
    baseline = BenchmarkResult(metrics={"tflops": 40.0}, valid=True)
    ctrl.commit_variant(1, None, "p1", baseline, "baseline", "baseline")
    # Regressed attempt → rejected, not in committed lineage
    worse = BenchmarkResult(metrics={"tflops": 35.0}, valid=True)
    v2 = ctrl.try_commit(2, None, "p2", worse, "bad opt", "unroll", correct=True)
    assert v2 is None
    assert len(store.all_variants()) == 1  # only baseline


def test_reject_variant_on_incorrect(tmp_env):
    store, kb, sf, tmp_path = tmp_env
    ctrl = EvolutionController(store, kb, sf, kernel_path="/path/to/kernel.py")
    baseline = BenchmarkResult(metrics={"tflops": 40.0}, valid=True)
    ctrl.commit_variant(1, None, "p1", baseline, "baseline", "baseline")
    # Better but incorrect → rejected
    better = BenchmarkResult(metrics={"tflops": 60.0}, valid=True)
    v2 = ctrl.try_commit(2, None, "p2", better, "broken opt", "unsafe", correct=False)
    assert v2 is None
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement EvolutionController**

```python
# src/aiter_forge/evolution/controller.py
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

        # Lineage context (Pt)
        if parent_id:
            lineage_prompt = self.lineage.format_lineage_prompt(parent_id)
            sections.append(lineage_prompt)
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
        """Unconditionally commit a variant (used for baseline)."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_evolution_controller.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aiter_forge/evolution/controller.py tests/test_evolution_controller.py
git commit -m "feat: evolution controller implementing AVO Agent(Pt, K, f) paradigm"
```

---

### Task 6: AITER Target Kernel Setup

**Files:**
- Create: `targets/aiter_mha/target.yaml`

This task selects the AITER MHA (Multi-Head Attention) Triton kernel as the first optimization target.

- [ ] **Step 1: Create target configuration**

```yaml
# targets/aiter_mha/target.yaml
#
# Portable target config. Machine-specific paths are resolved at runtime
# via environment variables. See targets/README.md for setup instructions.
#
name: aiter-flash-attention-triton
description: AITER Flash Attention Triton kernel (forward pass). Phase 1 uses Triton; FlyDSL is the eventual authoring target.

kernel:
  # Relative to $AITER_ROOT (set in env or targets/local.env)
  path: aiter/ops/triton/_triton_kernels/attention/mha.py
  repo_env: AITER_ROOT       # environment variable holding the repo root
  language: triton

correctness:
  # bench_mha.py -test_mode runs torch reference comparison
  command: "cd $AITER_ROOT && python op_tests/op_benchmarks/triton/bench_mha.py -b 4 -hq 16 -hk 16 -d 128 -sq 4096 -sk 4096 -test_mode"
  pass_pattern: "test passed"  # stdout must contain this

benchmark:
  # Triton perf_report output: table with provider columns like "fwd(TFLOPS)"
  command: "cd $AITER_ROOT && python op_tests/op_benchmarks/triton/bench_mha.py -b {batch} -hq {hq} -hk {hk} -d {d} -sq {sq} -sk {sk} -metric throughput"
  shapes:
    - {batch: 1, hq: 32, hk: 8, d: 128, sq: 1, sk: 8192}      # GQA decode (latency bound)
    - {batch: 4, hq: 16, hk: 16, d: 128, sq: 4096, sk: 4096}   # MHA prefill (compute bound)
    - {batch: 1, hq: 48, hk: 48, d: 128, sq: 1024, sk: 1024}   # MHA medium
  aggregate: geomean

scoring:
  primary_metric: tflops
  higher_is_better: true
  # Parser must handle Triton perf_report table format:
  #   BATCH  HQ  HK  N_CTX_Q  N_CTX_K  fwd(TFLOPS)
  #   4      16  16  4096     4096     45.2
  output_format: triton_perf_report

hardware:
  target: MI355X
  gpu_ids: [0]
  # SSH connection is NOT stored here. Use targets/local.env or ~/.ssh/config.
```

Also create `targets/local.env.example` (gitignored, copied to `local.env` by each developer):

```bash
# targets/local.env.example — copy to targets/local.env and fill in
AITER_ROOT=/path/to/aiter
AITER_FORGE_SSH_TARGET=mi355-gpu-9    # SSH host alias from ~/.ssh/config
```

And add to `.gitignore`:

```
targets/local.env
```

- [ ] **Step 2: Commit**

```bash
git add targets/aiter_mha/
git commit -m "feat: add AITER MHA Triton kernel as first optimization target"
```

---

### Task 7: MI355X Environment Verification Script

**Files:**
- Create: `scripts/verify_mi355.sh`

- [ ] **Step 1: Create verification script**

```bash
#!/bin/bash
# Verify MI355X environment for AITER-Forge
# Run this ON the remote MI355X node (not locally).
# Usage: ssh mi355-gpu-9 'bash -s' < scripts/verify_mi355.sh
set -e

echo "=== MI355X Environment Verification ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "=== ROCm Version ==="
cat /opt/rocm/.info/version 2>/dev/null || echo "ROCm version file not found"
rocm-smi --showid 2>/dev/null | head -20 || echo "rocm-smi not available"

echo ""
echo "=== GPU Info ==="
rocminfo 2>/dev/null | grep -A5 "Name:" | head -30 || echo "rocminfo not available"

echo ""
echo "=== Python ==="
python3 --version
pip3 list 2>/dev/null | grep -i -E "torch|triton|aiter" || echo "Key packages not found"

echo ""
echo "=== AITER Check ==="
# AITER_ROOT should match the value in targets/local.env
AITER_ROOT="${AITER_ROOT:-}"
if [ -z "$AITER_ROOT" ]; then
    echo "AITER_ROOT not set. Searching common locations..."
    for p in "$HOME/workspace/aiter" "$HOME/aiter" "/opt/aiter"; do
        if [ -d "$p" ]; then AITER_ROOT="$p"; break; fi
    done
fi
if [ -n "$AITER_ROOT" ] && [ -d "$AITER_ROOT" ]; then
    echo "AITER found at $AITER_ROOT"
else
    echo "AITER not found. Set AITER_ROOT env var."
fi

echo ""
echo "=== Triton Test ==="
python3 -c "import triton; print(f'Triton version: {triton.__version__}')" 2>/dev/null || echo "Triton not importable"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not importable"

echo ""
echo "=== Verification Complete ==="
```

- [ ] **Step 2: Commit**

```bash
chmod +x scripts/verify_mi355.sh
git add scripts/
git commit -m "feat: MI355X environment verification script"
```

---

### Task 8: Integration Smoke Test (Pt + K + f end-to-end)

**Files:**
- Create: `tests/test_integration.py`

This test verifies the full happy path using real Triton `perf_report` output format and the `commit_variant()`/`try_commit()` API:
- Baseline: parse perf_report → `commit_variant()` (unconditional)
- Improved variant: `try_commit(correct=True)` → committed
- Regressed variant: `try_commit(correct=True)` → rejected (no improvement)
- Incorrect variant: `try_commit(correct=False)` → rejected

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
"""End-to-end smoke test: ScoringFunction → EvolutionController → next prompt.

Uses real Triton perf_report output format and commit_variant/try_commit API.
"""
from pathlib import Path

from aiter_forge.evolution.controller import EvolutionController
from aiter_forge.evolution.scoring import ScoringFunction
from aiter_forge.knowledge.base import KnowledgeBase
from aiter_forge.lineage.store import LineageStore

# Sample Triton perf_report table (matches bench_mha.py output format)
BASELINE_OUTPUT = """\
       batch     hq     hk      d     sq     sk      fwd(TFLOPS)
           4     16     16    128   4096   4096           42.50
"""

IMPROVED_OUTPUT = """\
       batch     hq     hk      d     sq     sk      fwd(TFLOPS)
           4     16     16    128   4096   4096           55.80
"""

REGRESSED_OUTPUT = """\
       batch     hq     hk      d     sq     sk      fwd(TFLOPS)
           4     16     16    128   4096   4096           38.10
"""


def test_full_pipeline(tmp_path):
    """Happy path: baseline → improved commit → regressed reject → incorrect reject."""
    # Setup knowledge base
    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    (patterns_dir / "cdna4_isa.md").write_text("# CDNA4\nMFMA 32x32x16, wavefront64, LDS 64KB/CU")
    (patterns_dir / "attention_kernels.md").write_text("# Attention\nFlash attention tiling, online softmax")

    store = LineageStore(tmp_path / "lineage")
    kb = KnowledgeBase(patterns_dir)
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    ctrl = EvolutionController(store, kb, sf, kernel_path="attention/mha.py")

    # 1. Baseline: parse perf_report, unconditionally commit
    # Note: parser maps column header "fwd(TFLOPS)" → metric key "tflops"
    baseline_results = sf.parse_all_rows(BASELINE_OUTPUT)
    assert len(baseline_results) >= 1
    baseline = baseline_results[0]
    assert baseline.valid
    assert abs(baseline.metrics["tflops"] - 42.5) < 0.01

    v1 = ctrl.commit_variant(
        round_num=0, parent_id=None,
        patch_path="patches/baseline.patch", result=baseline,
        description="baseline run", strategy="baseline",
    )
    assert v1 is not None

    # 2. Improved variant: try_commit should succeed
    improved = sf.parse_all_rows(IMPROVED_OUTPUT)[0]
    v2 = ctrl.try_commit(
        round_num=1, parent_id=v1.variant_id,
        patch_path="patches/v2.patch", result=improved,
        description="round 1 optimization", strategy="llm_suggested",
        correct=True,
    )
    assert v2 is not None  # committed because 55.80 > 42.50

    # 3. Regressed variant: try_commit should reject
    regressed = sf.parse_all_rows(REGRESSED_OUTPUT)[0]
    v3 = ctrl.try_commit(
        round_num=2, parent_id=v2.variant_id,
        patch_path="patches/v3.patch", result=regressed,
        description="round 2 optimization", strategy="llm_suggested",
        correct=True,
    )
    assert v3 is None  # rejected because 38.10 < 55.80

    # 4. Incorrect variant: try_commit should reject without checking metrics
    v4 = ctrl.try_commit(
        round_num=3, parent_id=v2.variant_id,
        patch_path="patches/v4.patch", result=improved,
        description="round 3 optimization", strategy="llm_suggested",
        correct=False,
    )
    assert v4 is None  # rejected because correct=False

    # 5. Verify committed lineage only has v1 (baseline) and v2 (improved)
    all_variants = store.all_variants()
    assert len(all_variants) == 2

    # 6. Generate next prompt — should contain lineage + knowledge + objective
    prompt = ctrl.generate_optimization_prompt(round_num=4, parent_id=v2.variant_id)
    assert v2.variant_id in prompt                         # lineage present
    assert "55.8" in prompt or "tflops" in prompt.lower()   # metrics present
    assert "MFMA" in prompt or "mfma" in prompt.lower()    # knowledge present
    assert "maximize" in prompt.lower() or "Maximize" in prompt  # objective present
```

- [ ] **Step 2: Run test**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end smoke test for Pt+K+f pipeline"
```

---

### Task 8.5: Local Harness Tests for mini_loop.py (Layer 2.5)

**Files:**
- Create: `tests/test_mini_loop.py`
- Depends on: Task 4 (ScoringFunction), Task 5 (EvolutionController), Task 9 (mini_loop.py)

This test file covers the orchestration glue in `mini_loop.py` **without GPU, SSH, or real benchmark commands**. All external calls are monkeypatched with fake outputs.

Note: This task is written before Task 9 in the plan document, but must be implemented **after** Task 9 since it imports from `mini_loop.py`. Execution order: Task 9 first, then Task 8.5.

- [ ] **Step 1: Write harness tests**

```python
# tests/test_mini_loop.py
"""Layer 2.5: Local harness tests for mini_loop.py.

Tests the orchestration glue without GPU/SSH. All subprocess calls are
monkeypatched with fake outputs.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from aiter_forge.mini_loop import (
    load_target,
    main,
    run_benchmark,
    run_correctness,
)
from aiter_forge.evolution.scoring import ScoringFunction


# --- Fake target config for all tests ---

FAKE_TARGET = {
    "name": "test_kernel",
    "kernel": {"path": "test/kernel.py"},
    "correctness": {
        "command": "echo test passed",
        "pass_pattern": "test passed",
    },
    "scoring": {
        "primary_metric": "tflops",
        "higher_is_better": True,
    },
    "benchmark": {
        "command": "echo benchmark",
        "shapes": [
            {"batch": 1, "hq": 32, "hk": 8, "d": 128, "sq": 1, "sk": 8192},
            {"batch": 4, "hq": 16, "hk": 16, "d": 128, "sq": 4096, "sk": 4096},
            {"batch": 1, "hq": 48, "hk": 48, "d": 128, "sq": 1024, "sk": 1024},
        ],
        "aggregate": "geomean",
    },
}

FAKE_PERF_OUTPUT_SHAPE1 = """\
FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          1  32   8        1     8192       12.50
"""

FAKE_PERF_OUTPUT_SHAPE2 = """\
FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          4  16  16     4096     4096       45.20
"""

FAKE_PERF_OUTPUT_SHAPE3 = """\
FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          1  48  48     1024     1024       38.70
"""

IMPROVED_PERF_OUTPUT = """\
FlashAttention-fwd:
       BATCH  HQ  HK  N_CTX_Q  N_CTX_K    fwd(TFLOPS)
0          1  32   8        1     8192       18.00
"""


def _write_target(tmp_path: Path, target: dict = None) -> Path:
    """Write a fake target.yaml and return target directory."""
    import yaml
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    yaml_content = yaml.dump(target or FAKE_TARGET)
    (target_dir / "target.yaml").write_text(yaml_content)
    # Create fake knowledge patterns dir
    patterns_dir = tmp_path / "patterns"
    patterns_dir.mkdir()
    (patterns_dir / "cdna4_isa.md").write_text("# CDNA4\nMFMA 32x32x16")
    return target_dir


# ---- Unit-level function tests ----


def test_run_correctness_pass():
    """Correctness command returns rc=0 and output contains pass_pattern → True."""
    target = {
        "correctness": {"command": "echo test passed", "pass_pattern": "test passed"}
    }
    with patch("aiter_forge.mini_loop.run_command", return_value=(0, "test passed\n")):
        assert run_correctness(target) is True


def test_run_correctness_fail_on_nonzero_exit():
    """Correctness returns rc!=0 → False, even if output contains pass pattern."""
    target = {
        "correctness": {"command": "false", "pass_pattern": "test passed"}
    }
    with patch("aiter_forge.mini_loop.run_command", return_value=(1, "test passed\n")):
        assert run_correctness(target) is False


def test_run_correctness_fail_on_missing_pattern():
    """Correctness returns rc=0 but output lacks pass_pattern → False."""
    target = {
        "correctness": {"command": "echo wrong", "pass_pattern": "test passed"}
    }
    with patch("aiter_forge.mini_loop.run_command", return_value=(0, "some other output\n")):
        assert run_correctness(target) is False


def test_run_benchmark_parses_multiple_shapes():
    """Benchmark runs 3 shapes, each returns valid perf_report → 3 BenchmarkResults."""
    fake_outputs = [
        (0, FAKE_PERF_OUTPUT_SHAPE1),
        (0, FAKE_PERF_OUTPUT_SHAPE2),
        (0, FAKE_PERF_OUTPUT_SHAPE3),
    ]
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    with patch("aiter_forge.mini_loop.run_command", side_effect=fake_outputs):
        results = run_benchmark(FAKE_TARGET, sf)
    assert len(results) == 3
    assert results[0].metrics["tflops"] == pytest.approx(12.5)
    assert results[1].metrics["tflops"] == pytest.approx(45.2)
    assert results[2].metrics["tflops"] == pytest.approx(38.7)


def test_run_benchmark_skips_failed_commands():
    """Benchmark command fails for one shape → skip it, return remaining."""
    fake_outputs = [
        (1, "error"),  # shape 1 fails
        (0, FAKE_PERF_OUTPUT_SHAPE2),
        (0, FAKE_PERF_OUTPUT_SHAPE3),
    ]
    sf = ScoringFunction(primary_metric="tflops", higher_is_better=True)
    with patch("aiter_forge.mini_loop.run_command", side_effect=fake_outputs):
        results = run_benchmark(FAKE_TARGET, sf)
    assert len(results) == 2


# ---- main() orchestration tests ----


def _run_main(tmp_path, correctness_returns, benchmark_returns, input_returns=None):
    """Helper: run main() with monkeypatched run_command and input()."""
    target_dir = _write_target(tmp_path)
    out_dir = tmp_path / "output"

    with patch("aiter_forge.mini_loop.run_command", side_effect=correctness_returns + benchmark_returns), \
         patch("aiter_forge.mini_loop.KnowledgeBase") as mock_kb, \
         patch("builtins.input", side_effect=input_returns or [""]):
        mock_kb.return_value.format_knowledge_prompt.return_value = "## Knowledge\nMFMA"
        mock_kb.return_value.query_relevant.return_value = []
        main(target_dir=str(target_dir), output_dir=str(out_dir), max_rounds=1 if input_returns else 0)

    return out_dir


def test_main_baseline_commit_and_report(tmp_path):
    """main() with 0 optimization rounds: baseline committed, report.json written."""
    correctness = [(0, "test passed\n")]
    benchmark = [
        (0, FAKE_PERF_OUTPUT_SHAPE1),
        (0, FAKE_PERF_OUTPUT_SHAPE2),
        (0, FAKE_PERF_OUTPUT_SHAPE3),
    ]
    out_dir = _run_main(tmp_path, correctness, benchmark)

    report = json.loads((out_dir / "report.json").read_text())
    assert report["target"] == "test_kernel"
    assert report["committed_variants"] >= 1
    assert "tflops" in report["baseline"]


def test_main_commits_improved_variant(tmp_path):
    """main() with 1 round: improved variant → committed, total variants = 2."""
    # Calls: correctness(baseline) + 3 benchmark shapes + correctness(round) + 3 benchmark shapes
    correctness_baseline = [(0, "test passed\n")]
    benchmark_baseline = [
        (0, FAKE_PERF_OUTPUT_SHAPE1),
        (0, FAKE_PERF_OUTPUT_SHAPE2),
        (0, FAKE_PERF_OUTPUT_SHAPE3),
    ]
    correctness_round = [(0, "test passed\n")]
    # Improved: all shapes return higher values
    benchmark_round = [
        (0, IMPROVED_PERF_OUTPUT.replace("18.00", "20.00")),
        (0, IMPROVED_PERF_OUTPUT.replace("18.00", "60.00")),
        (0, IMPROVED_PERF_OUTPUT.replace("18.00", "50.00")),
    ]
    out_dir = _run_main(
        tmp_path,
        correctness_baseline + correctness_round,
        benchmark_baseline + benchmark_round,
        input_returns=[""],
    )
    report = json.loads((out_dir / "report.json").read_text())
    assert report["committed_variants"] == 2


def test_main_rejects_regressed_variant(tmp_path):
    """main() with 1 round: regressed variant → rejected, total variants = 1."""
    correctness_baseline = [(0, "test passed\n")]
    benchmark_baseline = [
        (0, FAKE_PERF_OUTPUT_SHAPE1),
        (0, FAKE_PERF_OUTPUT_SHAPE2),
        (0, FAKE_PERF_OUTPUT_SHAPE3),
    ]
    correctness_round = [(0, "test passed\n")]
    # Regressed: all shapes return lower values
    low_output = FAKE_PERF_OUTPUT_SHAPE1.replace("12.50", "5.00")
    benchmark_round = [
        (0, low_output),
        (0, low_output),
        (0, low_output),
    ]
    out_dir = _run_main(
        tmp_path,
        correctness_baseline + correctness_round,
        benchmark_baseline + benchmark_round,
        input_returns=[""],
    )
    report = json.loads((out_dir / "report.json").read_text())
    assert report["committed_variants"] == 1  # only baseline


def test_main_aborts_when_baseline_correctness_fails(tmp_path):
    """Baseline correctness fails → sys.exit(1)."""
    target_dir = _write_target(tmp_path)
    out_dir = tmp_path / "output"

    with patch("aiter_forge.mini_loop.run_command", return_value=(1, "FAIL\n")), \
         patch("aiter_forge.mini_loop.KnowledgeBase") as mock_kb, \
         pytest.raises(SystemExit, match="1"):
        mock_kb.return_value.format_knowledge_prompt.return_value = ""
        main(target_dir=str(target_dir), output_dir=str(out_dir), max_rounds=0)


def test_main_aborts_when_no_valid_baseline_results(tmp_path):
    """Baseline benchmark returns unparseable output → sys.exit(1)."""
    correctness = [(0, "test passed\n")]
    # All 3 shapes return garbage
    benchmark = [(0, "garbage output\n")] * 3

    target_dir = _write_target(tmp_path)
    out_dir = tmp_path / "output"

    with patch("aiter_forge.mini_loop.run_command", side_effect=correctness + benchmark), \
         patch("aiter_forge.mini_loop.KnowledgeBase") as mock_kb, \
         pytest.raises(SystemExit, match="1"):
        mock_kb.return_value.format_knowledge_prompt.return_value = ""
        main(target_dir=str(target_dir), output_dir=str(out_dir), max_rounds=0)


def test_main_aborts_on_missing_required_field(tmp_path):
    """target.yaml missing a required field → sys.exit(1)."""
    import yaml
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    # Missing 'benchmark.command'
    incomplete_target = {
        "name": "incomplete",
        "kernel": {"path": "test.py"},
        "correctness": {"command": "echo ok"},
        "scoring": {"primary_metric": "tflops"},
        "benchmark": {"shapes": [{"batch": 1}]},
        # no "command" or "aggregate"
    }
    (target_dir / "target.yaml").write_text(yaml.dump(incomplete_target))
    out_dir = tmp_path / "output"

    with pytest.raises(SystemExit, match="1"):
        main(target_dir=str(target_dir), output_dir=str(out_dir), max_rounds=0)
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/test_mini_loop.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_mini_loop.py
git commit -m "test: add Layer 2.5 local harness tests for mini_loop.py"
```

---

### Task 9: Mini Loop — The Runnable Optimization Loop

**Files:**
- Create: `src/aiter_forge/mini_loop.py`

This is the most important file in Phase 1. It makes the system actually runnable: load target → correctness check → baseline benchmark → generate optimization prompt → (human or LLM applies edit) → re-benchmark → commit if improved.

- [ ] **Step 1: Implement mini_loop.py**

```python
# src/aiter_forge/mini_loop.py
"""Minimal AVO optimization loop: target → baseline → optimize → commit best."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .evolution.controller import EvolutionController
from .evolution.scoring import BenchmarkResult, ScoringFunction
from .knowledge.base import KnowledgeBase
from .lineage.store import LineageStore


def load_target(target_dir: Path) -> dict:
    """Load target.yaml and resolve env vars."""
    cfg = yaml.safe_load((target_dir / "target.yaml").read_text())
    return cfg


def resolve_env(s: str) -> str:
    """Expand $VAR references in a string."""
    return os.path.expandvars(s)


def run_command(cmd: str, label: str) -> tuple[int, str]:
    """Run a shell command, return (returncode, stdout+stderr)."""
    print(f"[{label}] Running: {cmd}", file=sys.stderr)
    result = subprocess.run(
        resolve_env(cmd), shell=True, capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    print(f"[{label}] Exit code: {result.returncode}", file=sys.stderr)
    return result.returncode, output


def run_correctness(target: dict) -> bool:
    """Run correctness harness. Returns True if passed."""
    corr = target.get("correctness", {})
    cmd = corr.get("command")
    if not cmd:
        print("[correctness] No correctness command, skipping", file=sys.stderr)
        return True
    rc, output = run_command(cmd, "correctness")
    pass_pattern = corr.get("pass_pattern", "test passed")
    passed = pass_pattern.lower() in output.lower() and rc == 0
    if not passed:
        print(f"[correctness] FAILED. Output:\n{output[:500]}", file=sys.stderr)
    return passed


def run_benchmark(target: dict, scoring: ScoringFunction) -> list:
    """Run benchmark for each shape, return list of BenchmarkResults."""
    bench = target.get("benchmark", {})
    cmd_template = bench.get("command", "")
    shapes = bench.get("shapes", [{}])
    results = []
    for shape in shapes:
        cmd = cmd_template.format(**shape) if shape else cmd_template
        rc, output = run_command(cmd, f"benchmark shape={shape}")
        if rc != 0:
            print(f"[benchmark] Command failed: {output[:300]}", file=sys.stderr)
            continue
        parsed = scoring.parse(output)
        if parsed.valid:
            results.append(parsed)
        else:
            print(f"[benchmark] Could not parse metrics from output", file=sys.stderr)
    return results


def main(
    target_dir: str = "targets/aiter_mha",
    output_dir: str = "optimization_logs",
    max_rounds: int = 1,
):
    """Run the mini AVO loop."""
    target_path = Path(target_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    target = load_target(target_path)

    # Validate required fields
    required = [
        ("name", target.get("name")),
        ("kernel.path", target.get("kernel", {}).get("path")),
        ("correctness.command", target.get("correctness", {}).get("command")),
        ("benchmark.command", target.get("benchmark", {}).get("command")),
        ("benchmark.shapes", target.get("benchmark", {}).get("shapes")),
        ("scoring.primary_metric", target.get("scoring", {}).get("primary_metric")),
        ("benchmark.aggregate", target.get("benchmark", {}).get("aggregate")),
    ]
    for field_name, value in required:
        if not value:
            print(f"ERROR: target.yaml missing required field: {field_name}", file=sys.stderr)
            sys.exit(1)

    print(f"Target: {target['name']}", file=sys.stderr)

    # Initialize components
    scoring_cfg = target.get("scoring", {})
    scoring = ScoringFunction(
        primary_metric=scoring_cfg.get("primary_metric", "tflops"),
        higher_is_better=scoring_cfg.get("higher_is_better", True),
    )
    knowledge_dir = Path(__file__).parent / "knowledge" / "patterns"
    kb = KnowledgeBase(knowledge_dir)
    store = LineageStore(out_path / "lineage")
    store.load()

    kernel_path = target.get("kernel", {}).get("path", "")
    ctrl = EvolutionController(store, kb, scoring, kernel_path=kernel_path)

    # Step 1: Correctness check
    if not run_correctness(target):
        print("ERROR: Baseline correctness check failed. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: Baseline benchmark
    print("\n=== Baseline Benchmark ===", file=sys.stderr)
    baseline_results = run_benchmark(target, scoring)
    if not baseline_results:
        print("ERROR: No valid baseline results. Aborting.", file=sys.stderr)
        sys.exit(1)

    aggregate = target.get("benchmark", {}).get("aggregate", "last")
    if aggregate == "geomean" and len(baseline_results) > 1:
        baseline = scoring.aggregate_geomean(baseline_results)
    else:
        baseline = baseline_results[-1]

    baseline_variant = ctrl.commit_variant(
        round_num=0, parent_id=None,
        patch_path="baseline", result=baseline,
        description="original kernel", strategy="baseline",
    )
    metric_val = baseline.metrics.get(scoring.primary_metric, "N/A")
    print(f"\nBaseline: {scoring.primary_metric}={metric_val}", file=sys.stderr)

    # Step 3: Optimization rounds
    current_best_id = baseline_variant.variant_id
    for round_num in range(1, max_rounds + 1):
        print(f"\n=== Round {round_num}/{max_rounds} ===", file=sys.stderr)

        # Generate optimization prompt
        prompt = ctrl.generate_optimization_prompt(round_num, current_best_id)
        prompt_path = out_path / f"round_{round_num}_prompt.md"
        prompt_path.write_text(prompt)
        print(f"Optimization prompt saved to: {prompt_path}", file=sys.stderr)
        print(">>> Human-in-the-loop: apply the edit (manually or feed prompt to LLM), then press Enter <<<", file=sys.stderr)
        input()  # Phase 1: human gate. Phase 2 will replace with LiteLLM auto-edit.

        # Correctness gate
        if not run_correctness(target):
            print(f"Round {round_num}: FAILED correctness. Reverting.", file=sys.stderr)
            continue

        # Benchmark
        round_results = run_benchmark(target, scoring)
        if not round_results:
            print(f"Round {round_num}: No valid results.", file=sys.stderr)
            continue

        if aggregate == "geomean" and len(round_results) > 1:
            round_result = scoring.aggregate_geomean(round_results)
        else:
            round_result = round_results[-1]

        # Try commit
        committed = ctrl.try_commit(
            round_num=round_num,
            parent_id=current_best_id,
            patch_path=str(out_path / f"round_{round_num}.patch"),
            result=round_result,
            description=f"round {round_num} optimization",
            strategy="llm_suggested",
            correct=True,
        )

        rv = round_result.metrics.get(scoring.primary_metric, 0)
        if committed:
            speedup = scoring.speedup(round_result, baseline)
            print(f"Round {round_num}: COMMITTED. {scoring.primary_metric}={rv:.2f} "
                  f"(speedup={speedup:.3f}x vs baseline)", file=sys.stderr)
            current_best_id = committed.variant_id
        else:
            print(f"Round {round_num}: REJECTED (no improvement). {scoring.primary_metric}={rv:.2f}", file=sys.stderr)

    # Summary
    best = store.best(scoring.primary_metric, scoring.higher_is_better)
    if best:
        best_result = BenchmarkResult(metrics=best.metrics, valid=True)
        speedup = scoring.speedup(best_result, baseline) if baseline.valid else 0
        print(f"\n=== Final Result ===", file=sys.stderr)
        print(f"Best: {best.variant_id} ({scoring.primary_metric}={best.metrics.get(scoring.primary_metric)})", file=sys.stderr)
        print(f"Total variants committed: {len(store.all_variants())}", file=sys.stderr)

    # Save report
    report = {
        "target": target["name"],
        "baseline": baseline.metrics,
        "best": best.to_dict() if best else None,
        "total_rounds": max_rounds,
        "committed_variants": len(store.all_variants()),
    }
    (out_path / "report.json").write_text(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AITER-Forge mini optimization loop")
    parser.add_argument("--target", default="targets/aiter_mha", help="Target directory")
    parser.add_argument("--output", default="optimization_logs", help="Output directory")
    parser.add_argument("--rounds", type=int, default=1, help="Max optimization rounds")
    args = parser.parse_args()
    main(target_dir=args.target, output_dir=args.output, max_rounds=args.rounds)
```

- [ ] **Step 2: Commit**

```bash
git add src/aiter_forge/mini_loop.py
git commit -m "feat: mini_loop.py - runnable optimization loop (Phase 1 deliverable)"
```

---

### Task 10: Run All Tests + Push Feature Branch

- [ ] **Step 1: Run full test suite (including integration)**

Run: `cd /Users/pensun/aiter-forge && python -m pytest tests/ -v`
Expected: All tests PASS (unit + integration)

- [ ] **Step 2: Push feature branch**

```bash
git push -u origin feat/phase1-foundation
```

Note: The feature branch was already created in Step 0 (see Version Control Plan below). Do NOT create it here.

- [ ] **Step 3: Open PR and wait for human review**

```bash
gh pr create --title "Phase 1: Triton-first mini AVO harness" --body "..."
```

Fallback if `gh` is not available: push the branch and manually create PR on GitHub web UI.

Do NOT merge automatically. Wait for Peng's review.

---

### Task 11: GitHub Action — Automated Codex Review

**Files:**
- Create: `.github/workflows/codex-review.yml`
- Create: `automation/review_prompt.txt`

- [ ] **Step 1: Create review prompt template**

```
automation/review_prompt.txt
```

Standard review prompt that gets sent to OpenAI API with the diff. Covers: correctness, interface contract compliance, test coverage, stale references, security.

- [ ] **Step 2: Create GitHub Action workflow**

```yaml
# .github/workflows/codex-review.yml
name: Codex Code Review
on:
  push:
    branches: [feat/*]

jobs:
  codex-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2
      - name: Get diff
        run: git diff HEAD~1 > /tmp/diff.txt
      - name: Call OpenAI API for review
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python automation/run_review.py /tmp/diff.txt
      - name: Post review comment on PR
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh pr comment --body-file /tmp/review_result.md
```

- [ ] **Step 3: Create review runner script**

```
automation/run_review.py
```

- Reads diff + review prompt
- Calls OpenAI Responses API (model: current GPT-5.4 family, structured JSON output)
- Writes structured review to `/tmp/review_result.md` (human-readable) and `/tmp/review_result.json` (machine-readable)
- JSON schema matches Task 12's `review_result.json` format

- [ ] **Step 4: Test locally with a sample diff**

Run: `python automation/run_review.py sample_diff.txt`
Expected: Produces valid `review_result.md` and `review_result.json`

- [ ] **Step 5: Add OPENAI_API_KEY to GitHub repo secrets**

Manual step: `gh secret set OPENAI_API_KEY`

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/codex-review.yml automation/
git commit -m "feat: add GitHub Action for automated Codex code review"
```

---

### Task 12: Structured Review Output

**Files:**
- Create: `automation/review_schema.json`
- Modify: `automation/run_review.py` (from Task 11)

- [ ] **Step 1: Define review result JSON schema**

```json
{
  "task_id": "string",
  "commit_sha": "string",
  "summary": "string",
  "status": "pass | fail | needs_revision",
  "findings": [
    {
      "severity": "high | medium | low",
      "category": "correctness | interface | coverage | style | security",
      "file": "string",
      "line": "number | null",
      "message": "string"
    }
  ],
  "highest_severity": "high | medium | low | none",
  "blocking": "boolean",
  "recommended_action": "proceed | fix_and_retry | human_review"
}
```

- [ ] **Step 2: Update run_review.py to output structured JSON**

Ensure OpenAI API response is parsed into the schema above. Write to `automation/review_result.json`.

- [ ] **Step 3: Add validation test**

```python
# tests/test_review_schema.py
def test_review_result_schema():
    """Validate review_result.json matches schema."""
```

- [ ] **Step 4: Commit**

```bash
git add automation/review_schema.json automation/run_review.py tests/test_review_schema.py
git commit -m "feat: structured JSON output for Codex review results"
```

---

### Task 13: Machine-Readable Task Spec

**Files:**
- Create: `automation/tasks.yaml`
- Create: `automation/task_schema.py`

- [ ] **Step 1: Create tasks.yaml from PLAN.md**

Each task entry contains:

```yaml
tasks:
  - id: 1
    title: "Project Scaffolding"
    phase: A
    inputs: []
    outputs:
      - "pyproject.toml"
      - "src/aiter_forge/__init__.py"
    completion_criteria: "pip install -e . succeeds"
    required_checks:
      - "pytest tests/ -v"
    human_gate: false
    retry_policy: {max_retries: 2, strategy: "fix_and_retry"}
    next_task: 2
  # ... all tasks 1-10
```

- [ ] **Step 2: Create schema validator**

```python
# automation/task_schema.py
def validate_tasks(path: str) -> list[str]:
    """Validate tasks.yaml, return list of errors (empty = valid)."""
```

- [ ] **Step 3: Write validation test**

```python
# tests/test_task_schema.py
def test_tasks_yaml_valid():
    errors = validate_tasks("automation/tasks.yaml")
    assert errors == []
```

- [ ] **Step 4: Commit**

```bash
git add automation/tasks.yaml automation/task_schema.py tests/test_task_schema.py
git commit -m "feat: machine-readable task spec for orchestrator"
```

---

### Task 14: Orchestrator + State Machine

**Files:**
- Create: `automation/orchestrator.py`
- Create: `automation/state.json`
- Create: `automation/task_runner.py`
- Create: `tests/test_orchestrator.py`

- [ ] **Step 1: Define state schema**

```json
{
  "current_task": 2,
  "completed_tasks": [1],
  "last_commit": "661caee",
  "last_review_status": "pass",
  "last_failure_reason": null,
  "retry_count": 0,
  "mode": "running | paused | await-human | failed | completed"
}
```

- [ ] **Step 2: Create task_runner.py**

Responsibilities:
- Accept single task spec from tasks.yaml
- Invoke Claude Code / Codex worker to implement the task
- Run required_checks locally
- Commit + push on success
- Return structured execution report

- [ ] **Step 3: Create orchestrator.py**

Responsibilities:
- Load `tasks.yaml` and `state.json`
- Find next uncompleted task
- Call `task_runner` for that task
- Wait for CI / review result (poll `review_result.json` or GitHub Action status)
- Apply gate rules:
  - pytest fail → `mode: paused`
  - review `blocking: true` → `mode: paused`
  - review `status: pass` + tests green → advance to next task
  - `human_gate: true` → `mode: await-human`
  - retry_count > max_retries → `mode: failed`
- Update `state.json` after each transition

- [ ] **Step 4: Write orchestrator tests**

```python
# tests/test_orchestrator.py
def test_advance_on_pass():
    """Green tests + passing review → advance to next task."""

def test_pause_on_test_failure():
    """Test failure → mode: paused."""

def test_pause_on_blocking_review():
    """Review with blocking finding → mode: paused."""

def test_human_gate():
    """Task with human_gate: true → mode: await-human."""

def test_retry_then_fail():
    """Exceed max_retries → mode: failed."""

def test_state_persistence():
    """State survives restart — orchestrator resumes from last checkpoint."""
```

- [ ] **Step 5: Commit**

```bash
git add automation/ tests/test_orchestrator.py
git commit -m "feat: orchestrator with state machine and gate rules"
```

---

### Automation Boundary for Phase 1

**Fully automated (no human gate):**
- Task 1-5 (pure Python modules)
- Task 8 (integration test)
- Task 8.5 (harness tests)

**Requires human gate:**
- Task 6-7 (MI355X / SSH / real hardware verification)
- Any correctness anomaly where benchmark looks improved but correctness fails
- Any suggestion to modify Triton / FlyDSL compiler behavior

**Gate Rules:**
- `pytest` failure → stop
- Schema validation failure → stop
- Artifact validation failure → stop
- Review finding with `severity: high` and `blocking: true` → stop
- Tests green + review non-blocking → proceed
- Task with `human_gate: true` → `mode: await-human`
- Compiler / FlyDSL escalation → `mode: expert-review-required`

---

## Test Plan

### Overview

6 test files, 32 test functions, 3.5 layers of testing (unit → component → harness → manual e2e). All automated tests run locally on macOS without GPU. MI355X e2e validation is manual (not automated in pytest).

### Test Matrix

| Test File | Task | Layer | Tests | What It Covers |
|-----------|------|-------|-------|----------------|
| `tests/test_lineage_store.py` | Task 2 | 1 | 5 | CRUD, chaining, best-variant query, persistence, prompt formatting |
| `tests/test_knowledge_base.py` | Task 3 | 1 | 4 | Pattern loading, single-pattern get, relevance query, prompt formatting |
| `tests/test_scoring.py` | Task 4 | 1 | 10 | Triton perf_report parsing (single/multi-row), geomean, compare, speedup + defensive edge cases |
| `tests/test_evolution_controller.py` | Task 5 | 1 | 5 | Initial prompt gen, prompt with lineage, commit on improvement, reject on regression, reject on incorrect |
| `tests/test_integration.py` | Task 8 | 2 | 1 | Full pipeline: parse → commit_variant → try_commit (3 scenarios) → lineage verification → prompt generation |
| `tests/test_mini_loop.py` | Task 8.5 | 2.5 | 11 | Orchestration glue: correctness gate, benchmark multi-shape, main() commit/reject/abort paths, report.json, required fields validation |

**Total: 24 unit tests + 1 integration test + 11 harness tests = 36 tests**

### Layer 1: Unit Tests (Task 2-5)

Each module is tested in isolation with in-memory or tmp_path fixtures.

**LineageStore (5 tests):**
- `test_add_and_retrieve_variant` — add a variant, get it back by ID
- `test_lineage_chain` — 3 variants with parent links, verify chain order
- `test_best_variant` — find best by metric name + direction (higher_is_better)
- `test_persistence` — save to disk, reload from new instance, verify data survives
- `test_format_lineage_prompt` — verify prompt string contains variant IDs and metrics

**KnowledgeBase (4 tests):**
- `test_load_patterns` — load 3 markdown pattern files, verify all found
- `test_get_pattern` — get single pattern by name, verify content
- `test_query_relevant` — query by keyword, verify relevant patterns returned
- `test_format_knowledge_prompt` — verify prompt string has section header

**ScoringFunction (10 tests):**
- `test_parse_triton_perf_report` — parse real `bench_mha.py` output format (single row), verify `tflops` extraction
- `test_parse_multi_row_perf_report` — 3-row table (3 shapes), verify all rows parsed with correct values
- `test_geomean_aggregate` — geometric mean of [10, 40] = 20, verify math
- `test_compare` — `is_better(50, 45)` = True, `is_better(45, 50)` = False
- `test_speedup` — 50/40 = 1.25x, verify calculation
- `test_parse_no_header` — no table header → `parse_all_rows()` returns [], `parse()` returns invalid
- `test_parse_header_but_missing_target_column` — header exists but target metric column missing → []
- `test_parse_non_numeric_value` — row with N/A in metric column → skip that row, parse remaining
- `test_geomean_empty_input` — empty list → invalid result
- `test_geomean_with_zero` — list containing 0 → result is 0 (not NaN/error)

**EvolutionController (5 tests):**
- `test_generate_initial_prompt` — empty lineage, verify prompt has kernel path + knowledge
- `test_generate_prompt_with_lineage` — 2 variants in store, verify prompt includes both IDs + metrics
- `test_commit_variant_on_improvement` — baseline 40 → candidate 50 (correct) → committed
- `test_reject_variant_on_regression` — baseline 40 → candidate 35 (correct) → rejected, lineage still 1
- `test_reject_variant_on_incorrect` — baseline 40 → candidate 60 (incorrect) → rejected despite better metric

### Layer 2: Integration Test (Task 8)

Tests the `Pt + K + f` pipeline in-process without any subprocess mocking.

`test_full_pipeline` exercises the complete in-process pipeline:

```
Parse perf_report (3 shapes) → commit_variant (baseline)
  → try_commit (improved, correct=True)  → committed ✓
  → try_commit (regressed, correct=True)  → rejected ✗
  → try_commit (incorrect, correct=False) → rejected ✗
  → verify len(committed_lineage) == 2
  → generate_optimization_prompt → verify lineage + knowledge + objective in output
```

Uses real Triton `perf_report` table format as test data, not toy strings.

### Layer 2.5: Local Harness Tests (Task 8.5)

Tests the orchestration glue in `mini_loop.py` without GPU/SSH. All external commands are monkeypatched.

**`test_mini_loop.py` (7 tests):**
- `test_run_correctness_pass` — mock rc=0 + "test passed" → True
- `test_run_correctness_fail_on_nonzero_exit` — mock rc=1 + "test passed" → False (exit code takes precedence)
- `test_run_correctness_fail_on_missing_pattern` — mock rc=0 + wrong output → False
- `test_run_benchmark_parses_multiple_shapes` — 3 shapes × fake perf_report → 3 BenchmarkResults with correct values
- `test_run_benchmark_skips_failed_commands` — 1 shape fails → skipped, returns remaining 2
- `test_main_baseline_commit_and_report` — 0 rounds, verify report.json written with baseline metrics
- `test_main_commits_improved_variant` — 1 round, improved metrics → committed, variants=2
- `test_main_rejects_regressed_variant` — 1 round, worse metrics → rejected, variants=1
- `test_main_aborts_when_baseline_correctness_fails` — correctness rc=1 → SystemExit(1)
- `test_main_aborts_when_no_valid_baseline_results` — unparseable benchmark output → SystemExit(1)
- `test_main_aborts_on_missing_required_field` — target.yaml missing required field → SystemExit(1)

Note: Test count in matrix shows 7 concerns; actual test functions are 11 to cover all branches.

### Layer 3: Manual E2E Validation (post-PR)

Not in pytest. Performed by human (Peng) on MI355X after PR merge:

| Step | Command | Expected |
|------|---------|----------|
| 1. Environment | `scripts/verify_mi355.sh` | ROCm version, GPU count, Triton import OK |
| 2. Correctness | Run `correctness.command` from `target.yaml` | "test passed" and exit code 0 |
| 3. Baseline benchmark | `python -m aiter_forge.mini_loop targets/aiter_mha` | Baseline TFLOPS printed for 3 shapes |
| 4. Human edit round | Apply a manual tweak, press Enter | Correctness + benchmark re-run, commit/reject decision |
| 5. Report | Check `optimization_logs/report.json` | baseline + best variant recorded |

### Test Execution Commands

```bash
# Run all tests (Phase A gate — must pass before any commit)
python -m pytest tests/ -v

# Run single module test
python -m pytest tests/test_lineage_store.py -v
python -m pytest tests/test_knowledge_base.py -v
python -m pytest tests/test_scoring.py -v
python -m pytest tests/test_evolution_controller.py -v

# Run integration test only (Layer 2)
python -m pytest tests/test_integration.py -v

# Run harness tests only (Layer 2.5)
python -m pytest tests/test_mini_loop.py -v

# Run with coverage (optional)
python -m pytest tests/ -v --cov=aiter_forge --cov-report=term-missing
```

### Test Data

All test data is inline (no external fixtures needed):

- **Triton perf_report tables**: hardcoded strings matching real `bench_mha.py` output format with `fwd(TFLOPS)` column
- **Knowledge patterns**: tmp_path markdown files created in-test
- **LineageStore**: tmp_path directory, auto-cleaned by pytest
- **BenchmarkResult**: constructed directly in test code

### What Is NOT Tested (automated)

- Network/SSH connectivity to MI355X (manual verification)
- Real `bench_mha.py` execution (requires GPU)
- Real kernel correctness check (requires GPU)
- `mini_loop.py` with real subprocess calls (Layer 2.5 uses monkeypatch)
- `target.yaml` env var resolution against real MI355X environment

These are all validated in Layer 3 (manual E2E on MI355X).

### What IS Now Tested (via Layer 2.5 monkeypatch)

- `mini_loop.py` orchestration logic end-to-end
- correctness gate pass/fail/missing-pattern branches
- benchmark multi-shape parsing + failed-command skipping
- geomean aggregation in main()
- commit/reject decision paths
- baseline correctness fail → abort
- no valid baseline results → abort
- report.json output format and content

---

## Test Failure Policy

Rules for what happens when tests fail at each layer. Subagents must follow these without exception.

| Layer | Failure | Action |
|-------|---------|--------|
| Layer 1 (unit) | Any test fails | **Block commit.** Fix before proceeding. |
| Layer 2 (integration) | Test fails | **Block commit.** Fix before proceeding. |
| Layer 2.5 (harness) | Any test fails | **Block commit.** Fix before proceeding. |
| Layer 3 (manual E2E) | `verify_mi355.sh` fails | **Block "Phase 1 complete" claim.** May still open PR with `manual-e2e-pending` label. |
| Layer 3 (manual E2E) | Benchmark or correctness fails on MI355X | **Do not claim "validated on MI355X".** PR stays open until resolved. |

Additional hard rules:
- **correctness fail + benchmark pass = reject.** Correctness is always a hard gate. No human override allowed.
- **Benchmark output unparseable = test failure.** Do not treat as "no improvement" and continue — abort the round.
- **All Layer 1 + 2 + 2.5 green, but Layer 3 not yet run:** allowed to open PR, but must tag `manual-e2e-pending`.

---

## Artifacts Validation

Beyond tests passing, these artifacts must exist and be well-formed before Task 10 (push + PR).

### Required Artifacts (checked by Layer 2.5 + manual)

| Artifact | Verified By | Check |
|----------|-------------|-------|
| `optimization_logs/report.json` | Layer 2.5 `test_main_baseline_commit_and_report` | exists, valid JSON, has `target`, `baseline`, `committed_variants` fields |
| `optimization_logs/lineage/lineage.json` | Layer 1 `test_persistence` | exists after LineageStore.save(), valid JSON, variants recoverable |
| `round_*_prompt.md` | Layer 2.5 `test_main_commits_improved_variant` | exists after optimization round, non-empty |
| `targets/local.env.example` | Task 6 implementation | exists, contains `AITER_ROOT` template |

### Required Artifacts (checked by Layer 3 manual E2E)

| Artifact | Check |
|----------|-------|
| `report.json` from real MI355X run | baseline metrics are real TFLOPS values (not 0 or mock) |
| `verify_mi355.sh` output | ROCm version printed, GPU count > 0, Triton import succeeds |
| PR description | Layer 3 manual verification results pasted or linked |

---

## Interface Contract

Stable interfaces for Phase 1. **Any change to these requires updating the corresponding Layer 1 / 2 / 2.5 tests.**

### `target.yaml` schema

```yaml
name: string                    # human-readable target name
kernel:
  path: string                  # relative path within $AITER_ROOT
  repo_env: string              # env var pointing to repo root (e.g. AITER_ROOT)
correctness:
  command: string               # shell command, supports $ENV_VAR
  pass_pattern: string          # substring to match in stdout (case-insensitive)
scoring:
  primary_metric: string        # must match a COLUMN_PATTERNS key (e.g. "tflops")
  higher_is_better: bool
benchmark:
  command: string               # shell command template with {batch}, {hq}, etc.
  shapes: list[dict]            # each dict has keys matching command template placeholders
  aggregate: "geomean" | "last" # how to combine multi-shape results
```

### Metric key convention

- `tflops` — maps to Triton perf_report columns matching `fwd(TFLOPS)`, `bwd(TFLOPS)`, `TFLOPS`
- `bandwidth_gbps` — maps to `fwd(GB/s)`, `bwd(GB/s)`, `GB/s`
- `time_ms` — maps to `ms`, `time_ms`

Column headers are mapped internally by `ScoringFunction._find_metric_column()`. External consumers (target.yaml, tests, controller) always use the short key.

### `EvolutionController` commit semantics

- `commit_variant(round_num, parent_id, patch_path, result, description, strategy)` — unconditional commit. Used for baseline only.
- `try_commit(..., correct: bool)` — commit only if `correct=True` AND `result` is better than current best. Returns `None` if rejected.
- Only committed variants enter `LineageStore`. Rejected attempts are logged but never in lineage.

### `report.json` schema

```json
{
  "target": "string",
  "baseline": {"tflops": 0.0},
  "best": {"variant_id": "...", "metrics": {...}, ...} | null,
  "total_rounds": 0,
  "committed_variants": 0
}
```

---

## Environment Contract

### Phase 1 hard requirements

| Requirement | Details |
|-------------|---------|
| **Hardware** | MI355X (CDNA4). Nodes: `mi355-gpu-9`, `mi355-gpu-15` |
| **Python** | >= 3.10 |
| **ROCm** | Version detected by `verify_mi355.sh` (no minimum pinned yet) |
| **AITER** | Must be cloned and accessible. Path set via `$AITER_ROOT` env var. |
| **Env var** | `AITER_ROOT` — required for `target.yaml` command resolution |

### Phase 1 soft requirements (fallback allowed)

| Requirement | Fallback |
|-------------|----------|
| `gh` (GitHub CLI) | Push branch manually + create PR via GitHub web UI |
| SSH to MI355X | Layer 3 deferred. PR opened with `manual-e2e-pending` label. |

### Local development (macOS, no GPU)

All Layer 1 / 2 / 2.5 tests run locally. Required:
- Python >= 3.10
- `pip install -e ".[dev]"` (pyyaml, jinja2, pytest)
- No GPU, no ROCm, no SSH needed

---

## Guardrails

Hard rules for Phase 1 execution. These apply to both human developers and subagents.

1. **Correctness is a hard gate.** No variant enters committed lineage without passing correctness. No exceptions, no manual override.
2. **Only improvements commit.** `try_commit()` rejects any variant that does not beat current best on `primary_metric`.
3. **No compiler modifications in Phase 1.** Do not modify Triton compiler, FlyDSL compilation stack, or any codegen behavior. Compiler is a static target.
4. **No "validated on MI355X" without Layer 3.** Until `verify_mi355.sh` passes and at least one real benchmark run completes on MI355X, do not claim hardware validation.
5. **Compiler-side proposals require human review.** If analysis suggests a compiler change would help, log it as a recommendation but do not implement. Route to the human review lane described in the Compiler Involvement section.
6. **Phase 1 scope is frozen.** Do not add GEAK integration, FlyDSL targets, multi-branch evolutionary population, or LLM-driven autonomous edits. These are Phase 2.
7. **Unparseable benchmark output = failure.** Do not silently skip or treat as "no improvement". Abort the round and surface the error.

---

## Layer 3 Manual E2E Runbook

Step-by-step instructions for validating Phase 1 on MI355X. Run after all automated tests (Layer 1/2/2.5) pass and PR is opened.

### Prerequisites

```bash
# SSH to MI355X node
ssh mi355-gpu-9   # or mi355-gpu-15

# Verify AITER_ROOT is set
echo $AITER_ROOT

# Clone aiter-forge and install
git clone https://github.com/sunway513/aiter-forge.git
cd aiter-forge && git checkout feat/phase1-foundation
pip install -e ".[dev]"
```

### Step 1: Environment verification

```bash
./scripts/verify_mi355.sh
```

**Pass criteria:** ROCm version printed, GPU count > 0, `python -c "import triton"` succeeds.
**On failure:** Record output. Do not proceed. Update PR with `env-verification-failed` label.

### Step 2: Correctness check (unmodified kernel)

```bash
# Use the exact command from target.yaml correctness.command:
cd $AITER_ROOT && python op_tests/op_benchmarks/triton/bench_mha.py -b 4 -hq 16 -hk 16 -d 128 -sq 4096 -sk 4096 -test_mode
```

**Pass criteria:** Output contains `test passed` and exit code 0.
**On failure:** Record output. Mark PR as `correctness-baseline-failed`. Do not proceed to benchmark.

### Step 3: Baseline benchmark (3 shapes)

```bash
python -m aiter_forge.mini_loop --target targets/aiter_mha --output optimization_logs --rounds 0
```

**Pass criteria:**
- Runs without error
- `optimization_logs/report.json` exists with real TFLOPS values (not 0 or mock)
- Baseline TFLOPS printed for 3 shapes + geomean
- `optimization_logs/lineage/lineage.json` contains 1 committed variant (baseline)

**On failure:** Record stderr + report.json content. Mark PR as `baseline-benchmark-failed`.

### Step 4: One optimization round (human edit)

```bash
python -m aiter_forge.mini_loop --target targets/aiter_mha --output optimization_logs --rounds 1
```

1. System prints optimization prompt, saves to `optimization_logs/round_1_prompt.md`
2. Apply a trivial edit to the kernel (e.g. change a comment, or a real optimization)
3. Press Enter
4. System runs correctness → benchmark → commit/reject

**Pass criteria:**
- Correctness gate runs and returns pass/fail
- Benchmark runs across 3 shapes
- Commit/reject decision printed
- `report.json` updated with final state

**On failure:** Record stderr. Note which step failed.

### Step 5: Validate report

```bash
cat optimization_logs/report.json | python -m json.tool
```

**Pass criteria:** Valid JSON with `target`, `baseline`, `best`, `total_rounds`, `committed_variants` fields.

### Posting results to PR

Paste into the PR comment:
1. `verify_mi355.sh` output (ROCm version, GPU info)
2. Baseline TFLOPS per shape + geomean
3. Round 1 result (committed or rejected, with TFLOPS)
4. `report.json` content
5. Any errors encountered

Remove `manual-e2e-pending` label after successful validation. Add `manual-e2e-passed`.

---

## Performance Measurement Protocol

Rules for benchmark reproducibility in Phase 1.

### Fixed shapes

All benchmark runs use the 3 shapes defined in `targets/aiter_mha/target.yaml`:

| Name | batch | hq | hk | d | sq | sk | Profile |
|------|-------|----|----|---|----|----|---------|
| GQA decode | 1 | 32 | 8 | 128 | 1 | 8192 | Latency-bound |
| MHA prefill | 4 | 16 | 16 | 128 | 4096 | 4096 | Compute-bound |
| MHA medium | 1 | 48 | 48 | 128 | 1024 | 1024 | Balanced |

### Aggregation

Geomean across all 3 shapes. Single-shape results are for diagnostics only — commit/reject decisions use the geomean.

### Run protocol

- **Minimum runs:** 1 (Triton's `perf_report` already does internal warmup + averaging via `@triton.testing.perf_report`)
- **Compile cache:** Allowed. Triton caches compiled kernels by default. Do not clear cache between shapes within the same run.
- **GPU warmup:** Not required beyond Triton's built-in warmup. If results seem noisy, run twice and take the second run.

### Required environment metadata

Every benchmark result posted to a PR or stored in `report.json` should be accompanied by:

```bash
# Capture with:
rocm-smi --showproductname   # GPU model
rocminfo | grep "Name:" | head -1  # GPU name
python -c "import torch; print(torch.version.hip)"  # ROCm/HIP version
cd $AITER_ROOT && git rev-parse --short HEAD  # AITER commit
cat targets/aiter_mha/target.yaml | head -3   # target config version
```

This ensures benchmark results are comparable across runs and reviewers can reproduce.

---

## Target Schema Required Fields

These fields in `target.yaml` are **required**. `mini_loop.py` must validate their presence at startup and fail fast with a clear error if any are missing.

```python
REQUIRED_FIELDS = {
    "name": "top-level target name",
    "kernel.path": "path to kernel source file",
    "correctness.command": "shell command for correctness check",
    "benchmark.command": "shell command template for benchmark",
    "benchmark.shapes": "list of shape dicts for multi-shape benchmark",
    "scoring.primary_metric": "metric key (e.g. tflops)",
    "benchmark.aggregate": "aggregation method (geomean or last)",
}
```

**Behavior on missing required field:** `sys.exit(1)` with message `"ERROR: target.yaml missing required field: {field_name}"`.

**Optional fields with defaults:**
- `correctness.pass_pattern` → defaults to `"test passed"`
- `scoring.higher_is_better` → defaults to `True`
- `kernel.repo_env` → defaults to `"AITER_ROOT"`

---

## Resume / Block Policy

What to do when something fails during execution.

### Task-level blocking

| Failure | Blocks | Action |
|---------|--------|--------|
| Layer 1/2/2.5 test fails in Task N | Task N only | Fix the failing test or implementation. Do not proceed to Task N+1. |
| `pytest` import error | Task N only | Fix missing dependency or module path. |
| Git commit rejected by hook | Task N only | Fix the issue flagged by the hook. Create a new commit (do not amend). |

### Phase-level blocking

| Failure | Blocks | Action |
|---------|--------|--------|
| Task 5 (EvolutionController) fails | Task 8, 8.5, 9 | All downstream tasks depend on controller. Must resolve first. |
| Task 9 (mini_loop.py) fails | Task 8.5, 10 | Harness tests and final push depend on mini_loop. |
| Layer 3 `verify_mi355.sh` fails | "Phase 1 complete" claim | May still open PR with `manual-e2e-pending`. |
| Layer 3 real benchmark fails | "Validated on MI355X" claim | PR stays open. Debug on MI355X. |

### PR opening policy

| Automated test status | Layer 3 status | Allowed? |
|----------------------|----------------|----------|
| All green | Passed | Open PR, merge after review |
| All green | Not yet run | Open PR with `manual-e2e-pending` label |
| All green | Failed | Open PR with `manual-e2e-blocked` label, do not merge |
| Any red | Any | Do NOT open PR |

---

## Minimal Design Standards

Phase 1 coding conventions. Keep it short — these are the only rules.

1. **Python 3.10+** — use `X | Y` union syntax, `match` statements allowed.
2. **Type hints on public API** — all public functions in `controller.py`, `scoring.py`, `store.py`, `mini_loop.py` must have parameter and return type annotations.
3. **New public API must have tests** — no public function ships without at least one test covering its happy path.
4. **File naming is fixed** — do not rename these without updating all imports and tests:
   - `optimization_logs/report.json`
   - `optimization_logs/lineage/lineage.json`
   - `round_{N}_prompt.md`
   - `targets/local.env.example`
5. **No new heavy dependencies in Phase 1** — only `pyyaml`, `jinja2`, and standard library. Adding a new dependency requires explicit plan approval.
6. **JSON/YAML schemas are stable** — changes to `target.yaml`, `report.json`, or `lineage.json` structure require updating Interface Contract + affected tests.
7. **No dead code** — do not leave commented-out code, unused imports, or placeholder functions. If it's not called, delete it.

---

## Execution Plan

### Executor

Claude Code (this session or a new session), using **superpowers:subagent-driven-development** — one subagent per Task, sequential execution (Tasks have interface dependencies).

No Cloud Code teams needed. Reasons:
1. Tasks are a linear dependency chain (Task 2 depends on Task 1's skeleton, Task 5 depends on Tasks 2-4, etc.)
2. Code volume is small (one file + one test file per task)
3. MI355X work is SSH-based, not cloud CI

### Execution Phases

**Phase A: Local development (Task 1-5, 8) — no GPU required**

All pure Python modules + pytest. Runs entirely on macOS.

| Order | Task | Depends On | Output |
|-------|------|------------|--------|
| 1 | Task 1: Project skeleton | — | pyproject.toml, directory structure, __init__.py |
| 2 | Task 2: LineageStore | Task 1 | src/aiter_forge/lineage/ + tests/test_lineage_store.py |
| 3 | Task 3: KnowledgeBase | Task 1 | src/aiter_forge/knowledge/ + tests/test_knowledge_base.py |
| 4 | Task 4: ScoringFunction | Task 1 | src/aiter_forge/evolution/scoring.py + tests/test_scoring.py |
| 5 | Task 5: EvolutionController | Tasks 2-4 | src/aiter_forge/evolution/controller.py + tests/test_evolution_controller.py |
| 6 | Task 8: Integration test | Tasks 2-5 | tests/test_integration.py |

Task 3 and Task 4 are independent of each other — a subagent could parallelize them, but the time savings is marginal given their size.

Note: Task 8.5 (test_mini_loop.py) depends on Task 9 (mini_loop.py) and is implemented in Phase C.

**Phase B: Target config + MI355X verification (Task 6-7)**

| Order | Task | Depends On | Output |
|-------|------|------------|--------|
| 7 | Task 6: Target config | Task 1 | targets/aiter_mha/target.yaml, targets/local.env.example |
| 8 | Task 7: MI355X verify script | — | scripts/verify_mi355.sh |

Task 6 can be written locally. Task 7 must be validated via SSH to MI355 node.

**Checkpoint:** After Phase B, manually SSH to MI355 and run `scripts/verify_mi355.sh` to confirm environment.

**Phase C: Runnable loop + ship (Task 9-10)**

| Order | Task | Depends On | Output |
|-------|------|------------|--------|
| 9 | Task 9: mini_loop.py | Tasks 2-6 | src/aiter_forge/mini_loop.py |
| 10 | Task 8.5: Harness tests | Task 9 | tests/test_mini_loop.py |
| 11 | Task 10: Full test + push + PR | Tasks 1-9, 8.5 | PR on GitHub |

Task 9 is written locally. Task 8.5 adds monkeypatched harness tests to verify orchestration logic without GPU. Real e2e validation (loading target, running benchmark on MI355X) is done manually after the PR.

**Phase D: Unattended automation infra (Task 11-14)**

| Order | Task | Depends On | Output |
|-------|------|------------|--------|
| 12 | Task 11: GitHub Action Codex review | Task 10 | `.github/workflows/codex-review.yml`, `automation/run_review.py` |
| 13 | Task 12: Structured review output | Task 11 | `automation/review_schema.json`, `automation/review_result.json` |
| 14 | Task 13: Machine-readable task spec | Task 10 | `automation/tasks.yaml`, `automation/task_schema.py` |
| 15 | Task 14: Orchestrator + state machine | Tasks 11-13 | `automation/orchestrator.py`, `automation/state.json`, `automation/task_runner.py` |

Phase D builds the unattended automation loop. Task 11 provides immediate value (auto-review on push). Tasks 12-14 progressively add automation until the system can execute tasks without human intervention (except for `human_gate` tasks).

### What the subagent sees

Each subagent receives:
- The full `docs/PLAN.md` (this file)
- The specific Task number to execute
- Access to the repo working directory on `feat/phase1-foundation` branch

Each subagent follows TDD: write failing test → verify fail → implement → verify pass → commit.

---

## Version Control Plan

### Branch Strategy

```
main (protected)
 └── feat/phase1-foundation (Task 1-10 commits land here)
     └── feat/automation-infra (Task 11-14 commits land here, branched after Task 10 merges)
```

- `main` stays clean — only receives code via reviewed PR
- All development happens on `feat/phase1-foundation`
- No worktrees (linear dependency chain, no parallel branch work)

### Step 0: Bootstrap (before Task 1)

```bash
# Initial commit on main — docs only
git add README.md docs/
git commit -m "docs: initial project plan and review"

# Create feature branch immediately
git checkout -b feat/phase1-foundation
```

### Per-Task Commits

Each task produces exactly one atomic commit on `feat/phase1-foundation`:

```
docs: initial project plan and review              (Step 0, on main)
feat: project skeleton and pyproject.toml          (Task 1)
feat: LineageStore with committed-only semantics   (Task 2)
feat: KnowledgeBase pattern loader                 (Task 3)
feat: ScoringFunction for Triton perf_report       (Task 4)
feat: EvolutionController with try_commit          (Task 5)
feat: AITER MHA target config                      (Task 6)
feat: MI355X environment verification script       (Task 7)
test: end-to-end integration smoke test            (Task 8)
feat: mini_loop.py human-in-the-loop harness       (Task 9)
test: Layer 2.5 local harness tests for mini_loop  (Task 8.5)
```

### Ship (Task 10)

```bash
git push -u origin feat/phase1-foundation
gh pr create --title "Phase 1: Triton-first mini AVO harness"
# Wait for Peng's review → merge to main
```

### Rules

1. **No commits to main after Step 0** — all work on feature branch
2. **One commit per task** — atomic, revertable, reviewable
3. **No squash at merge** — preserve per-task history for traceability
4. **No force push** — linear history, no rebase
5. **Tests must pass before each commit** — TDD enforced by subagent skill

---

## Phase 1 Success Criteria

Phase 1 is complete when `mini_loop.py` can:
1. Load `targets/aiter_mha/target.yaml`
2. Run correctness check on unmodified kernel (pass)
3. Run baseline benchmark across 3 shapes and report geomean TFLOPS
4. Generate an optimization prompt with lineage + knowledge + objective
5. After human applies an edit: re-run correctness + benchmark
6. Commit only if correct AND improved; reject otherwise
7. Produce `optimization_logs/report.json` with baseline and best variant

## Dashboard Integration: Closed-Loop Operator Optimization

AITER-Forge connects two existing dashboards to form a complete optimization and validation pipeline:

```
┌─────────────────────────────┐
│  project-dashboard          │  sunway513/project-dashboard
│  (AI Operator Tracking)     │  Tracks 12 AMD GPU ecosystem projects
│                             │
│  AITER operators ──────────────┐
│  ATOM operators  ──────────────┤  Tuning target source
│  FlyDSL operators ─────────────┘
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  aiter-forge                 │  This project
│  (Kernel Tuning Engine)     │
│                             │
│  baseline → optimize →      │
│  correctness → benchmark →  │
│  commit best variant        │
└─────────────────────────────┘
              │
              ▼
┌─────────────────────────────┐
│  ATOM Benchmark Dashboard   │  rocm.github.io/ATOM/benchmark-dashboard/
│  (E2E Validation)           │
│                             │
│  Metrics:                   │
│  - Throughput (tok/s)       │  Per model × backend × ISL/OSL × concurrency
│  - TPOT (ms)                │  Time per output token
│  - TTFT (ms)                │  Time to first token
│  - Accuracy                 │  Correctness validation
│  - Regression detection     │  Auto-flags perf drops
│                             │
│  mi355x (baseline)          │  Before kernel tuning
│  mi355x_tuned               │  After aiter-forge optimization
└─────────────────────────────┘
```

### Phase 1 scope (this plan)

- **Input:** Single AITER Triton attention kernel (`mha.py`) as tuning target, manually selected
- **Output:** `report.json` with baseline vs best TFLOPS at operator level
- **E2E validation:** Not yet integrated with ATOM dashboard

### Phase 2+ integration roadmap

1. **project-dashboard → aiter-forge target generation**
   - Auto-identify candidate operators from project-dashboard tracked repos (AITER, ATOM)
   - Generate `target.yaml` configs for each candidate operator
   - Priority ranking based on: operator usage frequency, perf gap vs CUDA, regression history

2. **aiter-forge → ATOM benchmark dashboard**
   - After operator-level tuning produces a committed variant, run ATOM e2e benchmarks
   - Compare `mi355x` (stock AITER operators) vs `mi355x_tuned` (aiter-forge optimized)
   - Track: throughput impact (tok/s), latency impact (TPOT/TTFT), accuracy preservation
   - Leverage ATOM dashboard's regression detection to ensure tuned operators don't degrade e2e

3. **Closed-loop feedback**
   - ATOM dashboard regression alerts → trigger re-tuning via aiter-forge
   - project-dashboard tracks tuned operator PRs as they flow back into AITER/ATOM

## Post Phase 1: Next Steps

Phase 2 (after Triton path proves out):
1. **LLM-driven edits** — Replace `input()` wait with LiteLLM call that generates code edits
2. **Multi-round autonomous** — Run N rounds without human intervention
3. **GEAK integration** (optional) — Wire as GEAK tool for access to its toolset
4. **FlyDSL support** — Add FlyDSL kernel targets once Triton path is validated
5. **Compiler observability** — Structured compile diagnostics (VGPR, LDS, MFMA stats) per Codex's recommendation
6. **Dashboard integration** — Wire project-dashboard as target source + ATOM benchmark dashboard as e2e validation (see Dashboard Integration section above)
