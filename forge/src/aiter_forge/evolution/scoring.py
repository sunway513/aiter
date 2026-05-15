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
        values = [r.metrics.get(self.primary_metric) for r in results
                  if self.primary_metric in r.metrics]
        if not values:
            return BenchmarkResult(valid=False)
        # Handle zero: if any value is zero, geomean is zero
        if any(v == 0.0 for v in values):
            return BenchmarkResult(metrics={self.primary_metric: 0.0}, valid=True)
        positive = [v for v in values if v > 0]
        if not positive:
            return BenchmarkResult(valid=False)
        geomean = math.exp(sum(math.log(v) for v in positive) / len(positive))
        return BenchmarkResult(metrics={self.primary_metric: geomean}, valid=True)

    def is_better(self, a: BenchmarkResult, b: BenchmarkResult) -> bool:
        if not a.valid or self.primary_metric not in a.metrics:
            return False
        if not b.valid or self.primary_metric not in b.metrics:
            return False
        av = a.metrics[self.primary_metric]
        bv = b.metrics[self.primary_metric]
        return (av > bv) if self.higher_is_better else (av < bv)

    def speedup(self, current: BenchmarkResult, baseline: BenchmarkResult) -> float:
        cv = current.metrics.get(self.primary_metric, 0)
        bv = baseline.metrics.get(self.primary_metric, 0)
        if bv == 0 or cv == 0:
            return 0.0
        return cv / bv if self.higher_is_better else bv / cv
