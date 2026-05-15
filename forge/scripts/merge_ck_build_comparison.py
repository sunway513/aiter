#!/usr/bin/env python3
"""Merge ck_on.jsonl + ck_off.jsonl into bench_results.jsonl.

Produces paired rows (one CK-on + one CK-off record per family/shape)
and a joined view that makes Δe2e easy to compute. The merge is lossy
by design: it keeps the original per-mode records as-is and adds a
third record kind ``paired`` that cross-references them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def key(r):
    sh = r.get("shape") or {}
    return (r.get("family"), r.get("model"), r.get("label"),
            tuple(sorted(sh.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck-on",  required=True, type=Path)
    ap.add_argument("--ck-off", required=True, type=Path)
    ap.add_argument("--out",    required=True, type=Path)
    args = ap.parse_args()

    on = load(args.ck_on)
    off = load(args.ck_off)

    on_bench = {key(r): r for r in on if r.get("record_kind") == "bench"}
    off_bench = {key(r): r for r in off if r.get("record_kind") == "bench"}
    all_keys = sorted(set(on_bench) | set(off_bench))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        # Provenance first.
        for r in on:
            if r.get("record_kind") == "provenance":
                f.write(json.dumps(r) + "\n")
        for r in off:
            if r.get("record_kind") == "provenance":
                f.write(json.dumps(r) + "\n")
        # Paired rows.
        for k in all_keys:
            on_r  = on_bench.get(k, {})
            off_r = off_bench.get(k, {})
            family, model, label, shape_tuple = k
            shape = dict(shape_tuple)
            ck_on_tflops  = on_r.get("tflops")
            ck_off_tflops = off_r.get("tflops")
            e2e_gap_pct = None
            if (ck_on_tflops is not None and ck_off_tflops is not None
                    and ck_on_tflops > 0):
                e2e_gap_pct = round(
                    (ck_off_tflops - ck_on_tflops) / ck_on_tflops * 100, 1)
            status = off_r.get("status")
            if status == "OK" and e2e_gap_pct is not None:
                if abs(e2e_gap_pct) <= 10:
                    classification = "safe"
                elif -30 <= e2e_gap_pct < -10:
                    classification = "acceptable"
                else:
                    classification = ("broken" if e2e_gap_pct < -30
                                      else "safe_faster")
            elif status == "BROKEN":
                classification = "broken"
            else:
                classification = "unknown"

            paired = dict(
                record_kind="paired",
                family=family, model=model, label=label, shape=shape,
                ck_on_tflops=ck_on_tflops,
                ck_on_dispatcher=on_r.get("dispatcher_tag"),
                ck_on_status=on_r.get("status"),
                ck_off_tflops=ck_off_tflops,
                ck_off_dispatcher=off_r.get("dispatcher_tag"),
                ck_off_status=off_r.get("status"),
                ck_off_error=(off_r.get("error") or "")[:200],
                e2e_gap_pct=e2e_gap_pct,
                classification=classification,
            )
            f.write(json.dumps(paired) + "\n")
        # Also emit the raw rows for full traceability.
        for r in on + off:
            if r.get("record_kind") == "bench":
                f.write(json.dumps(r) + "\n")

    # Summary to stdout.
    classes = {"safe": 0, "safe_faster": 0, "acceptable": 0, "broken": 0,
               "unknown": 0}
    with args.out.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_kind") == "paired":
                classes[r["classification"]] = classes.get(r["classification"], 0) + 1
    print("Summary:")
    for k, v in classes.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
