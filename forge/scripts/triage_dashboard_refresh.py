#!/usr/bin/env python3
"""Triage a refresh JSON into stale / genuine-low / confirmed-ok buckets.

Usage: python3 triage_dashboard_refresh.py results/refresh_<model>.json
"""
import json
import statistics
import sys


def main():
    if len(sys.argv) < 2:
        print("usage: triage_dashboard_refresh.py <refresh.json>")
        return 1
    d = json.load(open(sys.argv[1]))
    rows = d['rows']

    stale = []         # refreshed >= 1.5x dashboard — dashboard must update
    genuine_low = []   # refreshed ratio < 0.7x B300 — aiter-forge tuning target
    regressed = []     # refreshed < 0.8x dashboard — must investigate
    ok = []            # everything else

    for r in rows:
        sp = r.get('speedup_over_dashboard')
        ratio = r.get('new_ratio_vs_b300')
        if sp and sp < 0.8:
            regressed.append(r)
        elif sp and sp >= 1.5:
            stale.append(r)
        elif ratio is not None and ratio < 0.7:
            genuine_low.append(r)
        else:
            ok.append(r)

    print(f"\n=== Triage for {sys.argv[1]} ===")
    print(f"  Total refreshed: {len(rows)}")
    print(f"  STALE       (dashboard behind, speedup>=1.5x):   {len(stale):>4}")
    print(f"  GENUINE-LOW (ratio<0.7x B300 after refresh):     {len(genuine_low):>4}")
    print(f"  REGRESSED   (refreshed<0.8x dash, investigate):  {len(regressed):>4}")
    print(f"  OK          (already in spec):                   {len(ok):>4}")

    def show(label, rs, n=10, key=lambda r: -r.get('speedup_over_dashboard', 0)):
        rs = sorted(rs, key=key)[:n]
        if not rs: return
        print(f"\n=== TOP {label} ({len(rs)} shown) ===")
        for r in rs:
            sp = r.get('speedup_over_dashboard') or 0
            nr = r.get('new_ratio_vs_b300') or 0
            print(f"  {r['op']:<14} {r.get('label','') or '':<16} M={r.get('M',0):>4} N={r.get('N') or 0:>5} K={r.get('K') or 0:>5}  dash={r.get('dashboard_mi355x_tflops',0):>8.2f}  new={r.get('refreshed_mi355x_tflops',0):>8.2f}  speedup={sp:>5.2f}x  ratio={nr:.3f}")

    show("stale", stale, n=10)
    show("regressed", regressed, n=10, key=lambda r: r.get('speedup_over_dashboard', 1))
    show("genuine-low", genuine_low, n=10, key=lambda r: r.get('new_ratio_vs_b300', 1))


if __name__ == "__main__":
    sys.exit(main() or 0)
