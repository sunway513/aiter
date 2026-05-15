#!/usr/bin/env python3
"""Parse rocprofv3 CSV output and summarize per-kernel counters."""
import csv
import sys
from collections import defaultdict


def summarize(csv_path, label):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    gemm_rows = [r for r in rows if ('gemm' in r['Kernel_Name'].lower() or
                                      'flydsl' in r['Kernel_Name'].lower() or
                                      'bf16gemm' in r['Kernel_Name'].lower())]
    print(f"\n=== {label} ({len(gemm_rows)} counter-rows) ===")
    if not gemm_rows:
        print("  kernel names in file:")
        for r in rows[:5]:
            print(f"   {r['Kernel_Name'][:100]}")
        return

    by_did = defaultdict(dict)
    for r in gemm_rows:
        did = r['Dispatch_Id']
        by_did[did]['kernel'] = r['Kernel_Name'][:60]
        by_did[did]['vgpr'] = r['VGPR_Count']
        by_did[did]['lds'] = r['LDS_Block_Size']
        by_did[did][r['Counter_Name']] = float(r['Counter_Value'])
        by_did[did]['dur_us'] = (int(r['End_Timestamp']) - int(r['Start_Timestamp'])) / 1000

    # Aggregate across all dispatches (all should be same GEMM)
    total = defaultdict(float)
    count = 0
    for did, d in by_did.items():
        count += 1
        for k, v in d.items():
            if isinstance(v, float):
                total[k] += v
    print(f"  n_dispatches = {count}")
    if count > 0:
        first = next(iter(by_did.values()))
        print(f"  kernel = {first['kernel']}")
        print(f"  VGPR = {first['vgpr']}, LDS = {first['lds']}")
        for k in sorted(total):
            avg = total[k] / count
            print(f"  avg {k} = {avg:,.1f}")


summarize(sys.argv[1], sys.argv[2])
if len(sys.argv) > 3:
    summarize(sys.argv[3], sys.argv[4])
