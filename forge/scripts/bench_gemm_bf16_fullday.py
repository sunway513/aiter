#!/usr/bin/env python3
"""Round 13: FULL-DAY exhaustive BF16 HGEMM sweep.

Designed to run 3-6 hours per GPU. Covers everything our earlier 12 rounds
missed:
  - All tile_m ∈ {16, 32, 48, 64, 80, 96, 112, 128, 160, 256}
  - All tile_n ∈ {64, 96, 128, 160, 192, 256}
  - All tile_k ∈ {32, 64, 96, 128, 160, 256}
  - All warp layouts: (1,1), (1,2), (2,1), (1,4), (4,1), (2,2), (2,4), (4,2)
  - pack_n ∈ {1, 2}
  - b_to_lds ∈ {False, True}
  - Full hidden-knob grid: waves_per_eu × persistent_n_tiles × n_tile_repeat × b_to_lds_unroll × split_k
  - Both kernel_family ∈ {'hgemm', 'small_m'}

Shape-sharded: accept --shape argument, run the full grid on one shape.
Launch 5 copies on 5 GPUs for 5 shapes in parallel.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

TILE_M = (16, 32, 48, 64, 80, 96, 112, 128, 160, 256)
TILE_N = (64, 96, 128, 160, 192, 256)
TILE_K = (32, 64, 96, 128, 160, 256)
WARP_LAYOUTS = ((1,1), (1,2), (2,1), (1,4), (4,1), (2,2), (2,4), (4,2))  # last two may exceed 256 threads
PACK_N = (1, 2)
WAVES_PER_EU = (0, 1, 2, 3)
PERSISTENT_N = (1, 2)
N_TILE_REPEAT = (1, 2)
B_LDS_UNROLL = (0, 2, 4, 8)
SPLIT_K = (1, 2, 4, 8)


def bench(fn, w=3, n=15):
    torch.cuda.synchronize()
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def lds_bytes(tm, tn, tk, b_lds):
    a = max(2 * tm * tk * 2, tm * tn * 2)
    if not b_lds:
        return a
    return ((a + 15) & ~15) + 2 * tn * tk * 2


def feasible(tm, tn, tk, bmw, bnw, pack_n, b_lds, M):
    if bmw * bnw * 64 > 512:  # allow up to 8 warps (512 threads)
        return False
    if any(x % 16 for x in (tm, tn, tk)):
        return False
    if tn % (16 * pack_n):
        return False
    if tm % (bmw * 16) or tn % (bnw * 16):
        return False
    if M % tm or M % tn or M % tk:
        return False
    if lds_bytes(tm, tn, tk, b_lds) > 160 * 1024:
        return False
    return True


def hgemm_configs(M):
    out = []
    for tm, tn, tk in itertools.product(TILE_M, TILE_N, TILE_K):
        if tm == 16 and tk > 128:
            continue
        for (bmw, bnw), pack_n, b_lds in itertools.product(WARP_LAYOUTS, PACK_N, (False, True)):
            if not feasible(tm, tn, tk, bmw, bnw, pack_n, b_lds, M):
                continue
            base = dict(tile_m=tm, tile_n=tn, tile_k=tk, pack_n=pack_n,
                        block_m_warps=bmw, block_n_warps=bnw)
            if b_lds:
                base.update(b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False)
            else:
                base.update(auto_shuffle_b=True)
            # hidden-knob stacks (dedupe mutually exclusive combos)
            for wpe, pnt, ntr, blu, sk in itertools.product(
                WAVES_PER_EU, PERSISTENT_N, N_TILE_REPEAT, B_LDS_UNROLL, SPLIT_K):
                if pnt != 1 and ntr != 1:
                    continue
                if sk > 1 and M % (tk * sk):
                    continue
                cfg = dict(base)
                if wpe: cfg["waves_per_eu"] = wpe
                if pnt != 1: cfg["persistent_n_tiles"] = pnt
                if ntr != 1: cfg["n_tile_repeat"] = ntr
                if blu: cfg["b_to_lds_unroll"] = blu
                if sk != 1: cfg["split_k"] = sk
                out.append(cfg)
    return out


def small_m_configs(M):
    out = []
    for tn, tk in itertools.product(TILE_N, TILE_K):
        if tn % 16 or tk % 16:
            continue
        if M % tn or M % tk:
            continue
        for bnw in (1, 2, 4):
            if tn % (bnw * 16):
                continue
            for wpe, pnt, ntr, blu in itertools.product(
                WAVES_PER_EU, PERSISTENT_N, N_TILE_REPEAT, B_LDS_UNROLL):
                if pnt != 1 and ntr != 1:
                    continue
                cfg = dict(kernel_family="small_m",
                           tile_m=16, tile_n=tn, tile_k=tk,
                           block_m_warps=1, block_n_warps=bnw,
                           b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False)
                if wpe: cfg["waves_per_eu"] = wpe
                if pnt != 1: cfg["persistent_n_tiles"] = pnt
                if ntr != 1: cfg["n_tile_repeat"] = ntr
                if blu: cfg["b_to_lds_unroll"] = blu
                out.append(cfg)
    return out


def main():
    from aiter.ops.flydsl import flydsl_hgemm
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--families", default="hgemm,small_m",
                    help="comma list of kernel families")
    ap.add_argument("--time-budget-sec", type=int, default=4*3600,
                    help="wall-time budget; will stop past this")
    args = ap.parse_args()

    M = args.shape
    families = set(args.families.split(","))
    cfgs = []
    if "hgemm" in families:
        cfgs.extend(hgemm_configs(M))
    if "small_m" in families:
        cfgs.extend(small_m_configs(M))
    print(f"Enumerated {len(cfgs)} configs for M={M} (families={families})", flush=True)

    a = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)

    out = Path(args.output)
    t0 = time.time()
    best_tf = 0.0
    best_cfg = None
    ok = 0
    err = 0
    with out.open("w") as f:
        for i, cfg in enumerate(cfgs):
            if time.time() - t0 > args.time_budget_sec:
                print(f"  time budget exceeded, stopping at {i}/{len(cfgs)}", flush=True)
                break
            try:
                ms = bench(lambda: flydsl_hgemm(a, b, **cfg))
                tf = 2 * M * M * M / ms / 1e9
                ok += 1
                f.write(json.dumps({"M": M, "cfg": cfg, "tflops": tf, "ms": ms}) + "\n")
                f.flush()
                if tf > best_tf:
                    best_tf = tf
                    best_cfg = cfg
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:6.0f}s i={i}/{len(cfgs)}] NEW BEST {tf:7.2f}  {cfg}", flush=True)
            except Exception as exc:
                err += 1
                f.write(json.dumps({"M": M, "cfg": cfg, "error": repr(exc)[:80]}) + "\n")
                f.flush()
    print(f"  => best {best_tf:7.2f} ({ok} ok, {err} errs, {time.time()-t0:.0f}s)", flush=True)
    print(f"     cfg: {best_cfg}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
