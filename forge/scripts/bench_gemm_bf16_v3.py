#!/usr/bin/env python3
"""Round 3: targeted b_to_lds path on round-1/round-2 winning tiles."""
import argparse
import json
import sys

import torch


def bench(fn, w=5, n=30):
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


def tf(M, N, K, ms):
    return 2 * M * N * K / ms / 1e9


def main():
    from aiter.ops.flydsl import flydsl_hgemm

    LDS = dict(b_to_lds=True, b_preshuffle=False, auto_shuffle_b=False)
    PRE = dict(auto_shuffle_b=True)

    configs = [
        # Best tile from round 1, try b_to_lds on them.
        (1024, dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, **LDS)),
        (1024, dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, async_copy=True, **LDS)),
        (2048, dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, **LDS)),
        (2048, dict(tile_m=128, tile_n=128, tile_k=128, block_m_warps=2, block_n_warps=2, async_copy=True, **LDS)),
        (4096, dict(tile_m=128, tile_n=256, tile_k=64, block_m_warps=1, block_n_warps=4, **LDS)),
        (4096, dict(tile_m=128, tile_n=256, tile_k=64, block_m_warps=1, block_n_warps=4, async_copy=True, **LDS)),
        (4096, dict(tile_m=128, tile_n=256, tile_k=128, block_m_warps=1, block_n_warps=4, **LDS)),
        (4096, dict(tile_m=128, tile_n=256, tile_k=128, block_m_warps=1, block_n_warps=4, async_copy=True, **LDS)),
        (8192, dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, **LDS)),
        (8192, dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, async_copy=True, **LDS)),
        (16384, dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, **LDS)),
        (16384, dict(tile_m=256, tile_n=128, tile_k=128, block_m_warps=4, block_n_warps=1, async_copy=True, **LDS)),
    ]

    results = []
    for M, cfg in configs:
        N = K = M
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        cfg_name = f"{cfg['tile_m']}x{cfg['tile_n']}x{cfg['tile_k']} w{cfg['block_m_warps']}x{cfg['block_n_warps']}"
        if cfg.get("async_copy"):
            cfg_name += " async"
        cfg_name += " b_lds"
        try:
            ms = bench(lambda: flydsl_hgemm(a, b, **cfg))
            t = tf(M, N, K, ms)
            print(f"M={M:5d}: {cfg_name:<45s} -> {t:7.1f} TFLOPS", flush=True)
            results.append({"M": M, "cfg": cfg, "tflops": t, "ms": ms})
        except Exception as exc:
            print(f"M={M:5d}: {cfg_name:<45s} ERROR: {repr(exc)[:60]}", flush=True)

    out = "/tmp/gemm_bf16_round3.jsonl"
    with open(out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
