#!/usr/bin/env python3
"""Round 14: asm variants with correct constraints (bpreshuffle/splitK)."""
import itertools
import json
import sys
import time

import torch

# Map: kernel_name -> (requires_bpreshuffle, requires_splitK>1)
ASM_KERNELS = {
    "bf16gemm_bf16_tn_256x256": (False, False),
    "bf16gemm_bf16_tn_256x256_bpreshuffle": (True, False),
    # All _bshuffle_splitk / _pf3_splitk / _splitk_clean: bpreshuffle + splitK>1
    "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_32x64_pf3_splitk": (False, True),
    "bf16gemm_fp32bf16_tn_32x64_splitk_clean": (False, True),
    "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_48x64_pf3_splitk": (False, True),
    "bf16gemm_fp32bf16_tn_48x64_splitk_clean": (False, True),
    "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_64x64_pf3_splitk": (False, True),
    "bf16gemm_fp32bf16_tn_64x64_splitk_clean": (False, True),
    "bf16gemm_fp32bf16_tn_80x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_80x64_splitk_clean": (False, True),
    "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_96x64_pf3_splitk": (False, True),
    "bf16gemm_fp32bf16_tn_96x64_splitk_clean": (False, True),
    "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_clean": (True, True),
    "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk": (True, True),
    "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk_clean": (True, True),
}


def bench(fn, w=5, n=25):
    torch.cuda.synchronize()
    for _ in range(w):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    from aiter.ops.gemm_op_a16w16 import gemm_a16w16_asm
    from aiter.ops.shuffle import shuffle_weight

    shapes = [1024, 2048, 4096, 8192, 16384]
    with open("/tmp/gemm_asm_full.jsonl", "w") as f:
        for M in shapes:
            a = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
            b = torch.randn(M, M, device="cuda", dtype=torch.bfloat16)
            b_shuf = shuffle_weight(b.contiguous(), layout=(16, 16))
            out = torch.empty(M, M, device="cuda", dtype=torch.bfloat16)
            print(f"\n=== M={M} ===", flush=True)
            best_tf = 0.0; best_spec = None
            for kname, (needs_bpre, needs_splitk) in ASM_KERNELS.items():
                splitKs = [2, 4, 8, 16] if needs_splitk else [1]
                for sk in splitKs:
                    if M % sk:
                        continue
                    B = b_shuf if needs_bpre else b
                    cfg = dict(kernelName=kname, splitK=sk, bpreshuffle=needs_bpre)
                    try:
                        ms = bench(lambda: gemm_a16w16_asm(a, B, out, **cfg))
                        tf = 2 * M * M * M / ms / 1e9
                        rec = {"M": M, **cfg, "tflops": tf, "ms": ms}
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        if tf > best_tf:
                            best_tf = tf; best_spec = cfg
                            print(f"  NEW BEST {tf:7.2f}  {kname} splitK={sk}", flush=True)
                    except Exception as exc:
                        rec = {"M": M, **cfg, "error": repr(exc)[:80]}
                        f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"  => best {best_tf:7.2f}  {best_spec}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
