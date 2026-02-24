#!/usr/bin/env python3
"""
Tests for CK-free fallback functions in aiter/ops/cache.py and
RoPE auto-detection in aiter/rotary_embedding.py.

Run: HIP_VISIBLE_DEVICES=0 python3 op_tests/test_ck_free_fallbacks.py
"""

import sys
import random
import torch

DEVICE = "cuda"


def check_close(a, b, name, atol=1e-3, rtol=1e-3):
    """Check tensors are close, print result."""
    if a.dtype != b.dtype:
        a = a.float()
        b = b.float()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    max_diff = (a - b).abs().max().item() if a.numel() > 0 else 0.0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name} (max_diff={max_diff:.6f})")
    return ok


def test_swap_blocks_fallback():
    """Test _swap_blocks_fallback correctness."""
    print("\n=== test_swap_blocks_fallback ===")
    from aiter.ops.cache import _swap_blocks_fallback

    num_blocks = 8
    block_size = 4
    head_size = 16
    src = torch.randn(num_blocks, block_size, head_size, device=DEVICE)
    dst = torch.zeros_like(src)
    # Swap block 0->2, block 3->5
    block_mapping = torch.tensor([[0, 2], [3, 5]], dtype=torch.long, device=DEVICE)
    _swap_blocks_fallback(src, dst, block_mapping)

    ok1 = check_close(dst[2], src[0], "block 0->2")
    ok2 = check_close(dst[5], src[3], "block 3->5")
    # Untouched blocks should be zero
    ok3 = check_close(dst[0], torch.zeros_like(dst[0]), "block 0 untouched")
    return ok1 and ok2 and ok3


def test_copy_blocks_fallback():
    """Test _copy_blocks_fallback correctness."""
    print("\n=== test_copy_blocks_fallback ===")
    from aiter.ops.cache import _copy_blocks_fallback

    num_blocks = 8
    block_size = 4
    head_size = 16
    key_caches = torch.randn(num_blocks, block_size, head_size, device=DEVICE)
    value_caches = torch.randn(num_blocks, block_size, head_size, device=DEVICE)
    # block_mapping flat: [src0, dst0, src1, dst1]
    block_mapping = torch.tensor([1, 3, 5, 7], dtype=torch.long, device=DEVICE)

    key_orig = key_caches.clone()
    val_orig = value_caches.clone()
    _copy_blocks_fallback(key_caches, value_caches, block_mapping)

    ok1 = check_close(key_caches[3], key_orig[1], "key block 1->3")
    ok2 = check_close(key_caches[7], key_orig[5], "key block 5->7")
    ok3 = check_close(value_caches[3], val_orig[1], "val block 1->3")
    ok4 = check_close(value_caches[7], val_orig[5], "val block 5->7")
    return ok1 and ok2 and ok3 and ok4


def test_reshape_and_cache_fallback():
    """Test _reshape_and_cache_fallback vs reference."""
    print("\n=== test_reshape_and_cache_fallback ===")
    from aiter.ops.cache import _reshape_and_cache_fallback

    num_tokens = 16
    num_heads = 4
    head_size = 64
    block_size = 8
    num_blocks = 4
    x = 8  # head_size element grouping

    key = torch.randn(
        num_tokens, num_heads, head_size, device=DEVICE, dtype=torch.bfloat16
    )
    value = torch.randn(
        num_tokens, num_heads, head_size, device=DEVICE, dtype=torch.bfloat16
    )
    key_cache = torch.zeros(
        num_blocks,
        num_heads,
        head_size // x,
        block_size,
        x,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    value_cache = torch.zeros(
        num_blocks,
        num_heads,
        head_size,
        block_size,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=DEVICE)

    _reshape_and_cache_fallback(
        key, value, key_cache, value_cache, slot_mapping, "auto"
    )

    # Reference: manually scatter
    all_ok = True
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_off = slot % block_size
        # Check key
        k_reshaped = key[i].reshape(num_heads, head_size // x, x)
        ok = check_close(
            key_cache[block_idx, :, :, block_off, :],
            k_reshaped,
            f"key token {i}",
        )
        all_ok = all_ok and ok
        # Check value (non-asm layout: [num_blocks, num_heads, head_size, block_size])
        ok = check_close(
            value_cache[block_idx, :, :, block_off],
            value[i],
            f"val token {i}",
        )
        all_ok = all_ok and ok
        if not ok:
            break  # Don't flood output
    return all_ok


def test_concat_and_cache_mla_fallback():
    """Test _concat_and_cache_mla_fallback vs reference."""
    print("\n=== test_concat_and_cache_mla_fallback ===")
    from aiter.ops.cache import _concat_and_cache_mla_fallback

    num_tokens = 32
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    block_size = 1
    num_blocks = num_tokens
    entry_size = kv_lora_rank + qk_rope_head_dim

    kv_c = torch.randn(num_tokens, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16)
    k_pe = torch.randn(
        num_tokens, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        num_blocks, block_size, entry_size, device=DEVICE, dtype=torch.bfloat16
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=DEVICE)
    scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    _concat_and_cache_mla_fallback(kv_c, k_pe, kv_cache, slot_mapping, "auto", scale)

    # Reference
    ref_cache = torch.zeros_like(kv_cache)
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_off = slot % block_size
        ref_cache[block_idx, block_off, :kv_lora_rank] = kv_c[i]
        ref_cache[block_idx, block_off, kv_lora_rank:] = k_pe[i]

    ok = check_close(kv_cache, ref_cache, "concat_and_cache_mla 2D")

    # Also test 3D input (with num_kv_heads dim)
    num_kv_heads = 2
    kv_c_3d = torch.randn(
        num_tokens, num_kv_heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    k_pe_3d = torch.randn(
        num_tokens, num_kv_heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    kv_cache_3d = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        entry_size,
        device=DEVICE,
        dtype=torch.bfloat16,
    )

    _concat_and_cache_mla_fallback(
        kv_c_3d, k_pe_3d, kv_cache_3d, slot_mapping, "auto", scale
    )

    ref_cache_3d = torch.zeros_like(kv_cache_3d)
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_off = slot % block_size
        kv_concat = torch.cat([kv_c_3d[i], k_pe_3d[i]], dim=-1)
        ref_cache_3d[block_idx, block_off] = kv_concat

    ok2 = check_close(kv_cache_3d, ref_cache_3d, "concat_and_cache_mla 3D (multi-head)")
    return ok and ok2


def test_fused_qk_rope_concat_and_cache_mla_fallback():
    """Test _fused_qk_rope_concat_and_cache_mla_fallback (Triton delegation)."""
    print("\n=== test_fused_qk_rope_concat_and_cache_mla_fallback ===")
    from aiter.ops.cache import _fused_qk_rope_concat_and_cache_mla_fallback
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_cat_and_cache_mla

    num_tokens = 32
    num_heads = 8
    num_kv_heads = 1
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    block_size = 1
    num_blocks = num_tokens
    entry_size = kv_lora_rank + qk_rope_head_dim

    q_nope = torch.randn(
        num_tokens, num_heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    q_pe = torch.randn(
        num_tokens, num_heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    kv_c = torch.randn(
        num_tokens, num_kv_heads, kv_lora_rank, device=DEVICE, dtype=torch.bfloat16
    )
    k_pe = torch.randn(
        num_tokens, num_kv_heads, qk_rope_head_dim, device=DEVICE, dtype=torch.bfloat16
    )
    kv_cache = torch.zeros(
        num_blocks * block_size,
        num_kv_heads,
        entry_size,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    q_out = torch.empty(
        num_tokens, num_heads, entry_size, device=DEVICE, dtype=torch.bfloat16
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=DEVICE)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    q_scale = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)
    positions = torch.arange(num_tokens, dtype=torch.long, device=DEVICE)

    # cos/sin cache: [max_pos, rope_dim//2]
    rope_dim = qk_rope_head_dim
    max_pos = num_tokens + 10
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_cache = freqs.cos().to(DEVICE, dtype=torch.bfloat16)
    sin_cache = freqs.sin().to(DEVICE, dtype=torch.bfloat16)

    # Run fallback (delegates to Triton)
    _fused_qk_rope_concat_and_cache_mla_fallback(
        q_nope,
        q_pe,
        kv_c,
        k_pe,
        kv_cache,
        q_out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        is_neox=True,
        is_nope_first=True,
    )

    # Run Triton directly for comparison
    kv_cache_ref = torch.zeros_like(kv_cache)
    q_out_ref = torch.empty_like(q_out)
    fused_qk_rope_cat_and_cache_mla(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope=kv_c,
        k_pe=k_pe,
        kv_cache=kv_cache_ref,
        slot_mapping=slot_mapping,
        pos=positions,
        cos=cos_cache,
        sin=sin_cache,
        k_scale=k_scale,
        is_neox=True,
        q_out=q_out_ref,
    )

    ok1 = check_close(q_out, q_out_ref, "q_out matches Triton")
    ok2 = check_close(kv_cache, kv_cache_ref, "kv_cache matches Triton")
    return ok1 and ok2


def test_rope_auto_detection():
    """Test RotaryEmbedding.forward() auto-detects CK-free and falls back to Triton."""
    print("\n=== test_rope_auto_detection ===")
    from aiter.rotary_embedding import RotaryEmbedding

    head_size = 128
    rotary_dim = 128
    max_pos = 1024
    base = 10000
    dtype = torch.bfloat16

    rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base, True, dtype)
    rope = rope.to(DEVICE)

    num_tokens = 4
    positions = torch.arange(num_tokens, device=DEVICE)
    query = torch.randn(num_tokens, 8, head_size, device=DEVICE, dtype=dtype)
    key = torch.randn(num_tokens, 2, head_size, device=DEVICE, dtype=dtype)

    # Get Triton reference
    q_tri, k_tri = rope.forward_triton(positions, query.clone(), key.clone())

    # Get forward() result (should use HIP if available, but both should be correct)
    q_fwd, k_fwd = rope.forward(positions, query.clone(), key.clone())

    # Test: mock forward_hip to raise, forcing fallback to Triton
    original_hip = rope.forward_hip

    def broken_hip(*args, **kwargs):
        raise RuntimeError("Simulated CK-free build: HIP rope module not available")

    rope.forward_hip = broken_hip
    q_fallback, k_fallback = rope.forward(positions, query.clone(), key.clone())
    rope.forward_hip = original_hip  # restore

    # The fallback path should match the Triton path exactly
    ok1 = check_close(q_fallback, q_tri, "q fallback matches Triton")
    ok2 = check_close(k_fallback, k_tri, "k fallback matches Triton")

    # And forward() result should also be numerically close (HIP and Triton may differ slightly)
    ok3 = check_close(q_fwd, q_tri, "q forward() close to Triton", atol=0.05, rtol=0.05)
    ok4 = check_close(k_fwd, k_tri, "k forward() close to Triton", atol=0.05, rtol=0.05)
    return ok1 and ok2 and ok3 and ok4


def test_rope_query_only():
    """Test RoPE forward with key=None (query-only mode used by MLA)."""
    print("\n=== test_rope_query_only ===")
    from aiter.rotary_embedding import RotaryEmbedding

    head_size = 128
    rotary_dim = 64  # partial rotary
    max_pos = 1024
    base = 10000
    dtype = torch.bfloat16

    rope = RotaryEmbedding(head_size, rotary_dim, max_pos, base, True, dtype)
    rope = rope.to(DEVICE)

    num_tokens = 4
    positions = torch.arange(num_tokens, device=DEVICE)
    query = torch.randn(num_tokens, 8, head_size, device=DEVICE, dtype=dtype)

    # Get Triton reference (key=None)
    q_tri = rope.forward_triton(positions, query.clone())

    # Mock HIP failure to force Triton fallback
    original_hip = rope.forward_hip
    rope.forward_hip = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mock"))
    q_fallback = rope.forward(positions, query.clone())
    rope.forward_hip = original_hip

    ok = check_close(q_fallback, q_tri, "query-only fallback matches Triton")
    return ok


if __name__ == "__main__":
    results = []
    tests = [
        test_swap_blocks_fallback,
        test_copy_blocks_fallback,
        test_reshape_and_cache_fallback,
        test_concat_and_cache_mla_fallback,
        test_fused_qk_rope_concat_and_cache_mla_fallback,
        test_rope_auto_detection,
        test_rope_query_only,
    ]

    for test_fn in tests:
        try:
            ok = test_fn()
            results.append((test_fn.__name__, ok))
        except Exception as e:
            print(f"  [ERROR] {test_fn.__name__}: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_fn.__name__, False))

    print("\n" + "=" * 60)
    print("AITER CK-Free Fallback Test Summary")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\nAll {len(results)} tests PASSED")
    else:
        failed = sum(1 for _, ok in results if not ok)
        print(f"\n{failed}/{len(results)} tests FAILED")
        sys.exit(1)
