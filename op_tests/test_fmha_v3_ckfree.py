# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test FMHA V3 ASM forward and backward correctness.
# Works in both full-CK and CK-free environments (exercises the V3 ASM path).

import pytest
import torch

import aiter
from aiter.test_mha_common import attention_ref, generate_qkv


def _v3_eligible(dtype, d, arch):
    """Check if shape is eligible for V3 ASM kernels."""
    if dtype != torch.bfloat16:
        return False
    if d not in (128, 192):
        return False
    if arch not in ("gfx942", "gfx950"):
        return False
    return True


def _get_arch():
    props = torch.cuda.get_device_properties(0)
    return props.gcnArchName.split(":")[0]


# V3-eligible shapes: bf16, d=128, various seqlens
@pytest.mark.parametrize("seqlen_q,seqlen_k", [(512, 512), (1024, 1024), (256, 512)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
def test_fmha_v3_fwd(seqlen_q, seqlen_k, causal, d):
    arch = _get_arch()
    dtype = torch.bfloat16
    if not _v3_eligible(dtype, d, arch):
        pytest.skip(f"V3 ASM not available on {arch}")

    batch_size = 2
    nheads = 8
    nheads_k = 8

    q, k, v = generate_qkv(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_k, d, d, dtype, "cuda"
    )

    # Reference
    out_ref, _, lse_ref = attention_ref(q, k, v, causal=causal)

    # V3 ASM
    out, lse, _ = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        return_lse=True,
        how_v3_bf16_cvt=2,
    )

    # Tolerances (bf16 matmul accumulation)
    atol = 5e-2
    rtol = 5e-2

    torch.testing.assert_close(out.float(), out_ref.float(), atol=atol, rtol=rtol)
    if lse is not None and lse_ref is not None:
        torch.testing.assert_close(lse.float(), lse_ref.float(), atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("seqlen_q,seqlen_k", [(512, 512), (256, 256)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [128])
def test_fmha_v3_bwd(seqlen_q, seqlen_k, causal, d):
    arch = _get_arch()
    dtype = torch.bfloat16
    if not _v3_eligible(dtype, d, arch):
        pytest.skip(f"V3 ASM not available on {arch}")

    batch_size = 2
    nheads = 8
    nheads_k = 8

    q, k, v = generate_qkv(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_k, d, d, dtype, "cuda"
    )
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    # Reference (with gradients)
    q_ref = q.detach().clone().float().requires_grad_(True)
    k_ref = k.detach().clone().float().requires_grad_(True)
    v_ref = v.detach().clone().float().requires_grad_(True)
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal)

    dout = torch.randn_like(out_ref, dtype=torch.float32)
    out_ref.backward(dout)

    # V3 ASM forward
    out, lse, _ = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=causal,
        return_lse=True,
        how_v3_bf16_cvt=2,
    )
    out.backward(dout.to(dtype))

    # Tolerances for backward (bf16 accumulation is less precise)
    atol = 1e-1
    rtol = 1e-1

    torch.testing.assert_close(out.float(), out_ref.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
