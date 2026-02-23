# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops

MD_NAME = "module_cache"


# ---- PyTorch-native fallbacks for CK-free builds ----


def _swap_blocks_fallback(src: Tensor, dst: Tensor, block_mapping: Tensor) -> None:
    for src_idx, dst_idx in block_mapping.cpu().tolist():
        dst[dst_idx].copy_(src[src_idx])


def _copy_blocks_fallback(
    key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor
) -> None:
    # block_mapping is a 1-D flat tensor: [src0, dst0, src1, dst1, ...]
    bm = block_mapping.cpu()
    num_pairs = bm.numel() // 2
    if isinstance(key_caches, (list, tuple)):
        for layer_idx in range(len(key_caches)):
            kc = key_caches[layer_idx]
            vc = value_caches[layer_idx]
            for i in range(num_pairs):
                src_idx = bm[2 * i].item()
                dst_idx = bm[2 * i + 1].item()
                kc[dst_idx].copy_(kc[src_idx])
                vc[dst_idx].copy_(vc[src_idx])
    else:
        for i in range(num_pairs):
            src_idx = bm[2 * i].item()
            dst_idx = bm[2 * i + 1].item()
            key_caches[dst_idx].copy_(key_caches[src_idx])
            value_caches[dst_idx].copy_(value_caches[src_idx])


def _reshape_and_cache_fallback(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    asm_layout: bool = False,
) -> None:
    # key/value:   [num_tokens, num_heads, head_size]
    # key_cache:   [num_blocks, num_heads, head_size/x, block_size, x]
    # value_cache: [num_blocks, num_heads, head_size, block_size]          (non-asm)
    #          or  [num_blocks, num_heads, block_size/x, head_size, x]    (asm)
    block_size = key_cache.shape[3]
    x = key_cache.shape[4]
    num_heads = key_cache.shape[1]
    head_size = key.shape[2]

    # Use clamp instead of boolean masking to keep fixed-size tensors
    # (compatible with CUDA graph capture). Invalid slots (< 0) write
    # harmlessly to block 0 / offset 0.
    slots = slot_mapping.clamp(min=0)
    block_indices = slots // block_size
    block_offsets = slots % block_size

    k = key
    v = value

    do_scale = kv_cache_dtype not in ("auto", "bf16") and k_scale is not None
    if do_scale:
        inv_k = 1.0 / k_scale.item() if k_scale.numel() == 1 else 1.0
        inv_v = (
            1.0 / v_scale.item()
            if v_scale is not None and v_scale.numel() == 1
            else 1.0
        )
        k = (k.float() * inv_k).to(key_cache.dtype)
        v = (v.float() * inv_v).to(value_cache.dtype)

    # key_cache: [num_blocks, num_heads, head_size/x, block_size, x]
    k_reshaped = k.reshape(-1, num_heads, head_size // x, x)
    key_cache[block_indices, :, :, block_offsets, :] = k_reshaped.to(key_cache.dtype)

    # Use actual tensor ndim — ATOM always allocates v_cache as 4D
    if value_cache.ndim == 5:
        # 5D asm layout: [num_blocks, num_heads, block_size/x, head_size, x]
        v_block_x_idx = block_offsets // x
        v_block_x_off = block_offsets % x
        n_valid = v.shape[0]
        for i in range(n_valid):
            value_cache[block_indices[i], :, v_block_x_idx[i], :, v_block_x_off[i]] = v[
                i
            ].to(value_cache.dtype)
    else:
        # 4D layout: [num_blocks, num_heads, head_size, block_size]
        value_cache[block_indices, :, :, block_offsets] = v.to(value_cache.dtype)


def _reshape_and_cache_flash_fallback(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None:
    # key_cache: [num_blocks, block_size, num_heads, head_size]
    block_size = key_cache.shape[1]
    slots = slot_mapping.clamp(min=0)
    block_indices = slots // block_size
    block_offsets = slots % block_size
    key_cache[block_indices, block_offsets] = key.to(key_cache.dtype)
    value_cache[block_indices, block_offsets] = value.to(value_cache.dtype)


def _reshape_and_cache_with_pertoken_quant_fallback(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_dequant_scales: Tensor,
    v_dequant_scales: Tensor,
    slot_mapping: Tensor,
    asm_layout: bool,
) -> None:
    # Per-token FP8 quantization + reshape_and_cache.
    # key/value:           [num_tokens, num_heads, head_size] in bf16/fp16
    # key_cache:           [num_blocks, num_heads, head_size/x, block_size, x]
    # value_cache non-asm: [num_blocks, num_heads, head_size, block_size]
    # value_cache asm:     [num_blocks, num_heads, block_size/x, head_size, x]
    # k/v_dequant_scales asm:     [num_blocks, num_heads, block_size]
    # k/v_dequant_scales non-asm: [num_heads, total_tokens]
    from aiter import pertoken_quant

    block_size = key_cache.shape[3]
    x = key_cache.shape[4]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    # Use clamp instead of boolean masking to keep fixed-size tensors
    # (compatible with CUDA graph capture). Invalid slots (< 0) write
    # harmlessly to block 0 / offset 0.
    slots = slot_mapping.clamp(min=0)
    block_indices = slots // block_size
    block_offsets = slots % block_size

    k = key
    v = value

    # pertoken_quant expects [batch, seq_len, num_heads, head_size]
    k_quant, k_scales = pertoken_quant(k.unsqueeze(1), quant_dtype=key_cache.dtype)
    k_quant = k_quant.squeeze(1)  # [num_tokens, num_heads, head_size]
    k_scales = k_scales.squeeze(1).squeeze(-1)  # [num_tokens, num_heads]

    v_quant, v_scales = pertoken_quant(v.unsqueeze(1), quant_dtype=value_cache.dtype)
    v_quant = v_quant.squeeze(1)
    v_scales = v_scales.squeeze(1).squeeze(-1)

    # Store quantized keys
    k_reshaped = k_quant.reshape(-1, num_heads, head_size // x, x)
    key_cache[block_indices, :, :, block_offsets, :] = k_reshaped

    # Store quantized values — use actual tensor ndim, not asm_layout flag,
    # because ATOM always allocates v_cache as 4D [blocks, heads, head_dim, block_size]
    if value_cache.ndim == 5:
        # 5D asm layout: [num_blocks, num_heads, block_size/x, head_size, x]
        v_block_x_idx = block_offsets // x
        v_block_x_off = block_offsets % x
        n_tokens = v_quant.shape[0]
        for i in range(n_tokens):
            value_cache[block_indices[i], :, v_block_x_idx[i], :, v_block_x_off[i]] = (
                v_quant[i]
            )
    else:
        # 4D layout: [num_blocks, num_heads, head_size, block_size]
        value_cache[block_indices, :, :, block_offsets] = v_quant

    # Store dequant scales
    if k_dequant_scales.ndim == 3:
        # [num_blocks, num_heads, block_size] — asm-style per-block scales
        k_dequant_scales[block_indices, :, block_offsets] = k_scales
        v_dequant_scales[block_indices, :, block_offsets] = v_scales
    else:
        # [num_heads, total_tokens] — flat per-token scales
        n_tokens = k_scales.shape[0]
        for i in range(n_tokens):
            k_dequant_scales[:, slots[i]] = k_scales[i]
            v_dequant_scales[:, slots[i]] = v_scales[i]


def _concat_and_cache_mla_fallback(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> None:
    # kv_c:       [num_tokens, kv_lora_rank] or [num_tokens, 1, kv_lora_rank]
    # k_pe:       [num_tokens, pe_dim] or [num_tokens, 1, pe_dim]
    # kv_cache:   [num_blocks, block_size, total_dim] or
    #             [num_blocks, block_size, num_kv_heads, total_dim]
    # slot_mapping: [num_tokens]
    # Flatten kv_c/k_pe to 2D if they have a head dim of 1
    if kv_c.ndim == 3:
        kv_c = kv_c.squeeze(1)
    if k_pe.ndim == 3:
        k_pe = k_pe.squeeze(1)
    # Concat along last dim: [num_tokens, kv_lora_rank + pe_dim]
    kv = torch.cat([kv_c, k_pe], dim=-1)
    # Scatter into kv_cache
    valid = slot_mapping >= 0
    slots = slot_mapping[valid]
    if kv_cache.ndim == 4:
        # [num_blocks, block_size, num_kv_heads, total_dim]
        block_size = kv_cache.shape[1]
        block_indices = slots // block_size
        block_offsets = slots % block_size
        data = kv[valid].to(kv_cache.dtype)
        if data.ndim == 2:
            # Single head squeezed: add head dim back
            data = data.unsqueeze(1)
        kv_cache[block_indices, block_offsets] = data
    else:
        # [num_blocks, block_size, total_dim]
        block_size = kv_cache.shape[1]
        block_indices = slots // block_size
        block_offsets = slots % block_size
        kv_cache[block_indices, block_offsets] = kv[valid].to(kv_cache.dtype)


def _fused_qk_rope_concat_and_cache_mla_fallback(
    q_nope: Tensor,
    q_pe: Tensor,
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    q_out: Tensor,
    slot_mapping: Tensor,
    k_scale: Tensor,
    q_scale: Tensor,
    positions: Tensor,
    cos_cache: Tensor,
    sin_cache: Tensor,
    is_neox: bool,
    is_nope_first: bool,
) -> None:
    # Delegate to the existing Triton fused kernel with parameter adaptation.
    from aiter.ops.triton.fusions.fused_kv_cache import (
        fused_qk_rope_cat_and_cache_mla,
    )

    # Squeeze cos/sin from 4D [seq, 1, 1, dim] to 2D [seq, dim] if needed
    cos = cos_cache
    sin = sin_cache
    if cos.ndim == 4:
        cos = cos.squeeze(1).squeeze(1)
    if sin.ndim == 4:
        sin = sin.squeeze(1).squeeze(1)

    # The Triton kernel returns (q_out, decode_q_pe_out, k_pe_out, q_nope_zeros_out)
    # but the CK interface is void (writes to q_out in-place and kv_cache in-place).
    fused_qk_rope_cat_and_cache_mla(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope=kv_c,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        pos=positions,
        cos=cos,
        sin=sin,
        k_scale=k_scale,
        is_neox=is_neox,
        q_out=q_out,
    )


@compile_ops("module_cache", fallback=_swap_blocks_fallback)
def swap_blocks(src: Tensor, dst: Tensor, block_mapping: Tensor) -> None: ...


@compile_ops("module_cache", fallback=_copy_blocks_fallback)
def copy_blocks(
    key_caches: Tensor, value_caches: Tensor, block_mapping: Tensor
) -> None: ...


@compile_ops("module_cache", fallback=_reshape_and_cache_fallback)
def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    asm_layout: bool = False,
) -> None: ...


@compile_ops("module_cache", fallback=_reshape_and_cache_flash_fallback)
def reshape_and_cache_flash(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    k_scale: Tensor,
    v_scale: Tensor,
) -> None: ...


@compile_ops("module_cache", fallback=_reshape_and_cache_with_pertoken_quant_fallback)
def reshape_and_cache_with_pertoken_quant(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_dequant_scales: Tensor,
    v_dequant_scales: Tensor,
    slot_mapping: Tensor,
    asm_layout: bool,
) -> None: ...


@compile_ops("module_cache")
def reshape_and_cache_with_block_quant(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    k_dequant_scales: Tensor,
    v_dequant_scales: Tensor,
    slot_mapping: Tensor,
    asm_layout: bool,
) -> None: ...


@compile_ops("module_cache")
def reshape_and_cache_with_block_quant_for_asm_pa(
    key: Tensor,  # [batch_size, seq_len, num_heads, head_size]
    value: Tensor,  # [batch_size, seq_len, num_heads, head_size]
    key_cache: Tensor,  # [num_blocks, num_heads, head_size/x, block_size:16, x]
    value_cache: Tensor,  # [num_blocks, num_heads, head_size, block_size:16] / [num_blocks, kvhead, block_size/x, head_size, x]
    k_dequant_scales: Tensor,  # [num_heads, num_blocks/(ori_block_size/block_size:16)]
    v_dequant_scales: Tensor,  # [num_heads, num_blocks/(ori_block_size/block_size:16)]
    slot_mapping: Tensor,
    asm_layout: bool,
    ori_block_size: int = 128,  # [128/256]
) -> None: ...


@compile_ops("module_cache", fallback=_concat_and_cache_mla_fallback)
def concat_and_cache_mla(
    kv_c: Tensor,
    k_pe: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype: str,
    scale: Tensor,
) -> None: ...


@compile_ops("module_cache")
def indexer_k_quant_and_cache(
    k: Tensor,
    kv_cache: Tensor,
    slot_mapping: Tensor,
    quant_block_size: int,
    scale_fmt: str,
) -> None: ...


@compile_ops("module_cache")
def cp_gather_indexer_k_quant_cache(
    kv_cache: Tensor,
    dst_k: Tensor,
    dst_scale: Tensor,
    block_table: Tensor,
    cu_seq_lens: Tensor,
) -> None: ...


@compile_ops("module_cache", fallback=_fused_qk_rope_concat_and_cache_mla_fallback)
def fused_qk_rope_concat_and_cache_mla(
    q_nope: Tensor,
    q_pe: Tensor,  # [num_tokens, num_heads, pe_dim]
    kv_c: Tensor,  # [num_tokens, kv_lora_rank] or [num_tokens, k_num_heads, kv_lora_rank]
    k_pe: Tensor,  # [num_tokens, pe_dim] or [num_tokens, k_num_heads, pe_dim]
    kv_cache: Tensor,  # [num_blocks, block_size, (kv_lora_rank + pe_dim)] or [num_blocks, block_size, k_num_heads, kv_lora_rank + pe_dim)]
    q_out: Tensor,  # [num_tokens, num_heads, qk_lora_rank+pe_dim]
    slot_mapping: Tensor,
    k_scale: Tensor,
    q_scale: Tensor,
    positions: Tensor,  # [num_tokens]
    cos_cache: Tensor,  # [max_position, rot_dim//2]
    sin_cache: Tensor,  # [max_position, rot_dim//2]
    is_neox: bool,
    is_nope_first: bool,
) -> None: ...
