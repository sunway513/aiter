# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional

MD_NAME = "module_rmsnorm"


# --- Triton/PyTorch fallbacks for CK-free builds ---
# Lazy imports to avoid loading Triton at module import time.


def _rms_norm_cu_fallback(
    out: Tensor, input: Tensor, weight: Tensor, epsilon: float
) -> None:
    x = input.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
    out.copy_((x / rms * weight.float()).to(input.dtype))


def _fused_add_rms_norm_cu_fallback(
    input: Tensor, residual_in: Tensor, weight: Tensor, epsilon: float
) -> None:
    x = input.float() + residual_in.float()
    residual_in.copy_(x.to(residual_in.dtype))
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + epsilon)
    input.copy_((x / rms * weight.float()).to(input.dtype))


def _rms_norm_triton_fallback(
    input: Tensor, weight: Tensor, epsilon: float, use_model_sensitive_rmsnorm: int = 0
) -> Tensor:
    from aiter.ops.triton.normalization.rmsnorm import rms_norm as _triton_rms_norm

    return _triton_rms_norm(input, weight, epsilon)


def _rmsnorm2d_fwd_with_add_ck_fallback(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_add as _triton_fwd_with_add,
    )

    _triton_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)


def _rmsnorm2d_fwd_with_smoothquant_fallback(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_smoothquant as _triton_smoothquant,
    )

    _triton_smoothquant(out, input, xscale, yscale, weight, epsilon)


def _rmsnorm2d_fwd_with_add_smoothquant_fallback(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    out_before_quant: Optional[Tensor] = None,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_add_smoothquant as _triton_add_smoothquant,
    )

    _triton_add_smoothquant(
        out, input, residual_in, residual_out, xscale, yscale, weight, epsilon
    )


def _rmsnorm2d_fwd_with_dynamicquant_ck_fallback(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_dynamicquant as _triton_dynamicquant,
    )

    _triton_dynamicquant(out, input, yscale, weight, epsilon)


def _rmsnorm2d_fwd_with_add_dynamicquant_ck_fallback(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_add_dynamicquant as _triton_add_dynamicquant,
    )

    _triton_add_dynamicquant(
        out, input, residual_in, residual_out, yscale, weight, epsilon
    )


# module_rmsnorm_quant fallbacks — route to Triton implementations


def _rmsnorm_fallback(
    out: Tensor, input: Tensor, weight: Tensor, epsilon: float
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import rms_norm as _triton_rms_norm

    result = _triton_rms_norm(input, weight, epsilon)
    out.copy_(result)


def _add_rmsnorm_fallback(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_add as _triton_fwd_with_add,
    )

    _triton_fwd_with_add(out, input, residual_in, residual_out, weight, epsilon)


def _rmsnorm_quant_fallback(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_dynamicquant as _triton_dynamicquant,
    )

    _triton_dynamicquant(out, input, scale, weight, epsilon)


def _add_rmsnorm_quant_fallback(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    from aiter.ops.triton.normalization.rmsnorm import (
        rmsnorm2d_fwd_with_add_dynamicquant as _triton_add_dynamicquant,
    )

    _triton_add_dynamicquant(
        out, input, residual_in, residual_out, scale, weight, epsilon
    )


# --- CK/HIP kernels with Triton/PyTorch fallbacks ---


@compile_ops("module_rmsnorm", fallback=_rms_norm_cu_fallback)
def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Cuda version of rmsnorm
    """
    ...


@compile_ops("module_rmsnorm", fallback=_fused_add_rms_norm_cu_fallback)
def fused_add_rms_norm_cu(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
) -> None:
    """
    Cuda version of rmsnorm fused add
    """
    ...


def gen_rms_norm_fake_tensor(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    return torch.empty_like(input, dtype=input.dtype, device=input.device)


@compile_ops(
    "module_rmsnorm",
    fc_name="rmsnorm2d_fwd",
    gen_fake=gen_rms_norm_fake_tensor,
    fallback=_rms_norm_triton_fallback,
)
def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    """
    CK version of rmsnorm
    """
    ...


def rmsnorm2d_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        out = rmsnorm2d_fwd_ck(input, weight, epsilon, use_model_sensitive_rmsnorm)
    else:
        out = torch.empty_like(input, dtype=input.dtype, device=input.device)
        rmsnorm(out, input, weight, epsilon)
    return out


def rmsnorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        rmsnorm2d_fwd_with_add_ck(
            out,
            input,
            residual_in,
            residual_out,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
    else:
        add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon)


@compile_ops(
    "module_rmsnorm",
    fallback=_rmsnorm2d_fwd_with_smoothquant_fallback,
)
def rmsnorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops(
    "module_rmsnorm",
    fallback=_rmsnorm2d_fwd_with_add_smoothquant_fallback,
)
def rmsnorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    out_before_quant: Optional[Tensor] = None,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


def rmsnorm2d_fwd_with_dynamicquant(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        assert group_size == 0, "group_size is not supported for ck rmsnorm"
        assert not shuffle_scale, "shuffle_scale is not supported for ck rmsnorm"
        rmsnorm2d_fwd_with_dynamicquant_ck(
            out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm
        )
    else:
        rmsnorm_quant(out, input, yscale, weight, epsilon, group_size, shuffle_scale)


def rmsnorm2d_fwd_with_add_dynamicquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if use_model_sensitive_rmsnorm > 0 or input.shape[-1] > 8192:
        assert group_size == 0, "group_size is not supported for ck rmsnorm"
        assert not shuffle_scale, "shuffle_scale is not supported for ck rmsnorm"
        rmsnorm2d_fwd_with_add_dynamicquant_ck(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
    else:
        add_rmsnorm_quant(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            group_size,
            shuffle_scale,
        )


@compile_ops(
    "module_rmsnorm",
    gen_fake=gen_rms_norm_fake_tensor,
    fc_name="rmsnorm2d_fwd",
    fallback=_rms_norm_triton_fallback,
)
def rmsnorm2d_fwd_ck(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor: ...


@compile_ops(
    "module_rmsnorm",
    fc_name="rmsnorm2d_fwd_with_add",
    fallback=_rmsnorm2d_fwd_with_add_ck_fallback,
)
def rmsnorm2d_fwd_with_add_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops(
    "module_rmsnorm",
    fc_name="rmsnorm2d_fwd_with_dynamicquant",
    fallback=_rmsnorm2d_fwd_with_dynamicquant_ck_fallback,
)
def rmsnorm2d_fwd_with_dynamicquant_ck(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops(
    "module_rmsnorm",
    fc_name="rmsnorm2d_fwd_with_add_dynamicquant",
    fallback=_rmsnorm2d_fwd_with_add_dynamicquant_ck_fallback,
)
def rmsnorm2d_fwd_with_add_dynamicquant_ck(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None: ...


@compile_ops("module_rmsnorm_quant", fallback=_add_rmsnorm_quant_fallback)
def add_rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant", fallback=_add_rmsnorm_fallback)
def add_rmsnorm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None: ...


@compile_ops("module_rmsnorm_quant", fallback=_rmsnorm_quant_fallback)
def rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant", fallback=_rmsnorm_fallback)
def rmsnorm(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None: ...
