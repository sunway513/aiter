# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_activation"


# --- PyTorch fallbacks for CK-free builds ---


def _silu_and_mul_fallback(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    out.copy_((F.silu(input[..., :d].float()) * input[..., d:].float()).to(out.dtype))


def _scaled_silu_and_mul_fallback(out: Tensor, input: Tensor, scale: Tensor) -> None:
    d = input.shape[-1] // 2
    result = F.silu(input[..., :d].float()) * input[..., d:].float()
    out.copy_((result * scale.float()).to(out.dtype))


def _gelu_and_mul_fallback(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    out.copy_((F.gelu(input[..., :d].float()) * input[..., d:].float()).to(out.dtype))


def _gelu_tanh_and_mul_fallback(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    out.copy_(
        (
            F.gelu(input[..., :d].float(), approximate="tanh") * input[..., d:].float()
        ).to(out.dtype)
    )


# --- CK/HIP kernels with PyTorch fallbacks ---


@compile_ops("module_activation", fallback=_silu_and_mul_fallback)
def silu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", fallback=_scaled_silu_and_mul_fallback)
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_activation", fallback=_gelu_and_mul_fallback)
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...


@compile_ops("module_activation", fallback=_gelu_tanh_and_mul_fallback)
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...
