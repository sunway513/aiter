# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
from torch import Generator, Tensor

from ..jit.core import compile_ops

# --- PyTorch fallbacks for CK-free builds ---


def _greedy_sample_fallback(out: Tensor, input: Tensor) -> None:
    out.copy_(input.argmax(dim=-1).to(out.dtype))


def _random_sample_outer_exponential_fallback(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperatures: Tensor,
    eps: float = 1e-10,
) -> None:
    logits = input.float()
    temp = temperatures.float().unsqueeze(-1).clamp(min=eps)
    probs = torch.softmax(logits / temp, dim=-1)
    modified = probs / (exponentials.float() + eps)
    out.copy_(modified.argmax(dim=-1).to(out.dtype))


def _random_sample_fallback(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None:
    logits = input.float()
    temp = temperatures.float().unsqueeze(-1).clamp(min=eps)
    probs = torch.softmax(logits / temp, dim=-1)
    exponentials = torch.empty_like(probs).exponential_(lambd, generator=generator)
    modified = probs / (exponentials + eps)
    out.copy_(modified.argmax(dim=-1).to(out.dtype))


def _mixed_sample_outer_exponential_fallback(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None:
    logits = input.float()
    temp = temperature.float().unsqueeze(-1)
    greedy_mask = temp.squeeze(-1) == 0
    greedy_tokens = logits.argmax(dim=-1)
    safe_temp = temp.clamp(min=eps)
    probs = torch.softmax(logits / safe_temp, dim=-1)
    modified = probs / (exponentials.float() + eps)
    random_tokens = modified.argmax(dim=-1)
    result = torch.where(greedy_mask, greedy_tokens, random_tokens)
    out.copy_(result.to(out.dtype))


def _mixed_sample_fallback(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None:
    logits = input.float()
    temp = temperature.float().unsqueeze(-1)
    greedy_mask = temp.squeeze(-1) == 0
    greedy_tokens = logits.argmax(dim=-1)
    safe_temp = temp.clamp(min=eps)
    probs = torch.softmax(logits / safe_temp, dim=-1)
    exponentials = torch.empty_like(probs).exponential_(lambd, generator=generator)
    modified = probs / (exponentials + eps)
    random_tokens = modified.argmax(dim=-1)
    result = torch.where(greedy_mask, greedy_tokens, random_tokens)
    out.copy_(result.to(out.dtype))


def _exponential_fallback(
    out: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None:
    out.exponential_(lambd, generator=generator)


# --- CK/HIP kernels with PyTorch fallbacks ---


@compile_ops("module_sample", fallback=_greedy_sample_fallback)
def greedy_sample(
    out: Tensor,
    input: Tensor,
) -> None: ...


@compile_ops("module_sample", fallback=_random_sample_outer_exponential_fallback)
def random_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperatures: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fallback=_random_sample_fallback)
def random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fallback=_mixed_sample_outer_exponential_fallback)
def mixed_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fallback=_mixed_sample_fallback)
def mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fallback=_exponential_fallback)
def exponential(
    out: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...
