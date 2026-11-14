# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm on the catalog's flashinfer kernels, and two calls overlapped on a second stream.

`RMSNorm` keeps the attribute names the decode kernels read (``weight``, ``variance_epsilon``). On bf16 and fp16 it
runs the kernels the built-in RMSNorm runs there, so it gives the same bits; on other dtypes it computes in fp32 the
way the built-in fallback does.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (  # noqa: E501
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

_KERNEL_DTYPES = (torch.float16, torch.bfloat16)


class RMSNorm(nn.Module):
    """``weight * x / sqrt(mean(x^2) + eps)``; with ``residual``, of ``x + residual``, returning the sum too.

    With ``residual`` on bf16 / fp16 the kernel updates ``x`` and ``residual`` in place, as the built-in one does.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hidden_states.dtype in _KERNEL_DTYPES:
            if residual is None:
                return flashinfer_rmsnorm(hidden_states, self.weight, self.variance_epsilon)
            flashinfer_fused_add_rmsnorm(
                hidden_states, residual, self.weight, self.variance_epsilon
            )
            return hidden_states, residual
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        if residual is not None:
            x = x + residual.to(torch.float32)
            residual = x.to(input_dtype)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        out = self.weight * x.to(input_dtype)
        return out if residual is None else (out, residual)


def run_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: Optional[torch.cuda.Stream],
) -> Tuple[Any, Any]:
    """``(fn0(), fn1())``, with ``fn1`` on ``aux_stream`` while a CUDA graph is captured, else one after the other.

    The engine turns the built-in overlap on only for CUDA-graph capture, where switching streams costs no host time
    at replay; this keeps that choice. A captured graph needs no ``record_stream``: every later use of ``aux_stream``
    first waits on the calling stream, as the next call's ``event0`` does.
    """
    if aux_stream is None or not torch.cuda.is_current_stream_capturing():
        return fn0(), fn1()
    event0.record()
    result0 = fn0()
    with torch.cuda.stream(aux_stream):
        event0.wait()
        result1 = fn1()
        event1.record()
    event1.wait()
    return result0, result1
