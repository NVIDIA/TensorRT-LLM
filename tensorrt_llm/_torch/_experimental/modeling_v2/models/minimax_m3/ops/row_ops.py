# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row ops of the M3.1 sampling glue as CUDA C++ torch ops (``cpp/tensorrt_llm/thop/minimaxM31/``), replacing the
targets' Triton row kernels:

  ``widen_rows``         ``mv2_minimax_m3::minimax_m31_widen_rows``: ``x.float()`` of 2-D bf16 / fp16 rows with a unit
  inner
                         stride, as a contiguous float32 tensor; exact, so bit for bit (DEP_LOGITS_WIDEN).
  ``fill_padding_rows``  ``mv2_minimax_m3::minimax_m31_fill_padding_rows``: the padding rows of contiguous 2-D float32 /
  bf16 /
                         fp16 rows set to a value (column 0 to another) in place, real rows untouched; bit for bit the
                         stock masked fills (DEP_PAD_FILL).
  ``widen_add_rows``     ``mv2_minimax_m3::minimax_m31_widen_add_rows``: ``base + bias.to(float32)`` for float32 rows
  with a
                         unit inner stride and a contiguous bf16 / float32 bias, as a contiguous float32 tensor; one
                         IEEE add, so bit for bit (DEP_MARKOV_GLUE's biased logits).

Each returns None (False for the in-place fill) when the build does not carry its op; the caller then runs its stock
torch line. Callers check their operands against the op's contract first. The ops launch with PDL.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

_OPS: dict = {}


def _op(name: str):
    """``torch.ops.trtllm.<name>`` when this build carries it, else None (looked up once)."""
    if name not in _OPS:
        _OPS[name] = getattr(family_ops(), name, None)
    return _OPS[name]


def widen_rows_op():
    return _op("minimax_m31_widen_rows")


def fill_padding_rows_op():
    return _op("minimax_m31_fill_padding_rows")


def widen_add_rows_op():
    return _op("minimax_m31_widen_add_rows")


def widen_rows(x: torch.Tensor) -> Optional[torch.Tensor]:
    """``x.float()`` bit for bit as a contiguous float32 tensor, or None when the op is not built."""
    op = widen_rows_op()
    return None if op is None else op(x)


def fill_padding_rows(
    rows: torch.Tensor, is_padding: torch.Tensor, value: float, col0_value: Optional[float] = None
) -> bool:
    """Fills the padding rows of ``rows`` in place; False (nothing written) when the op is not built."""
    op = fill_padding_rows_op()
    if op is None:
        return False
    op(rows, is_padding, value, col0_value)
    return True


def widen_add_rows(base: torch.Tensor, bias: torch.Tensor) -> Optional[torch.Tensor]:
    """``base + bias.to(float32)`` bit for bit as a contiguous float32 tensor, or None when the op is not built."""
    op = widen_add_rows_op()
    return None if op is None else op(base, bias)
