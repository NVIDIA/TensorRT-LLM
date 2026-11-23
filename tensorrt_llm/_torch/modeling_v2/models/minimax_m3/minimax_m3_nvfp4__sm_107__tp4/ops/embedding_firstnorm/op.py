# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the MiniMax-M3 decode embedding + first-layer RMSNorm (m3_embedding_firstnorm_kernel.py): H 6144,
1..8 rows, into caller-owned buffers.

Compiled on the first call per (rows, id dtype, table shape, PDL), which must happen outside CUDA-graph capture
(:class:`DecodeEmbedding` compiles at construction).
"""

import sys
import threading
from typing import Dict, Tuple, cast

import cuda.bindings.driver as cuda_driver  # noqa: F401 — used by the bound family implementation
import cutlass.cute as cute  # noqa: F401 — used by the bound family implementation
import torch
from cutlass.cute.runtime import from_dlpack as from_dlpack

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.embedding import (
    decode_embedding as _embedding_shared,
)

from ....kernels.embedding_firstnorm.m3_embedding_firstnorm_kernel import (  # noqa: F401
    m3_embed_norm as _m3_embed_norm_jit,
)

HIDDEN = 6144  # capacity: fixed: the checkpoint's hidden size
MAX_ROWS = 8  # capacity: fixed: the kernel's built rows (decode steps)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def arg(tensor):
    return _embedding_shared.arg(_EMBED_TARGET, tensor)


def _check(ids, table, weight, raw, out) -> None:
    rows = ids.shape[0] if ids.dim() == 1 else 0
    if not (
        1 <= rows <= MAX_ROWS
        and ids.dtype in (torch.int32, torch.int64)
        and ids.is_contiguous()
        and table.dtype == torch.bfloat16
        and table.dim() == 2
        and table.shape[1] == HIDDEN
        and table.is_contiguous()
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (HIDDEN,)
        and all(
            t.dtype == torch.bfloat16 and tuple(t.shape) == (rows, HIDDEN) and t.is_contiguous()
            for t in (raw, out)
        )
    ):
        raise ValueError(
            "m3_embed_norm: expects ids int32/int64 [1..8], a contiguous bf16 table [V, 6144], bf16 weight [6144] "
            "and contiguous bf16 raw / out [rows, 6144]"
        )


def _compile(ids, table, weight, eps, raw, out, use_pdl: bool):
    return _embedding_shared._compile(_EMBED_TARGET, ids, table, weight, eps, raw, out, use_pdl)


def m3_embed_norm(
    ids: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    raw: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Write ``table[ids]`` into ``raw`` (a zero row for an id outside ``[0, V)``) and its RMSNorm with ``weight`` /
    ``eps`` into ``out``; returns ``(out, raw)``."""
    return _embedding_shared.m3_embed_norm(
        _EMBED_TARGET, ids, table, weight, eps, raw, out, use_pdl
    )


class DecodeEmbedding:
    """Caller-owned buffers of one (rows, id dtype) decode embedding: ``raw`` (the gathered rows, the first residual)
    and ``output`` (their RMSNorm), over ``table`` and the first layer's plain (non-Gemma) input norm ``weight`` /
    ``eps``. Built (and compiled) before CUDA-graph capture."""

    def __init__(self, table, weight, eps, rows, id_dtype, use_pdl=True):
        _embedding_shared.init_decode_embedding(
            self, _EMBED_TARGET, table, weight, eps, rows, id_dtype, use_pdl
        )

    def __call__(self, ids):
        return _embedding_shared.call_decode_embedding(self, _EMBED_TARGET, ids)


_EMBED_TARGET = cast(_embedding_shared.EmbeddingTarget, sys.modules[__name__])
