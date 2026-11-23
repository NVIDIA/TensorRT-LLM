# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host side of the MiniMax-M3 decode embedding + first-layer RMSNorm (m3_embedding_firstnorm_kernel.py): H 6144,
1..8 rows, into caller-owned buffers.

Compiled on the first call per (rows, id dtype, table shape, PDL), which must happen outside CUDA-graph capture
(:class:`DecodeEmbedding` compiles at construction).
"""

import threading
from typing import Dict, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .m3_embedding_firstnorm_kernel import m3_embed_norm as _m3_embed_norm_jit

HIDDEN = 6144  # capacity: fixed: the checkpoint's hidden size
MAX_ROWS = 8  # capacity: fixed: the kernel's built rows (decode steps)

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def arg(tensor):
    return from_dlpack(
        tensor.detach(),
        assumed_align=tensor.element_size()
        if tensor.ndim == 1 and tensor.dtype in (torch.int32, torch.int64)
        else 16,
    )


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
    key = (ids.shape[0], ids.dtype, tuple(table.shape), table.device, bool(use_pdl))
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_embed_norm compiles on its first call per shape: call it once before capture"
            )
        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                stream = cuda_driver.CUstream(torch.cuda.current_stream(table.device).cuda_stream)
                fn = _compiled[key] = cute.compile(
                    _m3_embed_norm_jit,
                    arg(ids),
                    arg(table),
                    arg(weight),
                    arg(raw),
                    arg(out),
                    int(table.shape[0]),
                    float(eps),
                    ids.shape[0],
                    HIDDEN,
                    bool(use_pdl),
                    stream,
                )
    return fn


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
    _check(ids, table, weight, raw, out)
    fn = _compile(ids, table, weight, eps, raw, out, use_pdl)
    stream = cuda_driver.CUstream(torch.cuda.current_stream(ids.device).cuda_stream)
    fn(
        arg(ids),
        arg(table),
        arg(weight),
        arg(raw),
        arg(out),
        int(table.shape[0]),
        float(eps),
        stream,
    )
    return out, raw


class DecodeEmbedding:
    """Caller-owned buffers of one (rows, id dtype) decode embedding: ``raw`` (the gathered rows, the first residual)
    and ``output`` (their RMSNorm), over ``table`` and the first layer's plain (non-Gemma) input norm ``weight`` /
    ``eps``. Built (and compiled) before CUDA-graph capture."""

    def __init__(self, table, weight, eps, rows, id_dtype, use_pdl=True):
        assert not torch.cuda.is_current_stream_capturing()
        assert table.dtype == torch.bfloat16 and table.is_contiguous() and table.shape[1] == HIDDEN
        assert weight.dtype == torch.bfloat16 and weight.shape == (HIDDEN,)
        assert 1 <= rows <= MAX_ROWS and id_dtype in (torch.int32, torch.int64)
        self.table, self.weight, self.rows, self.id_dtype, self.use_pdl = (
            table,
            weight,
            rows,
            id_dtype,
            use_pdl,
        )
        self.eps = float(eps)
        self.raw = torch.empty(rows, HIDDEN, device=table.device, dtype=torch.bfloat16)
        self.output = torch.empty_like(self.raw)
        ids = torch.empty(rows, device=table.device, dtype=id_dtype)
        _compile(ids, table, weight, self.eps, self.raw, self.output, use_pdl)

    def __call__(self, ids):
        assert ids.shape == (self.rows,) and ids.dtype == self.id_dtype and ids.is_contiguous()
        return m3_embed_norm(
            ids, self.table, self.weight, self.eps, self.raw, self.output, self.use_pdl
        )
