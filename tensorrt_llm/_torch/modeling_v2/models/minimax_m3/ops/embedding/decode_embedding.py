# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common embedding host implementation, bound to per-target state and callbacks."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Callable, Protocol, Tuple

import torch


class EmbeddingTarget(Protocol):
    HIDDEN: int
    MAX_ROWS: int
    _lock: AbstractContextManager[bool]
    _compiled: dict[tuple, Callable[..., None]]
    _compile: Callable[..., Callable[..., None]]
    _check: Callable[..., None]
    arg: Callable[[torch.Tensor], Any]
    m3_embed_norm: Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    _m3_embed_norm_jit: Any
    cuda_driver: Any
    cute: Any
    from_dlpack: Callable[..., Any]


def arg(state: EmbeddingTarget, /, tensor: torch.Tensor) -> Any:
    return state.from_dlpack(
        tensor.detach(),
        assumed_align=tensor.element_size()
        if tensor.ndim == 1 and tensor.dtype in (torch.int32, torch.int64)
        else 16,
    )


def _compile(
    state: EmbeddingTarget,
    /,
    ids: torch.Tensor,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    raw: torch.Tensor,
    out: torch.Tensor,
    use_pdl: bool,
) -> Callable[..., None]:
    key = (ids.shape[0], ids.dtype, tuple(table.shape), table.device, bool(use_pdl))
    fn = state._compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "m3_embed_norm compiles on its first call per shape: call it once before capture"
            )
        with state._lock:
            fn = state._compiled.get(key)
            if fn is None:
                stream = state.cuda_driver.CUstream(
                    torch.cuda.current_stream(table.device).cuda_stream
                )
                fn = state._compiled[key] = state.cute.compile(
                    state._m3_embed_norm_jit,
                    state.arg(ids),
                    state.arg(table),
                    state.arg(weight),
                    state.arg(raw),
                    state.arg(out),
                    int(table.shape[0]),
                    float(eps),
                    ids.shape[0],
                    state.HIDDEN,
                    bool(use_pdl),
                    stream,
                )
    return fn


def m3_embed_norm(
    state: EmbeddingTarget,
    /,
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
    state._check(ids, table, weight, raw, out)
    fn = state._compile(ids, table, weight, eps, raw, out, use_pdl)
    stream = state.cuda_driver.CUstream(torch.cuda.current_stream(ids.device).cuda_stream)
    fn(
        state.arg(ids),
        state.arg(table),
        state.arg(weight),
        state.arg(raw),
        state.arg(out),
        int(table.shape[0]),
        float(eps),
        stream,
    )
    return (out, raw)


def init_decode_embedding(
    self: Any,
    state: EmbeddingTarget,
    /,
    table: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    rows: int,
    id_dtype: torch.dtype,
    use_pdl: bool = True,
) -> None:
    assert not torch.cuda.is_current_stream_capturing()
    assert (
        table.dtype == torch.bfloat16 and table.is_contiguous() and (table.shape[1] == state.HIDDEN)
    )
    assert weight.dtype == torch.bfloat16 and weight.shape == (state.HIDDEN,)
    assert 1 <= rows <= state.MAX_ROWS and id_dtype in (torch.int32, torch.int64)
    self.table, self.weight, self.rows, self.id_dtype, self.use_pdl = (
        table,
        weight,
        rows,
        id_dtype,
        use_pdl,
    )
    self.eps = float(eps)
    self.raw = torch.empty(rows, state.HIDDEN, device=table.device, dtype=torch.bfloat16)
    self.output = torch.empty_like(self.raw)
    ids = torch.empty(rows, device=table.device, dtype=id_dtype)
    state._compile(ids, table, weight, self.eps, self.raw, self.output, use_pdl)


def call_decode_embedding(
    self: Any, state: EmbeddingTarget, /, ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert ids.shape == (self.rows,) and ids.dtype == self.id_dtype and ids.is_contiguous()
    return state.m3_embed_norm(
        ids, self.table, self.weight, self.eps, self.raw, self.output, self.use_pdl
    )


def embedding_firstnorm_reference(
    ids: torch.Tensor, table: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The original float64 embedding and first-layer RMSNorm reference math."""
    valid = (ids >= 0) & (ids < table.shape[0])
    rows = table[torch.where(valid, ids, torch.zeros_like(ids)).long()] * valid.unsqueeze(1).to(
        table.dtype
    )
    x = rows.double()
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return ((x * rstd * weight.double()).to(torch.bfloat16), rows)
