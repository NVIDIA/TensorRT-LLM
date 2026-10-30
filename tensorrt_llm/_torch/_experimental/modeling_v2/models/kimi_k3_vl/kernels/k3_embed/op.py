# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Torch ops of the decode-size embedding gather: ``trtllm::k3_embed`` (``table[ids]`` for up to ``MAX_TOKENS`` int32
or int64 ids and a bf16 table whose rows are whole 16-byte vectors, one vector per thread) and ``trtllm::k3_embed_norm``
(the same rows written into a caller's buffer, plus their RMSNorm, bit-identical to ``flashinfer.norm.rmsnorm``).
Compiled on the first call for its shape, which must happen outside CUDA-graph capture."""

from __future__ import annotations

import os
import threading
from typing import Dict

import torch

MAX_TOKENS = 64

_lock = threading.Lock()
_compiled: Dict[tuple, object] = {}


def _arg(t: torch.Tensor):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.dim() - 1)


def supports(ids: torch.Tensor, table: torch.Tensor) -> bool:
    """Whether ``k3_embed`` gathers ``table[ids]``."""
    return (
        ids.is_cuda
        and ids.dim() == 1
        and ids.dtype in (torch.int32, torch.int64)
        and 0 < ids.numel() <= MAX_TOKENS
        and table.dtype == torch.bfloat16
        and table.dim() == 2
        and table.is_contiguous()
        and table.shape[1] % 8 == 0
        and table.data_ptr() % 16 == 0
    )


@torch.library.custom_op("trtllm::k3_embed", mutates_args=())
def k3_embed(ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """``table[ids]`` for int32 / int64 ``ids`` [N <= 64] and a contiguous bf16 ``table`` [V, H] (H % 8 == 0); ids
    outside [0, V) give zero rows."""
    import cuda.bindings.driver as cuda_driver

    if not supports(ids, table):
        raise ValueError(
            f"k3_embed: unsupported call: ids {tuple(ids.shape)} {ids.dtype}, table {tuple(table.shape)} {table.dtype} "
            f"(int32 / int64 [N <= {MAX_TOKENS}], contiguous bf16 [V, H % 8 == 0])"
        )
    from . import k3_embed_kernel as kern

    n = ids.numel()
    vocab, hidden = table.shape
    out = torch.empty(n, hidden, dtype=torch.bfloat16, device=ids.device)
    args = (
        _arg(ids.contiguous()),
        _arg(table.view(-1).view(torch.int32)),
        _arg(out.view(-1).view(torch.int32)),
    )
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(ids.device).cuda_stream)
    key = (n, hidden // 8, ids.dtype, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_embed must run once per shape outside CUDA-graph capture first"
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.k3_embed, *args, int(vocab), n, hidden // 8, use_pdl, stream
                )
    fn(*args, int(vocab), stream)
    return out


@k3_embed.register_fake
def _(ids, table):
    return table.new_empty((ids.numel(), table.shape[1]))


def _plain_arg(t: torch.Tensor, align: int = 16):
    from cutlass.cute.runtime import from_dlpack

    return from_dlpack(t.detach(), assumed_align=align)


def norm_supports_hidden(hidden: int) -> bool:
    """Row widths ``k3_embed_norm`` reproduces flashinfer's RMSNorm for."""
    from . import k3_embed_kernel as kern

    return kern.norm_supports_hidden(hidden)


def supports_norm(
    ids: torch.Tensor, table: torch.Tensor, weight: torch.Tensor, raw: torch.Tensor
) -> bool:
    """Whether ``k3_embed_norm`` gathers ``table[ids]`` into ``raw`` and norms it with ``weight``."""
    hidden = table.shape[1] if table.dim() == 2 else 0
    return (
        supports(ids, table)
        and norm_supports_hidden(hidden)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (hidden,)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and raw.dtype == torch.bfloat16
        and tuple(raw.shape) == (ids.numel(), hidden)
        and raw.is_contiguous()
        and raw.data_ptr() % 16 == 0
    )


@torch.library.custom_op("trtllm::k3_embed_norm", mutates_args=("raw",))
def k3_embed_norm(
    ids: torch.Tensor, table: torch.Tensor, weight: torch.Tensor, eps: float, raw: torch.Tensor
) -> torch.Tensor:
    """``raw[:] = table[ids]`` (ids outside [0, V) give zero rows) and returns ``rmsnorm(raw, weight, eps)``, bf16
    [N, H]: bit-identical to ``k3_embed`` followed by ``flashinfer.norm.rmsnorm`` (see ``k3_embed_kernel``)."""
    import cuda.bindings.driver as cuda_driver

    if not supports_norm(ids, table, weight, raw):
        raise ValueError(
            f"k3_embed_norm: unsupported call: ids {tuple(ids.shape)} {ids.dtype}, table {tuple(table.shape)} "
            f"{table.dtype}, weight {tuple(weight.shape)} {weight.dtype}, raw {tuple(raw.shape)} {raw.dtype} "
            f"(int32 / int64 [N <= {MAX_TOKENS}], contiguous bf16 [V, H] with 6144 < H <= 16384 and H % 1024 == 0, "
            f"weight [H], raw [N, H])"
        )
    from . import k3_embed_kernel as kern

    n = ids.numel()
    vocab, hidden = table.shape
    out = torch.empty(n, hidden, dtype=torch.bfloat16, device=ids.device)
    ids = ids.contiguous()
    args = (_plain_arg(ids, ids.element_size()),) + tuple(
        _plain_arg(t) for t in (table, weight, raw, out)
    )
    use_pdl = os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"
    stream = cuda_driver.CUstream(torch.cuda.current_stream(ids.device).cuda_stream)
    key = ("norm", n, vocab, hidden, ids.dtype, use_pdl)
    fn = _compiled.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "trtllm::k3_embed_norm must run once per shape outside CUDA-graph capture first"
            )
        import cutlass.cute as cute

        with _lock:
            fn = _compiled.get(key)
            if fn is None:
                fn = _compiled[key] = cute.compile(
                    kern.k3_embed_norm, *args, int(vocab), float(eps), n, hidden, use_pdl, stream
                )
    fn(*args, int(vocab), float(eps), stream)
    return out


@k3_embed_norm.register_fake
def _(ids, table, weight, eps, raw):
    return table.new_empty((ids.numel(), table.shape[1]))
