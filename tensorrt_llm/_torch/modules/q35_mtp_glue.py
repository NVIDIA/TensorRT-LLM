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
"""Qwen3.5 MTP layer fc input in one launch.

Computes this rank's chunk of ``concat([gemma_rmsnorm(embed(ids)), gemma_rmsnorm(hidden)], -1)`` from a
replicated embedding table, with the stock math: ids outside ``[0, V)`` embed to zero, RMSNorm
``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32, rounded once to the activation dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _q35_mtp_input_kernel(
    ids_ptr,
    emb_ptr,
    hid_ptr,
    hid_stride,
    w_emb_ptr,
    w_hid_ptr,
    out_ptr,
    out_stride,
    vocab,
    eps_emb,
    eps_hid,
    chunk_lo,
    chunk_w,
    kv_ptr,
    kv_rows,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_KV: tl.constexpr,
):
    row = tl.program_id(0)
    half = tl.program_id(1)  # 0: embedding half [0, H), 1: hidden half [H, 2H) of the concat
    if HAS_KV:
        # The draft loop's deferred per-pass ``kv_lens_cuda[:rows] += 1``.
        if (half == 0) & (row < kv_rows):
            tl.store(kv_ptr + row, tl.load(kv_ptr + row) + 1)
    lo = tl.maximum(chunk_lo, half * H)
    hi = tl.minimum(chunk_lo + chunk_w, (half + 1) * H)
    if lo < hi:  # this half intersects the rank's fc input chunk
        offs = tl.arange(0, BLOCK_H)
        m = offs < H
        if half == 0:
            tok = tl.load(ids_ptr + row).to(tl.int64)
            valid = (tok >= 0) & (tok < vocab)
            tok = tl.where(valid, tok, 0)
            x = tl.load(emb_ptr + tok * H + offs, mask=m, other=0.0).to(tl.float32)
            x = tl.where(valid, x, 0.0)
            w = tl.load(w_emb_ptr + offs, mask=m, other=0.0).to(tl.float32)
            eps = eps_emb
        else:
            x = tl.load(hid_ptr + row.to(tl.int64) * hid_stride + offs, mask=m, other=0.0).to(
                tl.float32
            )
            w = tl.load(w_hid_ptr + offs, mask=m, other=0.0).to(tl.float32)
            eps = eps_hid
        var = tl.sum(x * x, axis=0) / H
        y = x * tl.rsqrt(var + eps) * (1.0 + w)
        col = offs + half * H  # column in the 2H-wide concat
        keep = m & (col >= lo) & (col < hi)
        tl.store(
            out_ptr + row.to(tl.int64) * out_stride + (col - chunk_lo),
            y.to(out_ptr.dtype.element_ty),
            mask=keep,
        )


def mtp_fused_input(
    input_ids: torch.Tensor,
    embed_weight: torch.Tensor,
    hidden_states: torch.Tensor,
    norm_emb_weight: torch.Tensor,
    norm_hid_weight: torch.Tensor,
    eps_emb: float,
    eps_hid: float,
    tp_size: int = 1,
    tp_rank: int = 0,
    kv_lens: torch.Tensor | None = None,
    kv_rows: int = 0,
) -> torch.Tensor:
    """``chunk(concat([gemma_rmsnorm(embed(ids)), gemma_rmsnorm(hidden)], -1), tp_size)[tp_rank]`` (contiguous
    ``[rows, 2H / tp_size]``) in one launch. ``embed_weight`` is the full ``[V, H]`` table. With ``kv_lens``, the
    same launch also applies ``kv_lens[:kv_rows] += 1`` (``kv_rows <= rows``)."""
    rows = input_ids.numel()
    h = hidden_states.shape[-1]
    assert embed_weight.shape[1] == h and hidden_states.shape[0] == rows, (
        embed_weight.shape,
        hidden_states.shape,
    )
    assert (2 * h) % tp_size == 0, (h, tp_size)
    assert hidden_states.stride(-1) == 1 and embed_weight.is_contiguous()
    chunk_w = 2 * h // tp_size
    out = torch.empty((rows, chunk_w), dtype=hidden_states.dtype, device=hidden_states.device)
    if rows == 0:
        return out
    block_h = triton.next_power_of_2(h)
    has_kv = kv_lens is not None and kv_rows > 0
    assert not has_kv or kv_rows <= rows, (kv_rows, rows)
    _q35_mtp_input_kernel[(rows, 2)](
        input_ids.reshape(-1),
        embed_weight,
        hidden_states,
        hidden_states.stride(0),
        norm_emb_weight,
        norm_hid_weight,
        out,
        out.stride(0),
        embed_weight.shape[0],
        float(eps_emb),
        float(eps_hid),
        tp_rank * chunk_w,
        chunk_w,
        kv_lens if has_kv else out,
        kv_rows if has_kv else 0,
        H=h,
        BLOCK_H=block_h,
        HAS_KV=has_kv,
        num_warps=8 if block_h >= 4096 else 4,
    )
    return out


__all__ = ["mtp_fused_input"]
