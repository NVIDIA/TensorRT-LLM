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
"""References for attention over a windowed causal K/V cache.

``dense_window_reference`` knows nothing about pages: it keeps every token of the rollout
in plain tensors and selects the visible ones by position. ``paged_window_reference`` has
the signature and contract of ``cute_dsl_paged_window_fmha_fwd`` and reads the cache's pool
through the page table, so it checks what a paged kernel is handed.
"""

import torch
import torch.nn.functional as F


def visible_positions(start: int, end: int, fixed: int, window: int) -> torch.Tensor:
    """Logical key positions of a causal block ``[start, end)``.

    The fixed region, and everything from ``window`` before the block to its end.
    """
    first = max(0, start - window)
    pos = torch.arange(end)
    return pos[(pos < fixed) | (pos >= first)]


def _attend(q: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, sm_scale: float):
    """fp32 attention of ``q`` ``[T, H_q, D]`` over ``keys``/``values`` ``[S, H_kv, D]``."""
    rep = q.shape[1] // keys.shape[1]
    kx = keys.repeat_interleave(rep, dim=1)
    vx = values.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).float().unsqueeze(0),
        kx.transpose(0, 1).float().unsqueeze(0),
        vx.transpose(0, 1).float().unsqueeze(0),
        scale=sm_scale,
    )
    return out.squeeze(0).transpose(0, 1).to(q.dtype)


def dense_window_reference(
    q: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    start: int,
    fixed: int,
    window: int,
    block_size: int,
    sm_scale: float,
) -> torch.Tensor:
    """Attention of the ``T`` tokens of ``q`` at logical positions ``[start, start + T)``.

    ``keys``/``values`` hold every token of the rollout up to ``start + T``, indexed by
    logical position (the staged tokens at their positions). Block ``j`` of ``block_size``
    tokens sees the fixed region, the ``window`` tokens before its first token and
    everything up to its own end.
    """
    outs = []
    for lo in range(0, q.shape[0], block_size):
        s, e = start + lo, start + lo + block_size
        pos = visible_positions(s, e, fixed, window).to(q.device)
        outs.append(_attend(q[lo : lo + block_size], keys[pos], values[pos], sm_scale))
    return torch.cat(outs)


def paged_window_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    page_table: torch.Tensor,
    cum_seqlen_k: torch.Tensor,
    kv_len: torch.Tensor,
    out: torch.Tensor,
    *,
    prefix_tokens: int,
    block_size: int,
    sm_scale: float,
) -> None:
    """Reference of ``cute_dsl_paged_window_fmha_fwd``.

    Gathers each sequence's keys, ``[0, prefix_tokens)`` and ``[start, start + length)``, from
    the pool through ``page_table`` and attends in fp32.
    """
    tokens_per_page = kv.shape[3]
    table = page_table.tolist()
    starts, lengths = cum_seqlen_k.tolist(), kv_len.tolist()
    for j, lo in enumerate(range(0, q.shape[0], block_size)):
        pos = sorted(set(range(prefix_tokens)) | set(range(starts[j], starts[j] + lengths[j])))
        pages = torch.tensor([table[p // tokens_per_page] for p in pos], device=kv.device)
        slots = torch.tensor([p % tokens_per_page for p in pos], device=kv.device)
        # [S, H_kv, D] for each plane.
        keys = kv[pages, 0, :, slots]
        values = kv[pages, 1, :, slots]
        out[lo : lo + block_size] = _attend(q[lo : lo + block_size], keys, values, sm_scale)
