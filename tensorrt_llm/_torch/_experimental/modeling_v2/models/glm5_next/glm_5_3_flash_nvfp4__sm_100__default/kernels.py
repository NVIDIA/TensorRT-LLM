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
"""Triton kernels of the GLM-5.3-Flash MTP layer.

* :func:`embed_rms_norm_columns` -- the MTP layer's token embedding, its ``enorm`` RMSNorm and the column slice this
  rank's row-parallel ``eh_proj`` consumes, in one launch.
* :func:`drafting_rows` -- each generation request's last accepted row in the verify layout (draft step 0's rows).
* :func:`mtp_commit` -- :func:`drafting_rows` and the KDA replay's accepted-draft record in one launch.
* :func:`verify_rows` -- the verify rows' positions, visible lengths, complete-pool counts and block-table rows.
* :func:`advance_after_step0` -- the lengths of draft step 1 (one row per request) in one launch.

Launch shapes depend only on host sizes, so they capture in CUDA graphs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _embed_rms_norm_columns_kernel(
    ids_ptr,
    table_ptr,
    table_stride,
    norm_weight_ptr,
    out_ptr,
    out_stride,
    vocab_size,
    hidden,
    eps,
    col_start,
    col_end,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.load(ids_ptr + row).to(tl.int64)
    valid = (token >= 0) & (token < vocab_size)
    cols = tl.arange(0, BLOCK)
    in_row = cols < hidden
    x = tl.load(table_ptr + token * table_stride + cols, mask=in_row & valid, other=0.0).to(
        tl.float32
    )
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / hidden + eps)
    w = tl.load(norm_weight_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
    y = x * rstd * w
    keep = (cols >= col_start) & (cols < col_end)
    tl.store(
        out_ptr + row * out_stride + (cols - col_start), y.to(out_ptr.dtype.element_ty), mask=keep
    )


def embed_rms_norm_columns(
    ids: torch.Tensor,
    table: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    col_start: int,
    col_end: int,
) -> torch.Tensor:
    """``rms_norm(embedding(ids), norm_weight, eps)[:, col_start:col_end]`` ([T, col_end - col_start], the table's
    dtype). Ids outside ``[0, vocab)`` (e.g. the -1 padding of rejected draft slots) embed as zeros, as the stock
    embedding's mask does, and normalize to zeros. The norm is over the full hidden size: ``x * rsqrt(mean(x^2) +
    eps) * w`` in fp32, the formula of the flashinfer RMSNorm it replaces."""
    assert ids.dim() == 1 and table.dim() == 2 and table.stride(-1) == 1
    vocab_size, hidden = table.shape
    assert norm_weight.shape == (hidden,) and 0 <= col_start < col_end <= hidden
    out = torch.empty((ids.shape[0], col_end - col_start), dtype=table.dtype, device=table.device)
    if ids.shape[0] == 0:
        return out
    _embed_rms_norm_columns_kernel[(ids.shape[0],)](
        ids,
        table,
        table.stride(0),
        norm_weight,
        out,
        out.stride(0),
        vocab_size,
        hidden,
        float(eps),
        col_start,
        col_end,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=4,
    )
    return out


@triton.jit
def _drafting_rows_kernel(
    batch_indices_ptr, num_accepted_ptr, out_ptr, n, width, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    m = i < n
    b = tl.load(batch_indices_ptr + i, mask=m, other=0).to(tl.int64)
    a = tl.load(num_accepted_ptr + i, mask=m, other=1).to(tl.int64)
    tl.store(out_ptr + i, b * width + a - 1, mask=m)


def drafting_rows(
    batch_indices: torch.Tensor, num_accepted: torch.Tensor, width: int
) -> torch.Tensor:
    """``batch_indices * width + num_accepted - 1`` (int64 [n]): the row of each generation request's last accepted
    token when every request verifies ``width`` rows."""
    n = num_accepted.shape[0]
    assert batch_indices.shape[0] == n
    out = torch.empty(n, dtype=torch.int64, device=num_accepted.device)
    if n:
        _drafting_rows_kernel[(1,)](
            batch_indices, num_accepted, out, n, width, BLOCK=max(16, triton.next_power_of_2(n))
        )
    return out


@triton.jit
def _mtp_commit_kernel(
    batch_indices_ptr,
    num_accepted_ptr,
    rows_ptr,
    state_indices_ptr,
    dummy_ptr,
    prev_ptr,
    prev_stride,
    n,
    width,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    m = i < n
    b = tl.load(batch_indices_ptr + i, mask=m, other=0).to(tl.int64)
    a = tl.load(num_accepted_ptr + i, mask=m, other=1).to(tl.int64)
    tl.store(rows_ptr + i, b * width + a - 1, mask=m)
    slot = tl.load(state_indices_ptr + i, mask=m, other=0).to(tl.int64)
    dummy = tl.load(dummy_ptr + i, mask=m, other=1) != 0
    accepted = tl.maximum(a - 1, 0).to(tl.int32)
    tl.store(prev_ptr + slot * prev_stride, accepted, mask=m & (dummy == 0))


def mtp_commit(
    batch_indices: torch.Tensor,
    num_accepted: torch.Tensor,
    width: int,
    state_indices: torch.Tensor,
    dummy_mask: torch.Tensor,
    prev_num_accepted: torch.Tensor,
) -> torch.Tensor:
    """:func:`drafting_rows` (returned) and the KDA replay record of each generation request in one launch: for
    every non-dummy request ``prev_num_accepted[state_indices[i]] = max(num_accepted[i] - 1, 0)`` (the accepted
    drafts the next verify replays; a dummy request's slot is left as is). ``prev_num_accepted`` may be strided
    (the cache manager's per-slot count is a column of its state pool)."""
    n = num_accepted.shape[0]
    assert batch_indices.shape[0] == n and state_indices.shape[0] == n and dummy_mask.shape[0] == n
    assert prev_num_accepted.dtype == torch.int32
    out = torch.empty(n, dtype=torch.int64, device=num_accepted.device)
    if n:
        _mtp_commit_kernel[(1,)](
            batch_indices,
            num_accepted,
            out,
            state_indices,
            dummy_mask,
            prev_num_accepted,
            prev_num_accepted.stride(0),
            n,
            width,
            BLOCK=max(16, triton.next_power_of_2(n)),
        )
    return out


@triton.jit
def _verify_rows_kernel(
    kv_lens_ptr,
    positions_ptr,
    visible_ptr,
    num_candidates_ptr,
    request_ids_ptr,
    n,
    first_table,
    T: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n * T
    b = i // T
    kv = tl.load(kv_lens_ptr + b, mask=mask, other=0).to(tl.int64)
    pos = kv - T + (i - b * T)
    tl.store(positions_ptr + i, pos, mask=mask)
    tl.store(visible_ptr + i, pos + 1, mask=mask)
    tl.store(num_candidates_ptr + i, ((pos + 1) // KPOOL).to(tl.int32), mask=mask)
    tl.store(request_ids_ptr + i, (first_table + b).to(tl.int32), mask=mask)


def verify_rows(
    kv_lens: torch.Tensor, tokens_per_request: int, kpool: int, first_table: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ``T = tokens_per_request`` verify rows of each request ``b`` (live length ``kv_lens[b]``, block-table row
    ``first_table + b``) in one launch: ``positions [B, T]`` int64 (``kv_lens[b] - T + t``), ``visible [B * T]``
    int64 (``positions + 1``), ``num_candidates [B * T]`` int32 (``visible // kpool``) and ``request_ids [B * T]``
    int32."""
    n = kv_lens.shape[0]
    device = kv_lens.device
    positions = torch.empty(n, tokens_per_request, dtype=torch.int64, device=device)
    visible = torch.empty(n * tokens_per_request, dtype=torch.int64, device=device)
    num_candidates = torch.empty(n * tokens_per_request, dtype=torch.int32, device=device)
    request_ids = torch.empty(n * tokens_per_request, dtype=torch.int32, device=device)
    if n:
        block = 256
        _verify_rows_kernel[(triton.cdiv(n * tokens_per_request, block),)](
            kv_lens,
            positions,
            visible,
            num_candidates,
            request_ids,
            n,
            first_table,
            T=tokens_per_request,
            KPOOL=kpool,
            BLOCK=block,
        )
    return positions, visible, num_candidates, request_ids


@triton.jit
def _advance_after_step0_kernel(
    seq_lens_ptr, kv_lens_ptr, num_accepted_ptr, n, num_contexts, draft_len, BLOCK: tl.constexpr
):
    b = tl.arange(0, BLOCK)
    mask = b < n
    kv = tl.load(kv_lens_ptr + b, mask=mask, other=0)
    accepted = tl.load(num_accepted_ptr + b, mask=mask & (b >= num_contexts), other=0)
    kv = tl.where(b < num_contexts, kv + 1, kv - (draft_len - accepted).to(kv.dtype))
    tl.store(kv_lens_ptr + b, kv, mask=mask)
    tl.store(seq_lens_ptr + b, tl.full([BLOCK], 1, seq_lens_ptr.dtype.element_ty), mask=mask)


def advance_after_step0(
    seq_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    num_accepted: torch.Tensor,
    num_contexts: int,
    draft_len: int,
) -> None:
    """In place, over the ``n = kv_lens.shape[0]`` requests: ``seq_lens[:n] = 1``; a context request's
    ``kv_lens`` grows by its first draft, a generation request's drops its rejected drafts
    (``kv_lens[b] -= draft_len - num_accepted[b]``)."""
    n = kv_lens.shape[0]
    if n:
        _advance_after_step0_kernel[(1,)](
            seq_lens,
            kv_lens,
            num_accepted,
            n,
            num_contexts,
            draft_len,
            BLOCK=max(16, triton.next_power_of_2(n)),
        )
