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
"""Strict linear-draft acceptance (+ synthetic forced acceptance) and the first draft step's bookkeeping, one
Triton launch each.

``fused_strict_accept`` replaces the kernels of ``SpecWorkerBase._sample_and_accept_draft_tokens_base`` after the
target sampling (including ``_apply_force_accepted_tokens``). Outputs are bit-identical to the unfused path:

* ``accepted_tokens[b, j]`` = target token ``j`` of generation request ``b`` (context requests: column 0
  only, the other columns are left untouched);
* ``num_accepted_tokens[b]`` = 1 + number of leading draft tokens equal to the target tokens;
* with a forced acceptance value ``f`` (``TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS``) the generation
  rows take ``min(int(f) + 1 + (pool[(counter' * C + slot * S) & (P - 1)] < frac(f)), k + 1)`` with
  ``counter' = counter + 1`` stored back (fixed-seed pool, device counter: every TP rank draws the same).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

MAX_DRAFT_LEN = 31  # columns of the accept tile (k + 1 <= 32)


@triton.jit
def _strict_accept_kernel(
    target_ptr,  # [num_contexts + num_gens * (K + 1)] sampled target tokens (any int dtype)
    draft_ptr,  # [num_gens, K] draft tokens of the previous step
    accepted_ptr,  # [batch, accepted_stride] int32 out
    num_accepted_ptr,  # [batch] int32 out
    pool_ptr,  # [POOL] float32 forced-acceptance pool
    counter_ptr,  # [1] int64 forced-acceptance counter (bumped in place)
    num_contexts,
    num_gens,
    accepted_stride,
    frac,  # fp32: probability of one more forced token
    K: tl.constexpr,
    FORCE: tl.constexpr,  # 0: natural acceptance, 1: forced base only, 2: forced base + random extra
    BASE_TOTAL: tl.constexpr,
    MAX_TOTAL: tl.constexpr,
    COUNTER_STRIDE: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    POOL_MASK: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ---- context requests: column 0 = their sampled token, one accepted token ----
    ctx = tl.arange(0, BLOCK_B)
    for c0 in range(0, num_contexts, BLOCK_B):
        rows = c0 + ctx
        ok = rows < num_contexts
        tok = tl.load(target_ptr + rows, mask=ok, other=0)
        tl.store(accepted_ptr + rows.to(tl.int64) * accepted_stride, tok.to(tl.int32), mask=ok)
        tl.store(num_accepted_ptr + rows, tl.full([BLOCK_B], 1, tl.int32), mask=ok)

    # ---- generation requests ----
    cols = tl.arange(0, BLOCK_K)
    col_ok = cols < (K + 1)
    draft_col_ok = cols < K
    if FORCE == 2:
        counter = tl.load(counter_ptr) + 1
    for g0 in range(0, num_gens, BLOCK_B):
        g = g0 + tl.arange(0, BLOCK_B)
        g_ok = g < num_gens
        tgt = tl.load(
            target_ptr + num_contexts + g[:, None] * (K + 1) + cols[None, :],
            mask=g_ok[:, None] & col_ok[None, :],
            other=0,
        )
        rows = (num_contexts + g).to(tl.int64)
        tl.store(
            accepted_ptr + rows[:, None] * accepted_stride + cols[None, :],
            tgt.to(tl.int32),
            mask=g_ok[:, None] & col_ok[None, :],
        )
        if FORCE == 0:
            drf = tl.load(
                draft_ptr + g[:, None] * K + cols[None, :],
                mask=g_ok[:, None] & draft_col_ok[None, :],
                other=0,
            )
            match = (drf.to(tl.int64) == tgt.to(tl.int64)) & draft_col_ok[None, :]
            # leading matches = index of the first mismatch among the K draft columns
            first_miss = tl.min(tl.where(match, BLOCK_K, cols[None, :]), axis=1)
            n_acc = 1 + tl.minimum(first_miss, K)
        elif FORCE == 1:
            n_acc = tl.full([BLOCK_B], BASE_TOTAL, tl.int32)
        else:
            idx = (counter * COUNTER_STRIDE + g.to(tl.int64) * SLOT_STRIDE) & POOL_MASK
            rnd = tl.load(pool_ptr + idx, mask=g_ok, other=1.0)
            extra = (rnd < frac).to(tl.int32)
            n_acc = tl.minimum(BASE_TOTAL + extra, MAX_TOTAL)
        tl.store(num_accepted_ptr + rows, n_acc.to(tl.int32), mask=g_ok)
    if FORCE == 2:
        tl.store(counter_ptr, counter)


def fused_strict_accept(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    force: Optional[tuple] = None,
) -> tuple:
    """``(accepted_tokens [batch, K + 1] int32, num_accepted_tokens [batch] int32)``.

    ``draft_tokens``: [num_gens, K]. ``force``: None (natural acceptance) or
    ``(base_total, max_total, frac, pool, counter, counter_stride, slot_stride, pool_size)`` of the
    forced acceptance (``frac == 0`` or ``base_total == max_total`` = deterministic base only)."""
    k = draft_tokens.shape[-1]
    num_gens = batch_size - num_contexts
    assert k + 1 <= MAX_DRAFT_LEN + 1, k
    device = target_tokens.device
    accepted = torch.empty((batch_size, k + 1), dtype=torch.int32, device=device)
    num_accepted = torch.empty((batch_size,), dtype=torch.int32, device=device)
    if force is None:
        mode, base_total, max_total, frac = 0, 1, k + 1, 0.0
        pool = counter = num_accepted  # unused
        c_stride = s_stride = pool_mask = 1
    else:
        base_total, max_total, frac, pool, counter, c_stride, s_stride, pool_size = force
        mode = 2 if (frac > 0.0 and base_total < max_total) else 1
        if pool is None:
            pool = counter = num_accepted  # unused in mode 1
        pool_mask = pool_size - 1
    draft = draft_tokens.contiguous() if num_gens > 0 else num_accepted
    block_b = max(16, min(64, triton.next_power_of_2(max(num_gens, num_contexts, 1))))
    _strict_accept_kernel[(1,)](
        target_tokens,
        draft,
        accepted,
        num_accepted,
        pool,
        counter,
        num_contexts,
        num_gens,
        accepted.stride(0),
        float(frac),
        K=k,
        FORCE=mode,
        BASE_TOTAL=int(base_total),
        MAX_TOTAL=int(max_total),
        COUNTER_STRIDE=int(c_stride),
        SLOT_STRIDE=int(s_stride),
        POOL_MASK=int(pool_mask),
        BLOCK_B=block_b,
        BLOCK_K=max(8, triton.next_power_of_2(k + 1)),
        num_warps=1,
    )
    return accepted, num_accepted


@triton.jit
def _step0_epilogue_kernel(
    seq_lens_ptr,
    num_accepted_ptr,
    hid_ptr,
    hid_stride,
    pos_ptr,
    pos_row_stride,
    ids_out_ptr,
    hid_out_ptr,
    pos_out_ptr,
    seq_lens_w_ptr,
    kv_lens_ptr,
    acc_tok_ptr,
    acc_tok_stride,
    last_out_ptr,
    num_contexts,
    num_ctx_tokens,
    batch_size,
    K: tl.constexpr,
    R: tl.constexpr,
    H: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LAST: tl.constexpr,
):
    """One program per request ``i`` (every write is to row ``i`` only; ``seq_lens_ptr`` is never written)."""
    i = tl.program_id(0)
    is_ctx = i < num_contexts
    offs = tl.arange(0, BLOCK_C)
    lens = tl.load(seq_lens_ptr + offs, mask=(offs <= i) & is_ctx, other=0).to(tl.int64)
    acc = tl.load(num_accepted_ptr + i).to(tl.int64)
    # contexts: cumsum(seq_lens)[i] - 1 (last prompt token); generations: last accepted token of the block
    gid = tl.where(
        is_ctx,
        tl.sum(lens, axis=0) - 1,
        num_ctx_tokens + (i - num_contexts).to(tl.int64) * (K + 1) + acc - 1,
    )
    dkv = tl.where(is_ctx, 1, acc - K)
    tl.store(ids_out_ptr + i, gid)
    h = tl.arange(0, BLOCK_H)
    hm = h < H
    v = tl.load(hid_ptr + gid * hid_stride + h, mask=hm)
    tl.store(hid_out_ptr + i.to(tl.int64) * H + h, v, mask=hm)
    for r in tl.static_range(R):
        p = tl.load(pos_ptr + r * pos_row_stride + gid)
        for j in tl.static_range(1, K):
            tl.store(pos_out_ptr + ((j - 1) * R + r) * batch_size + i, p + j)
    tl.store(seq_lens_w_ptr + i, 1)
    kv = tl.load(kv_lens_ptr + i)
    tl.store(kv_lens_ptr + i, kv + dkv.to(kv.dtype))
    if LAST:
        # next_new_tokens[:, 0] = accepted_tokens[i, num_accepted[i] - 1]
        last = tl.load(acc_tok_ptr + i.to(tl.int64) * acc_tok_stride + acc - 1)
        tl.store(last_out_ptr + i, last.to(last_out_ptr.dtype.element_ty))


class Step0Epilogue:
    """Everything the first draft step does after the (k + 1)-token draft forward, besides the head, in one
    launch. ``gather_ids`` and ``hidden`` are the stock step-0 ``gather_ids`` /
    ``hidden_states[gather_ids]``; ``positions(j)`` is the position input of draft pass ``j`` (``1 <= j < K``),
    i.e. the stock ``_select_mtp_position_ids(position_ids, gather_ids) + j``; the kernel also applies the stock
    step-0 -> 1 metadata update to ``seq_lens_cuda[:b] = 1`` and ``kv_lens_cuda`` (contexts ``+= 1``,
    generations ``-= K - num_accepted``)."""

    def __init__(self, gather_ids, hidden, pos_all, pos_prefix):
        self.gather_ids = gather_ids
        self.hidden = hidden
        self._pos_all = pos_all
        self._pos_prefix = pos_prefix

    def positions(self, j: int) -> torch.Tensor:
        b = self.gather_ids.shape[0]
        return self._pos_all[j - 1].view(*self._pos_prefix, b)


def step0_epilogue(
    seq_lens: torch.Tensor,
    num_accepted: torch.Tensor,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    seq_lens_out: torch.Tensor,
    kv_lens: torch.Tensor,
    num_contexts: int,
    batch_size: int,
    draft_len: int,
    num_ctx_tokens: int,
    accepted_tokens: Optional[torch.Tensor] = None,
    last_out: Optional[torch.Tensor] = None,
) -> Optional[Step0Epilogue]:
    """``seq_lens`` must not alias ``seq_lens_out`` (read the pre-draft-loop copy). With ``accepted_tokens``
    ``[b, >= draft_len + 1]`` and ``last_out`` (contiguous ``[b]``), also ``last_out[i] =
    accepted_tokens[i, num_accepted[i] - 1]``. Returns None when the shapes are not supported (caller keeps the
    stock path)."""
    if (
        seq_lens.data_ptr() == seq_lens_out.data_ptr()
        or hidden_states.dim() != 2
        or hidden_states.stride(-1) != 1
        or position_ids.stride(-1) != 1
        or batch_size < 1
        or batch_size > 1024
        or draft_len < 1
    ):
        return None
    pos2 = position_ids.reshape(-1, position_ids.shape[-1])
    if pos2.dim() != 2 or (pos2.shape[0] > 1 and pos2.stride(0) < pos2.shape[1]):
        return None
    rows = pos2.shape[0]
    h = hidden_states.shape[1]
    dev = hidden_states.device
    ids = torch.empty((batch_size,), dtype=torch.int64, device=dev)
    hid = torch.empty((batch_size, h), dtype=hidden_states.dtype, device=dev)
    pos_all = torch.empty(
        (max(draft_len - 1, 1), rows, batch_size), dtype=position_ids.dtype, device=dev
    )
    last = accepted_tokens is not None and last_out is not None
    if last:
        assert (
            accepted_tokens.stride(-1) == 1
            and last_out.is_contiguous()
            and last_out.numel() >= batch_size
        )
    _step0_epilogue_kernel[(batch_size,)](
        seq_lens,
        num_accepted,
        hidden_states,
        hidden_states.stride(0),
        pos2,
        pos2.stride(0) if rows > 1 else 0,
        ids,
        hid,
        pos_all,
        seq_lens_out,
        kv_lens,
        accepted_tokens if last else ids,
        accepted_tokens.stride(0) if last else 0,
        last_out if last else ids,
        num_contexts,
        num_ctx_tokens,
        batch_size,
        K=draft_len,
        R=rows,
        H=h,
        BLOCK_C=max(16, triton.next_power_of_2(max(num_contexts, 1))),
        BLOCK_H=triton.next_power_of_2(h),
        LAST=last,
        num_warps=4,
    )
    return Step0Epilogue(ids, hid, pos_all, tuple(position_ids.shape[:-1]))


__all__ = [
    "MAX_DRAFT_LEN",
    "fused_strict_accept",
    "step0_epilogue",
    "Step0Epilogue",
]
