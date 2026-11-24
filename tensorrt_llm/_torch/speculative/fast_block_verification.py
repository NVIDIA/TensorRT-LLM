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
"""Block verification for the fused rejection-sampling path (``TLLM_SPEC_BLOCK_VERIFY=1``).

Block verification (Sun et al., ICLR 2025, arXiv:2403.10444, Algorithm 2) replaces the token-wise
accept loop of :func:`fast_rejection_sampling.fast_rs_verify` by a joint decision over the draft
block. It is a drop-in verifier with the same inputs and the same outputs: the emitted tokens follow
the target distribution exactly (Theorem 1 of the paper); by Theorem 2 of the paper the expected
number of accepted tokens is at least token-wise verification's, in expectation over the draft block
and not for every block. It needs the draft block's per-position proposals ``q_i`` and the
target's per-position distributions ``p_i`` along the sampled draft path, i.e. exactly what the
fused path already holds (``draft_probs`` and the ``K + 1`` target rows).

Notation (one request; ``d_0 .. d_{K-1}`` the drafts, ``q_i`` the stored proposal ``d_i`` was drawn
from, ``p_i`` the target distribution of position ``i`` given the draft prefix, ``i = 0 .. K``)::

    w_0 = 1,    w_{i+1} = min(w_i * p_i(d_i) / q_i(d_i), 1)                        i < K
    R_i = sum_x max(w_i * p_i(x) - q_i(x), 0)                                      i < K
    h_i = R_i / (R_i + 1 - w_i)   (1 where the denominator is 0),   h_K = w_K      i >= 1
    tau = max({0} + {i in 1..K : eta_i <= h_i}),  eta_i independent uniforms
    Y ~ max(w_tau * p_tau - q_tau, 0) normalised (tau < K),  Y ~ p_K (tau = K)

The request emits ``d_0 .. d_{tau-1}, Y`` (``num_accepted = tau + 1``). Token-wise verification
stops at the first rejected position; here every sub-block length ``i`` gets its own acceptance test
and the longest accepted sub-block wins. For ``K = 1`` the two are the same test: they coincide in
distribution, and decision by decision on the same uniforms (the token-wise kernel, given the seeds
``seed ^ BV_KEY_XOR``, draws the words block verification draws).

Kernels (the target-row statistics are the ones ``fast_rs_verify`` uses):

  1. ``fast_rejection_sampling._rs_row_stats_kernel`` on the ``K + 1`` target rows.
  2. :func:`_bv_resid_kernel` per (request, chunk): walks the ``K + 1`` positions once, carrying the
     weight ``w_i`` (a few scalar loads per position), and writes this chunk's mass and last nonzero
     index of ``max(w_i p_i - q_i, 0)`` (``p_K`` alone for the last position) for every position.
  3. :func:`_bv_pick_kernel` per request: sums the chunk masses in fixed order, forms ``h_i``, draws
     ``tau``, picks the correction token from the residual row of ``tau`` with the inverse-CDF rule of
     the token-wise path, and writes the ``[K + 1]`` output row.

Random stream: Philox key ``seed[b] ^ BV_KEY_XOR``, subsequence ``b``, offset ``offset[b]``, ``K + 1``
words per step and none used twice: word ``i - 1`` is the uniform ``eta_i`` of the sub-block of
length ``i`` (``i = 1 .. K``) and word ``K`` the correction token. The per-step offset window is the
one the token-wise verify uses, so the engine's offset bookkeeping is untouched. The key differs
from the one the fused draft kernels use (``seed ^ DRAFT_KEY_XOR``) and from the raw seed (the
token-wise verify stream, and the reference draft sampler, which reads only the first request's
seed), so ``eta`` is independent of the drafts whichever sampler drew them.

Exactness conditions (the paper's, the same ones token-wise rejection sampling has): every stored
proposal row ``q_i`` is the distribution the draft ``d_i`` was drawn from (given the draft prefix
and the context), drawn with randomness fresh at every position, and the verifier's uniforms are
independent of the drafts. A DSpark block with sampled-token Markov feedback
(``TLLM_DSPARK_MARKOV_SAMPLED_FEEDBACK=1``) satisfies them: its proposal at position ``i`` is a
function of the sampled previous token. So does a proposal that ignores the prefix, and a
placeholder draft with a one-hot proposal (what a request that has not drafted yet carries). Every
reduction runs in a fixed order, so tensor-parallel ranks holding the same logits draw the same
tokens and the kernels capture into CUDA graphs (static shapes, no host sync).

A correction row of zero mass (``w_tau * p_tau <= q_tau`` everywhere) emits token 0, the same
fallback as the token-wise kernel. With valid proposals a sub-block length ``tau < K`` of positive
probability has such a row only through float32 rounding, when ``p == q``; an invalid proposal that
dominates the target row (``q >= w * p``) has one by construction.

Only the temperature-only sampling mode is covered (``AdvancedSamplingMode.NO_TOPK_NO_TOPP``); with
a top-p nucleus the token-wise nucleus kernels stay in use.
"""

import os
from typing import Tuple

import torch
import triton
import triton.language as tl

from .fast_rejection_sampling import (
    _CHUNK,
    _NUM_WARPS,
    _WORKSPACE,
    _check_request_args,
    _curand_uniform,
    _inv_temperature,
    _num_chunks,
    _pick_chunk,
    _row_max_sum,
    _row_stats,
)

BLOCK_VERIFY_ENV_VAR = "TLLM_SPEC_BLOCK_VERIFY"

# Key of the block-verification uniform stream: seed ^ BV_KEY_XOR (see the module docstring).
BV_KEY_XOR = 0x1F83D9ABFB41BD6B
_BV_KEY_XOR = tl.constexpr(BV_KEY_XOR)


def block_verify_enabled() -> bool:
    """Whether ``TLLM_SPEC_BLOCK_VERIFY=1`` selects block verification for the fused path."""
    return os.environ.get(BLOCK_VERIFY_ENV_VAR, "0") == "1"


@triton.jit
def _bv_resid_kernel(
    tlogits_ptr,
    stride_row,
    temps_ptr,
    slot_ptr,
    dummy_row,
    num_draft,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    wts_ptr,
    rsum_ptr,
    rlast_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
):
    """Per (request, chunk): the weights ``w_i`` and this chunk's share of every position's residual.

    Row ``req * (K + 1) + i`` of ``rsum`` / ``rlast`` (``[rows, num_chunks]``) holds the chunk's
    mass and last nonzero index of ``max(w_i * p_i - q_i, 0)``; the last position (no proposal)
    holds ``p_K`` itself. ``wts[row]`` is ``w_i``, written by chunk 0.
    """
    req = tl.program_id(0).to(tl.int64)
    ch = tl.program_id(1)
    slot = tl.load(slot_ptr + req)
    is_pad = slot == dummy_row
    offs = ch * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    w = 1.0
    for i in range(num_draft):
        row = req * (num_draft + 1) + i
        big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
        inv_s = tl.math.div_rn(1.0, big_s)
        inv_t = _inv_temperature(tl.load(temps_ptr + row))
        x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
        x = tl.where(is_pad, 0.0, x * inv_t)
        p_t = tl.exp(x - big_m) * inv_s
        q = tl.load(probs_ptr + slot * stride_slot + i * stride_step + offs, mask=mask, other=0.0)
        res = tl.where(mask, tl.maximum(w * p_t - q, 0.0), 0.0)
        tl.store(rsum_ptr + row * num_chunks + ch, tl.sum(res, 0))
        tl.store(rlast_ptr + row * num_chunks + ch, tl.max(tl.where(res > 0.0, offs, -1), 0))
        if ch == 0:
            tl.store(wts_ptr + row, w)
        # Weight of the next sub-block from the two probabilities of the draft token. A token the
        # proposal excludes (q = 0: only a stale or mismatched proposal row can exclude a drawn
        # token) gets the limit of min(w * p / q, 1) as q goes to 0: 1 if w * p > 0, else 0.
        d = tl.load(draft_ptr + req * num_draft + i).to(tl.int64)
        d = tl.minimum(tl.maximum(d, 0), vocab - 1)
        x_d = tl.load(tlogits_ptr + row * stride_row + d)
        x_d = tl.where(is_pad, 0.0, x_d * inv_t)
        p_d = tl.exp(x_d - big_m) * inv_s
        q_d = tl.load(probs_ptr + slot * stride_slot + i * stride_step + d)
        w = tl.where(
            q_d > 0.0,
            tl.minimum(w * p_d / q_d, 1.0),
            tl.where((p_d > 0.0) & (w > 0.0), 1.0, 0.0),
        )

    # Position K has no proposal: its residual is the target row (weight 1).
    row = req * (num_draft + 1) + num_draft
    big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
    inv_s = tl.math.div_rn(1.0, big_s)
    inv_t = _inv_temperature(tl.load(temps_ptr + row))
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    x = tl.where(is_pad, 0.0, x * inv_t)
    res = tl.where(mask, tl.exp(x - big_m) * inv_s, 0.0)
    tl.store(rsum_ptr + row * num_chunks + ch, tl.sum(res, 0))
    tl.store(rlast_ptr + row * num_chunks + ch, tl.max(tl.where(res > 0.0, offs, -1), 0))
    if ch == 0:
        tl.store(wts_ptr + row, w)


@triton.jit
def _bv_pick_kernel(
    tlogits_ptr,
    stride_row,
    temps_ptr,
    slot_ptr,
    dummy_row,
    num_draft,
    pmax_ptr,
    psum_ptr,
    probs_ptr,
    stride_slot,
    stride_step,
    draft_ptr,
    seed_ptr,
    offset_ptr,
    wts_ptr,
    rsum_ptr,
    rlast_ptr,
    out_tokens_ptr,
    out_num_ptr,
    vocab,
    num_chunks,
    CHUNK: tl.constexpr,
    NC_POW2: tl.constexpr,
    KP1_POW2: tl.constexpr,
):
    """Per request: ``tau`` from the sub-block acceptance probabilities, the correction token from
    the residual row of ``tau``, and the ``[K + 1]`` output row."""
    req = tl.program_id(0).to(tl.int64)
    seed = tl.load(seed_ptr + req) ^ _BV_KEY_XOR
    offset = tl.load(offset_ptr + req)
    slot = tl.load(slot_ptr + req)
    is_pad = slot == dummy_row
    base = req * (num_draft + 1)

    # Sub-block of length i is accepted when eta_i <= h_i; the longest accepted one wins.
    chunk_ids = tl.arange(0, NC_POW2)
    chunk_mask = chunk_ids < num_chunks
    tau = 0
    for i in range(1, num_draft + 1):
        w_i = tl.load(wts_ptr + base + i)
        r_i = tl.sum(
            tl.load(rsum_ptr + (base + i) * num_chunks + chunk_ids, mask=chunk_mask, other=0.0), 0
        )
        denom = r_i + (1.0 - w_i)
        h = tl.where(denom > 0.0, r_i / denom, 1.0)
        h = tl.where(i == num_draft, w_i, h)
        eta = _curand_uniform(seed, offset, req, i - 1)
        tau = tl.where(eta <= h, i, tau)

    # Correction (or bonus) token: draw K, the word after the K sub-block draws.
    u = _curand_uniform(seed, offset, req, num_draft)
    row = base + tau
    total = 0.0
    for k in range(num_chunks):
        total = total + tl.load(rsum_ptr + row * num_chunks + k)
    u = u * total
    kstar, cbase, last = _pick_chunk(rsum_ptr, rlast_ptr, row, num_chunks, u)
    big_m, big_s = _row_max_sum(pmax_ptr, psum_ptr, row, num_chunks, NC_POW2)
    inv_s = tl.math.div_rn(1.0, big_s)
    inv_t = _inv_temperature(tl.load(temps_ptr + row))
    kk = tl.maximum(kstar, 0)
    offs = kk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < vocab
    x = tl.load(tlogits_ptr + row * stride_row + offs, mask=mask, other=0.0)
    x = tl.where(is_pad, 0.0, x * inv_t)
    p_t = tl.exp(x - big_m) * inv_s
    has_draft = tau < num_draft
    w_tau = tl.where(has_draft, tl.load(wts_ptr + row), 1.0)
    q_d = tl.load(
        probs_ptr + slot * stride_slot + tl.minimum(tau, num_draft - 1) * stride_step + offs,
        mask=mask & has_draft,
        other=0.0,
    )
    res = tl.where(mask, tl.maximum(w_tau * p_t - q_d, 0.0), 0.0)
    cdf = cbase + tl.cumsum(res, 0)
    hit = (res > 0.0) & (cdf > u)
    first = tl.min(tl.where(hit, offs, 2147483647), 0)
    chunk_last = tl.max(tl.where(res > 0.0, offs, -1), 0)
    token = tl.where(first < 2147483647, first, chunk_last)
    token = tl.where(kstar >= 0, token, last)
    token = tl.where(token >= 0, token, 0)
    cols = tl.arange(0, KP1_POW2)
    drafts = tl.load(draft_ptr + req * num_draft + cols, mask=cols < num_draft, other=0)
    out = tl.where(cols < tau, drafts, tl.where(cols == tau, token, -1))
    tl.store(
        out_tokens_ptr + req * (num_draft + 1) + cols, out.to(tl.int32), mask=cols < num_draft + 1
    )
    tl.store(out_num_ptr + req, (tau + 1).to(tl.int32))


def fast_rs_verify_block(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    draft_tokens: torch.Tensor,
    slot_ids: torch.Tensor,
    dummy_row: int,
    draft_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block-verify the generation requests' drafts against the target.

    Same contract as :func:`fast_rejection_sampling.fast_rs_verify`.

    Args:
        logits: ``[num_gens * (K + 1), vocab]`` fp32 target logits, request-major.
        temperatures: ``[num_gens * (K + 1)]`` fp32 per-row temperature.
        draft_tokens: ``[num_gens, K]`` contiguous int32 draft tokens.
        slot_ids: ``[num_gens]`` int64 ``draft_probs`` slot of each request.
        dummy_row: slot of CUDA-graph padding requests (their logits are treated as zeros).
        draft_probs: ``[slots, >= K, vocab]`` fp32 proposals stored by the previous draft step.
        seed, offset: ``[num_gens]`` int64 per-request Philox seed and offset, the tensors the
            token-wise verifier takes. The Philox key is ``seed ^ BV_KEY_XOR``, not the raw seed
            (see the module docstring).
    Returns:
        ``accepted_tokens [num_gens, K + 1]`` int32 (the accepted prefix of the drafts, the
        correction token, then ``-1``) and ``num_accepted [num_gens]`` int32 (accepted drafts + 1).
    """
    num_gens, k = draft_tokens.shape
    rows, vocab = logits.shape
    device = logits.device
    accepted = torch.empty((num_gens, k + 1), dtype=torch.int32, device=device)
    num_acc = torch.empty((num_gens,), dtype=torch.int32, device=device)
    if num_gens == 0:
        return accepted, num_acc
    assert rows == num_gens * (k + 1), (rows, num_gens, k)
    assert logits.dtype == torch.float32 and logits.stride(1) == 1
    assert draft_probs.dtype == torch.float32 and draft_probs.stride(2) == 1
    assert draft_probs.shape[2] == vocab and draft_probs.shape[1] >= k
    assert draft_tokens.dtype == torch.int32 and draft_tokens.is_contiguous()
    _check_request_args(temperatures, slot_ids, seed, offset, rows, num_gens)
    nc, nc_pow2 = _num_chunks(vocab)
    pmax, psum = _row_stats(logits, temperatures, 1, slot_ids, dummy_row, k + 1, "target")
    wts = _WORKSPACE.get("bv.wts", (rows,), torch.float32, device)
    rsum = _WORKSPACE.get("bv.rsum", (rows, nc), torch.float32, device)
    rlast = _WORKSPACE.get("bv.rlast", (rows, nc), torch.int32, device)
    _bv_resid_kernel[(num_gens, nc)](
        logits,
        logits.stride(0),
        temperatures,
        slot_ids,
        dummy_row,
        k,
        pmax,
        psum,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        wts,
        rsum,
        rlast,
        vocab,
        nc,
        CHUNK=_CHUNK,
        NC_POW2=nc_pow2,
        num_warps=_NUM_WARPS,
    )
    _bv_pick_kernel[(num_gens,)](
        logits,
        logits.stride(0),
        temperatures,
        slot_ids,
        dummy_row,
        k,
        pmax,
        psum,
        draft_probs,
        draft_probs.stride(0),
        draft_probs.stride(1),
        draft_tokens,
        seed,
        offset,
        wts,
        rsum,
        rlast,
        accepted,
        num_acc,
        vocab,
        nc,
        CHUNK=_CHUNK,
        NC_POW2=nc_pow2,
        KP1_POW2=triton.next_power_of_2(k + 1),
        num_warps=_NUM_WARPS,
    )
    return accepted, num_acc
