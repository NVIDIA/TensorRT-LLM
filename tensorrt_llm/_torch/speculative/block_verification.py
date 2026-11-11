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
"""Block verification for one-model speculative decoding.

Token-level rejection sampling (Leviathan et al., 2023) tests the drafts one at
a time: draft ``j`` survives with probability ``min(1, p_j(x_j) / q_j(x_j))``
and the first failure resamples from ``max(p_j - q_j, 0)``. Block verification
(Sun et al., 2024, https://arxiv.org/abs/2403.10444) judges the drafted block
jointly, so a draft the drafter over-proposed can still be kept when the drafts
after it make up for it. It is lossless like the token-level rule, identical to
it for a single draft, and accepts at least as many tokens in expectation.

With 0-based draft positions ``j = 0..K-1``, drafts ``x_j ~ q_j``, target rows
``p_j`` (``p_K`` is the bonus row) and i.i.d. uniforms ``u_j``:

* capped running joint ratio ``w_j = min(1, w_{j-1} p_j(x_j) / q_j(x_j))``,
  ``w_{-1} = 1``;
* look-ahead residual mass ``r_j = sum_x max(w_j p_{j+1}(x) - q_{j+1}(x), 0)``
  for ``j < K - 1``;
* acceptance thresholds ``h_j = r_j / (r_j + 1 - w_j)`` (1 when the denominator
  is 0) for ``j < K - 1``, and ``h_{K-1} = w_{K-1}``;
* accepted length ``L``: the largest ``j + 1`` with ``u_j <= h_j``, else 0.
  Every position is tested; a failed position does not stop the scan;
* emitted token: ``y ~ max(w_{L-1} p_L - q_L, 0)`` renormalized when ``L < K``,
  the bonus ``y ~ p_K`` when ``L = K``.

This is the rule vLLM serves as ``rejection_sample_method="block"``.

Everything here is tensor arithmetic on static shapes with no host
synchronization, so it can run inside a captured CUDA graph.

The torch functions above are the rule from NVIDIA/TensorRT-LLM#21475 (dense:
several passes over ``[B, K, V]`` fp32 and about a hundred small launches for
the uniforms). ``block_verify_one_model`` is the same rule as two Triton
launches that read each target and draft row once; ``TRTLLM_SPEC_BLOCK_VERIFICATION=1``
selects it for batches of up to ``TRTLLM_SPEC_BLOCK_VERIFICATION_MAX_BATCH`` requests
(``verify_drafts``; larger batches keep the token-level kernel, which stops at the first
rejection and is cheaper there). The engine never takes the torch path; the torch
functions are the Triton path's reference (``block_verify_reference``), used by the tests.
"""

import os
from typing import Callable, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

_MASK32 = 0xFFFFFFFF
# Keeps these draws apart from any other hash of the same (seed, offset).
_UNIFORM_DOMAIN = 0x5BD1E995


def _mul_u32(values: torch.Tensor, constant: int) -> torch.Tensor:
    """``(values * constant) mod 2**32`` for int64 tensors holding uint32 values.

    The 32-bit constant is split into 16-bit halves so no intermediate product
    leaves the int64 range.
    """
    low = values * (constant & 0xFFFF)
    high = ((values * (constant >> 16)) & 0xFFFF) << 16
    return (low + high) & _MASK32


def _mix_u32(values: torch.Tensor) -> torch.Tensor:
    """A full-avalanche 32-bit integer finalizer (lowbias32)."""
    values = values ^ (values >> 16)
    values = _mul_u32(values, 0x7FEB352D)
    values = values ^ (values >> 15)
    values = _mul_u32(values, 0x846CA68B)
    return values ^ (values >> 16)


def counter_uniforms(seed: torch.Tensor, offset: torch.Tensor, num_draws: int) -> torch.Tensor:
    """Counter-based uniforms in the open interval (0, 1).

    Draw ``d`` of row ``r`` is a hash of ``(seed[r], offset[r], d)``: the same
    key gives the same value on every rank and in every replay of a captured
    graph, and a key never seen before gives a fresh, independent value. That
    makes it the uniform source for the per-request ``(seed, offset)`` streams
    the one-model samplers already carry.

    Args:
        seed: ``[rows]`` integer seeds.
        offset: ``[rows]`` integer offsets.
        num_draws: values per row.
    Returns:
        ``[rows, num_draws]`` float64, with 32 random bits per value.
    """
    seed = seed.to(torch.int64).unsqueeze(-1)
    offset = offset.to(torch.int64).unsqueeze(-1)
    draws = torch.arange(num_draws, dtype=torch.int64, device=seed.device)
    key = _mix_u32((seed & _MASK32) ^ _UNIFORM_DOMAIN)
    key = _mix_u32(key ^ ((seed >> 32) & _MASK32))
    key = _mix_u32(key ^ (offset & _MASK32))
    key = _mix_u32(key ^ ((offset >> 32) & _MASK32))
    bits = _mix_u32(_mix_u32(key ^ draws))
    return (bits.to(torch.float64) + 0.5) * 2.0**-32


def sample_from_weights(weights: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF draw from each row's nonnegative, unnormalized weights.

    Args:
        weights: ``[rows, vocab]``.
        uniforms: ``[rows]`` values in (0, 1).
    Returns:
        ``[rows]`` int64 indices. A row whose weights sum to zero gets an index
        of no meaning; callers must not pass one.
    """
    cdf = weights.to(torch.float64).cumsum(-1)
    threshold = uniforms.to(torch.float64).unsqueeze(-1) * cdf[:, -1:]
    index = torch.searchsorted(cdf, threshold, right=True).squeeze(-1)
    return index.clamp_(max=weights.shape[-1] - 1)


def block_acceptance_thresholds(
    draft_tokens: torch.Tensor, draft_probs: torch.Tensor, target_probs: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The running capped joint ratios and the acceptance thresholds.

    Args:
        draft_tokens: ``[B, K]`` draft token ids.
        draft_probs: ``[B, K, V]`` distributions the drafts were drawn from.
        target_probs: ``[B, K + 1, V]`` target distributions (the request's
            sampling filters applied).
    Returns:
        ``log_w`` and ``h``, both ``[B, K]`` float32: ``log_w[:, j] = log w_j``
        and ``h[:, j] = h_j`` in the notation of the module docstring.
    """
    num_drafts = draft_tokens.shape[1]
    vocab_size = target_probs.shape[-1]
    tokens = draft_tokens.long().clamp(0, vocab_size - 1).unsqueeze(-1)
    p_x = target_probs[:, :num_drafts].gather(-1, tokens).squeeze(-1).float()
    q_x = draft_probs.gather(-1, tokens).squeeze(-1).float()
    # A draft its own proposal gives no mass to was not drawn from it; such a
    # draft (and a non-finite row) must never be accepted.
    log_ratio = torch.where(q_x > 0, p_x.log() - q_x.log(), torch.full_like(q_x, float("-inf")))
    log_ratio = torch.nan_to_num(log_ratio, nan=float("-inf"))

    log_w = torch.empty_like(log_ratio)
    running = torch.zeros_like(log_ratio[:, 0])
    for j in range(num_drafts):
        running = torch.clamp(running + log_ratio[:, j], max=0.0)
        log_w[:, j] = running
    w = log_w.exp()

    # h_{K-1} = w_{K-1}; the look-ahead thresholds of positions 0..K-2 are
    # computed for all of them at once (one [B, K-1, V] pass, no per-position
    # launches).
    h = w.clone()
    if num_drafts > 1:
        residual = w[:, :-1, None] * target_probs[:, 1:num_drafts].float()
        residual.sub_(draft_probs[:, 1:].float()).clamp_min_(0.0)
        mass = residual.sum(-1)
        denom = mass + 1.0 - w[:, :-1]
        h[:, :-1] = torch.where(denom > 0, mass / denom, torch.ones_like(mass))
    return log_w, h


def block_verify(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniforms: torch.Tensor,
    resample_uniforms: Optional[torch.Tensor] = None,
    sample_emitted: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Verify a batch of drafted blocks with block verification.

    Args:
        draft_tokens: ``[B, K]`` draft token ids.
        draft_probs: ``[B, K, V]`` distributions the drafts were drawn from.
        target_probs: ``[B, K + 1, V]`` target distributions (the request's
            sampling filters applied); row ``K`` is the bonus row.
        uniforms: ``[B, K]`` acceptance uniforms in (0, 1).
        resample_uniforms: ``[B]`` uniforms in (0, 1) for the emitted token,
            drawn by inverse CDF (:func:`sample_from_weights`).
        sample_emitted: alternatively, ``([B, V] nonnegative weights) -> [B]``
            ids: draws the emitted token from the normalized weights (e.g. a
            seeded sampler kernel). Exactly one of the two must be given.
    Returns:
        ``accepted_tokens`` ``[B, K + 1]`` int32 -- the ``L`` accepted drafts,
        the emitted token, then ``-1`` -- and ``num_accepted_tokens`` ``[B]``
        int32 ``= L + 1``, the layout of the token-level kernel.
    """
    num_drafts = draft_tokens.shape[1]
    vocab_size = target_probs.shape[-1]
    log_w, h = block_acceptance_thresholds(draft_tokens, draft_probs, target_probs)

    positions = torch.arange(1, num_drafts + 1, dtype=torch.int64, device=draft_tokens.device)
    passed = uniforms.to(torch.float64) <= h.to(torch.float64)
    accepted_len = (positions * passed).amax(-1)

    row = accepted_len.view(-1, 1, 1)
    p_row = target_probs.gather(1, row.expand(-1, 1, vocab_size)).squeeze(1).float()
    q_row = (
        draft_probs.gather(1, row.clamp(max=num_drafts - 1).expand(-1, 1, vocab_size))
        .squeeze(1)
        .float()
    )
    # w_{L-1}, with w_{-1} = 1.
    log_w_prev = torch.cat([torch.zeros_like(log_w[:, :1]), log_w], dim=1).gather(
        1, accepted_len.view(-1, 1)
    )
    is_bonus = (accepted_len == num_drafts).unsqueeze(-1)
    residual = torch.where(is_bonus, p_row, (log_w_prev.exp() * p_row - q_row).clamp_min(0.0))
    # The residual is empty only with probability zero; fall back to the
    # target row so rounding can never leave nothing to draw from.
    has_mass = residual.sum(-1, keepdim=True) > 0
    residual = torch.where(has_mass, residual, p_row)
    if sample_emitted is not None:
        emitted = sample_emitted(residual).long().clamp(0, vocab_size - 1)
    else:
        assert resample_uniforms is not None, (
            "block_verify needs resample_uniforms or sample_emitted"
        )
        emitted = sample_from_weights(residual, resample_uniforms)

    slots = torch.arange(num_drafts + 1, dtype=torch.int64, device=draft_tokens.device)
    drafts = torch.cat([draft_tokens.long(), torch.zeros_like(draft_tokens[:, :1]).long()], dim=1)
    accepted_tokens = torch.where(
        slots < accepted_len.unsqueeze(-1), drafts, torch.full_like(drafts, -1)
    )
    accepted_tokens = torch.where(
        slots == accepted_len.unsqueeze(-1), emitted.unsqueeze(-1), accepted_tokens
    )
    return accepted_tokens.to(torch.int32), (accepted_len + 1).to(torch.int32)


# ---------------------------------------------------------------------------
# Fused path
# ---------------------------------------------------------------------------

_ENV = "TRTLLM_SPEC_BLOCK_VERIFICATION"
# Vocabulary chunks per row: the parallelism the row reductions get, and the
# granularity at which the emitted token's chunk is located.
_MAX_CHUNKS = 16
_BLOCK = 2048


def block_verification_enabled() -> bool:
    """Whether rejection-sampled steps verify drafted blocks jointly (default off)."""
    return os.environ.get(_ENV, "0") == "1"


@triton.jit
def _mix32(x):
    # lowbias32 on uint32; the same finalizer as ``_mix_u32``.
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    return x ^ (x >> 16)


@triton.jit
def _draw(key, d: tl.constexpr):
    # Draw ``d`` of ``counter_uniforms``: (0, 1) float64 with 32 random bits.
    bits = _mix32(_mix32(key ^ d))
    return (bits.to(tl.float64) + 0.5) * 2.3283064365386963e-10


@triton.jit
def _block_mass_kernel(
    tokens_ptr,
    p_ptr,
    q_ptr,
    part_ptr,
    V,
    stride_pb,
    stride_pr,
    stride_qb,
    stride_qr,
    K: tl.constexpr,
    NC: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``part[b, r, c]``: chunk ``c`` of ``sum_x max(w_{r-1} p_r(x) - q_r(x), 0)``.

    ``w_{r-1}`` is the capped running joint ratio of the first ``r`` drafts
    (1 for ``r = 0``). Row ``K`` is the bonus row: weight 1 and no draft row.
    For ``1 <= r <= K - 1`` the row sum is the look-ahead residual mass; for any
    ``r`` it is the mass of the distribution the token emitted at length ``r``
    is drawn from.
    """
    b = tl.program_id(0)
    r = tl.program_id(1)
    c = tl.program_id(2)
    b64 = b.to(tl.int64)
    w = 1.0
    for i in range(0, r):
        x = tl.load(tokens_ptr + b * K + i).to(tl.int64)
        x = tl.minimum(tl.maximum(x, 0), V - 1)
        pv = tl.load(p_ptr + b64 * stride_pb + i * stride_pr + x)
        qv = tl.load(q_ptr + b64 * stride_qb + i * stride_qr + x)
        # A draft its own proposal gives no mass was not drawn from it (and a
        # non-finite row): never accepted.
        w = tl.where((qv > 0.0) & (pv == pv), tl.minimum(w * pv / qv, 1.0), 0.0)
    wr = tl.where(r == K, 1.0, w)
    p_row = p_ptr + b64 * stride_pb + r * stride_pr
    q_row = q_ptr + b64 * stride_qb + tl.minimum(r, K - 1) * stride_qr
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    lo = c * CHUNK
    hi = tl.minimum(lo + CHUNK, V)
    for off in range(lo, hi, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < hi
        p_vec = tl.load(p_row + idx, mask=m, other=0.0)
        q_vec = tl.load(q_row + idx, mask=m & (r < K), other=0.0)
        acc += tl.maximum(wr * p_vec - q_vec, 0.0)
    tl.store(part_ptr + (b * (K + 1) + r) * NC + c, tl.sum(acc, axis=0))


@triton.jit
def _block_finish_kernel(
    tokens_ptr,
    p_ptr,
    q_ptr,
    part_ptr,
    seed_ptr,
    offset_ptr,
    slots_ptr,
    stats_ptr,
    out_tokens_ptr,
    out_num_ptr,
    V,
    dummy_slot,
    stride_pb,
    stride_pr,
    stride_qb,
    stride_qr,
    K: tl.constexpr,
    NC: tl.constexpr,
    NCP: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_STATS: tl.constexpr,
):
    """One program per request: accepted length, then the emitted token.

    Position ``j`` passes when ``u_j <= h_j``; the accepted length is the last
    passing ``j + 1`` (0 if none). The emitted token is an inverse-CDF draw from
    the row-``L`` weights of ``_block_mass_kernel``: the chunk is found from the
    row's chunk sums and only that chunk is scanned.
    """
    b = tl.program_id(0)
    b64 = b.to(tl.int64)
    seed = tl.load(seed_ptr + b).to(tl.int64)
    offset = tl.load(offset_ptr + b).to(tl.int64)
    key = _mix32((seed & 0xFFFFFFFF).to(tl.uint32) ^ 0x5BD1E995)
    key = _mix32(key ^ ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32))
    key = _mix32(key ^ (offset & 0xFFFFFFFF).to(tl.uint32))
    key = _mix32(key ^ ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32))
    ar = tl.arange(0, NCP)

    w = 1.0
    wl = 1.0
    length = 0
    no_mass = 0
    for j in tl.static_range(K):
        x = tl.load(tokens_ptr + b * K + j).to(tl.int64)
        x = tl.minimum(tl.maximum(x, 0), V - 1)
        pv = tl.load(p_ptr + b64 * stride_pb + j * stride_pr + x)
        qv = tl.load(q_ptr + b64 * stride_qb + j * stride_qr + x)
        # A drafted token its stored proposal gives no mass was not drawn from it: counted, so a
        # stale or mismatched draft row shows as a nonzero counter, not as a quiet change in AL.
        no_mass += (qv > 0.0).to(tl.int32) ^ 1
        w = tl.where((qv > 0.0) & (pv == pv), tl.minimum(w * pv / qv, 1.0), 0.0)
        if j < K - 1:
            part = tl.load(part_ptr + (b * (K + 1) + j + 1) * NC + ar, mask=ar < NC, other=0.0)
            mass = tl.sum(part, axis=0)
            denom = mass + 1.0 - w
            h = tl.where(denom > 0.0, mass / denom, 1.0)
        else:
            h = w
        hit = _draw(key, j) <= h.to(tl.float64)
        length = tl.where(hit, j + 1, length)
        wl = tl.where(hit, w, wl)

    # The token emitted after ``length`` accepted drafts is drawn from
    # max(w_{length-1} p_length - q_length, 0), or from p_K for a full block.
    wr = tl.where(length == K, 1.0, wl)
    pc = tl.load(part_ptr + (b * (K + 1) + length) * NC + ar, mask=ar < NC, other=0.0)
    total = tl.sum(pc, axis=0)
    thr = (_draw(key, K) * total.to(tl.float64)).to(tl.float32)
    cs = tl.cumsum(pc, axis=0)
    chunk = tl.sum(((cs <= thr) & (ar < NC)).to(tl.int32), axis=0)
    # Rounding can put the threshold at the very end of the support: stay on its last chunk.
    chunk = tl.maximum(tl.minimum(chunk, tl.max(tl.where(pc > 0.0, ar, -1), axis=0)), 0)
    # The threshold relative to the chunk: the scan below adds up values of the chunk's size,
    # not of the whole row's.
    local = tl.maximum(thr - tl.sum(tl.where(ar < chunk, pc, 0.0), axis=0), 0.0)

    p_row = p_ptr + b64 * stride_pb + length * stride_pr
    q_row = q_ptr + b64 * stride_qb + tl.minimum(length, K - 1) * stride_qr
    lo = chunk * CHUNK
    hi = tl.minimum(lo + CHUNK, V)
    run = 0.0
    tok = -1
    last = -1
    for off in range(lo, hi, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < hi
        p_vec = tl.load(p_row + idx, mask=m, other=0.0)
        q_vec = tl.load(q_row + idx, mask=m & (length < K), other=0.0)
        val = tl.maximum(wr * p_vec - q_vec, 0.0)
        crossed = (run + tl.cumsum(val, axis=0) > local) & (val > 0.0) & m
        first = tl.min(tl.where(crossed, idx, V), axis=0)
        tok = tl.where((tok < 0) & (first < V), first, tok)
        last = tl.maximum(last, tl.max(tl.where((val > 0.0) & m, idx, -1), axis=0))
        run += tl.sum(val, axis=0)
    # No crossing (rounding at the end of the support, or a degenerate row): the
    # last token with mass, else token 0, so the draw is always in the vocabulary. A row
    # whose residual has no mass at all (w p_L <= q_L everywhere: p = q to the last bit, or a
    # padding/NaN row) takes token 0 here where the torch reference falls back to the target row.
    tok = tl.where(tok >= 0, tok, last)
    tok = tl.maximum(tok, 0)

    for i in tl.static_range(K + 1):
        d = tl.load(tokens_ptr + b * K + min(i, K - 1))
        d = tl.minimum(tl.maximum(d, 0), V - 1)
        tl.store(
            out_tokens_ptr + b * (K + 1) + i,
            tl.where(i < length, d, tl.where(i == length, tok, -1)),
        )
    tl.store(out_num_ptr + b, length + 1)
    if HAS_STATS:
        if tl.load(slots_ptr + b) != dummy_slot:
            tl.atomic_add(stats_ptr + 1, (length * 0 + 1).to(tl.int64))
            tl.atomic_add(stats_ptr + 2, (length + 1).to(tl.int64))
            tl.atomic_add(stats_ptr + 3, no_mass.to(tl.int64))
        if b == 0:
            tl.atomic_add(stats_ptr, (length * 0 + 1).to(tl.int64))


@triton.jit
def _count_token_level_kernel(num_ptr, slots_ptr, stats_ptr, dummy_slot):
    """Engagement counters of a step the token-level rule served (``_STATS[4:7]``)."""
    b = tl.program_id(0)
    n = tl.load(num_ptr + b).to(tl.int64)
    if tl.load(slots_ptr + b) != dummy_slot:
        tl.atomic_add(stats_ptr + 5, n * 0 + 1)
        tl.atomic_add(stats_ptr + 6, n)
    if b == 0:
        tl.atomic_add(stats_ptr + 4, n * 0 + 1)


# Device counters for the engagement line, keyed by device and allocated by
# ``poll_block_verification_stats`` (eager, before any graph capture) so a captured launch never
# allocates them: [fused steps, fused requests, fused accepted tokens, drafts the proposal gave
# no mass, token-level steps, token-level requests, token-level accepted tokens].
_STATS: dict = {}
_POLLS = 0
_LAST_POLL = (0,) * 7
_POLL_EVERY = 2048
_MAX_BATCH_ENV = "TRTLLM_SPEC_BLOCK_VERIFICATION_MAX_BATCH"
# The fused rule's cost is the bytes it reads (every row of the block), while the token-level
# kernel stops at the first rejection: the fused rule is cheaper for small batches and dearer for
# large ones. 48 is the crossover for 7-draft blocks over a vocabulary of about 200k.
_DEFAULT_MAX_BATCH = 48


def block_verification_max_batch() -> int:
    """Largest generation batch verified jointly; larger batches keep the token-level rule."""
    return int(os.environ.get(_MAX_BATCH_ENV, str(_DEFAULT_MAX_BATCH)))


def _layout(vocab_size: int) -> Tuple[int, int, int]:
    """(elements per load, vocabulary chunks per row, elements per chunk)."""
    block = _BLOCK if vocab_size >= 4 * _BLOCK else max(16, triton.next_power_of_2(vocab_size))
    chunks = max(1, min(_MAX_CHUNKS, triton.cdiv(vocab_size, block)))
    return block, chunks, triton.cdiv(triton.cdiv(vocab_size, chunks), block) * block


def _aligned(t: torch.Tensor) -> torch.Tensor:
    """``t`` itself, or a copy when its pointer is not 16-byte aligned.

    Triton compiles a separate kernel specialization for each pointer alignment, and the per-request
    vectors arrive as views at the batch's first generation row (odd when the batch has an odd
    number of context requests): an unaligned view would compile a new specialization inside the
    serving loop, after the graphs are captured. The test is on the host and the copy is two
    tiny launches, taken only on such steps (graph captures run with no context requests).
    """
    return t if t.data_ptr() % 16 == 0 else t.clone()


def block_verify_one_model(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    slot_ids: Optional[torch.Tensor] = None,
    dummy_slot: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Block verification of a batch of drafted blocks: two launches, no host sync.

    Same rule and output layout as :func:`block_verify`; the random numbers are
    ``counter_uniforms(seed, offset, K + 1)``: draws ``0..K-1`` are the
    acceptance uniforms and draw ``K`` picks the emitted token by inverse CDF.

    Args:
        draft_tokens: ``[B, K]`` int32.
        draft_probs: ``[B, K, V]`` fp32, the distributions the drafts were drawn from.
        target_probs: ``[B, K + 1, V]`` fp32 (the request's sampling filters applied).
        seed, offset: ``[B]`` int64, the request's acceptance-slot Philox key.
        slot_ids: ``[B]`` sequence slots; rows at ``dummy_slot`` (CUDA-graph
            padding) are left out of the engagement counters.
    Returns:
        ``accepted_tokens`` ``[B, K + 1]`` int32 and ``num_accepted_tokens`` ``[B]`` int32.
    """
    batch, num_drafts = draft_tokens.shape
    vocab_size = target_probs.shape[-1]
    seed, offset = _aligned(seed), _aligned(offset)
    if slot_ids is not None:
        slot_ids = _aligned(slot_ids)
    assert draft_probs.shape == (batch, num_drafts, vocab_size)
    assert target_probs.shape == (batch, num_drafts + 1, vocab_size)
    assert draft_probs.stride(-1) == 1 and target_probs.stride(-1) == 1
    assert draft_tokens.is_contiguous() and draft_tokens.dtype == torch.int32
    block, chunks, chunk = _layout(vocab_size)
    part = torch.empty(
        (batch, num_drafts + 1, chunks), dtype=torch.float32, device=draft_tokens.device
    )
    _block_mass_kernel[(batch, num_drafts + 1, chunks)](
        draft_tokens,
        target_probs,
        draft_probs,
        part,
        vocab_size,
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        K=num_drafts,
        NC=chunks,
        CHUNK=chunk,
        BLOCK=block,
    )
    accepted_tokens = torch.empty(
        (batch, num_drafts + 1), dtype=torch.int32, device=draft_tokens.device
    )
    num_accepted = torch.empty(batch, dtype=torch.int32, device=draft_tokens.device)
    stats = _STATS.get(draft_tokens.device) if slot_ids is not None else None
    _block_finish_kernel[(batch,)](
        draft_tokens,
        target_probs,
        draft_probs,
        part,
        seed,
        offset,
        slot_ids if stats is not None else seed,
        stats if stats is not None else part,
        accepted_tokens,
        num_accepted,
        vocab_size,
        dummy_slot,
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        K=num_drafts,
        NC=chunks,
        NCP=max(2, triton.next_power_of_2(chunks)),
        CHUNK=chunk,
        BLOCK=block,
        HAS_STATS=stats is not None,
    )
    logger.info_once(
        f"Block verification: drafted blocks of {num_drafts} are verified jointly in two fused "
        f"launches ({_ENV}=1); vocabulary {vocab_size}, {chunks} chunks of {chunk}.",
        key="spec.block_verification_fused",
    )
    return accepted_tokens, num_accepted


def verify_drafts(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    slot_ids: Optional[torch.Tensor],
    dummy_slot: int,
    token_level: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The step's rejection verification when ``TRTLLM_SPEC_BLOCK_VERIFICATION=1``.

    Batches of up to ``TRTLLM_SPEC_BLOCK_VERIFICATION_MAX_BATCH`` (48) requests take
    :func:`block_verify_one_model`; larger ones take ``token_level`` (the engine's
    ``rejection_sampling_one_model``, called as it is called with the env unset), which is
    cheaper there. Both rules are lossless. The batch is static in a captured graph, so the
    choice is made at capture. The engagement counters say which rule served each step.
    """
    batch = draft_tokens.shape[0]
    limit = block_verification_max_batch()
    logger.info_once(
        f"Block verification ({_ENV}=1): batches of up to {limit} requests are verified jointly, "
        f"larger batches keep the token-level rule ({_MAX_BATCH_ENV}).",
        key="spec.block_verification_route",
    )
    if batch <= limit:
        return block_verify_one_model(
            draft_tokens, draft_probs, target_probs, seed, offset, slot_ids, dummy_slot
        )
    tokens, num_accepted = token_level(
        draft_probs=draft_probs,
        draft_token_ids=draft_tokens,
        target_probs=target_probs,
        deterministic=True,
        seed=seed,
        offset=offset,
    )
    stats = _STATS.get(num_accepted.device) if slot_ids is not None else None
    if stats is not None:
        _count_token_level_kernel[(batch,)](num_accepted, _aligned(slot_ids), stats, dummy_slot)
    logger.info_once(
        f"Block verification: a batch of {batch} requests is served by the token-level rule.",
        key="spec.block_verification_token_level",
    )
    return tokens, num_accepted


def block_verify_reference(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """:func:`block_verify_one_model` in torch (the dense #21475 rule), same random numbers."""
    num_drafts = draft_tokens.shape[1]
    uniforms = counter_uniforms(seed, offset, num_drafts + 1)
    return block_verify(
        draft_tokens,
        draft_probs,
        target_probs,
        uniforms[:, :num_drafts],
        resample_uniforms=uniforms[:, num_drafts],
    )


def poll_block_verification_stats() -> None:
    """Log the engagement line every ``_POLL_EVERY`` calls (host side, once per step).

    Allocates the counters on first call. Reading them waits for the device, so it
    runs rarely; a no-op unless ``TRTLLM_SPEC_BLOCK_VERIFICATION=1``.
    """
    global _POLLS, _LAST_POLL
    if not block_verification_enabled():
        return
    device = torch.device("cuda", torch.cuda.current_device())
    stats = _STATS.get(device)
    if stats is None:
        _STATS[device] = torch.zeros(7, dtype=torch.int64, device=device)
        return
    _POLLS += 1
    if _POLLS % _POLL_EVERY:
        return
    cur = tuple(stats.tolist())
    steps, rows, accepted, no_mass, steps_t, rows_t, accepted_t = cur
    last = _LAST_POLL
    _LAST_POLL = cur

    def mean(tokens: int, requests: int) -> str:
        return f"{tokens / requests:.3f}" if requests else "n/a"

    if rows > last[1] or rows_t > last[5]:
        logger.info(
            f"Block verification (up to {block_verification_max_batch()} requests joint, larger "
            f"token-level): joint {steps} steps / {rows} requests, mean accepted length "
            f"{mean(accepted, rows)} ({mean(accepted - last[2], rows - last[1])} recent); "
            f"token-level {steps_t} steps / {rows_t} requests, mean accepted length "
            f"{mean(accepted_t, rows_t)} ({mean(accepted_t - last[6], rows_t - last[5])} recent); "
            f"{no_mass} drafts its proposal gave no mass (must be 0)."
        )
    if no_mass > last[3]:
        logger.warning(
            f"Block verification: {no_mass - last[3]} drafted tokens had no mass in their "
            "stored draft distribution: the stored rows are not those the drafts were drawn from."
        )
