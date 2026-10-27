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
"""

from typing import Callable, Optional, Tuple

import torch

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
