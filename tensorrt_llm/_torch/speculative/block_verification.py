# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block verification (Sun et al., 2024, arXiv:2403.10444) for one-model speculative decoding.

For drafts ``x_j ~ q_j`` (``j < K``), target rows ``p_j`` (bonus row ``p_K``) and uniforms ``u_j``:
``w_j = min(1, w_{j-1} p_j(x_j) / q_j(x_j))``, ``r_j = sum_x max(w_j p_{j+1}(x) - q_{j+1}(x), 0)``,
``h_j = r_j / (r_j + 1 - w_j)`` and ``h_{K-1} = w_{K-1}``. ``L`` is the largest ``j + 1`` with
``u_j <= h_j`` (else 0); the emitted token is drawn from ``max(w_{L-1} p_L - q_L, 0)``, or ``p_K`` if ``L = K``.
"""

from typing import Callable

import torch

_MASK32 = 0xFFFFFFFF


def _mul_u32(values: torch.Tensor, constant: int) -> torch.Tensor:
    """``(values * constant) mod 2**32``, multiplying 16-bit halves so int64 never overflows."""
    low = values * (constant & 0xFFFF)
    high = ((values * (constant >> 16)) & 0xFFFF) << 16
    return (low + high) & _MASK32


def _mix_u32(values: torch.Tensor) -> torch.Tensor:
    """The lowbias32 finalizer."""
    values = values ^ (values >> 16)
    values = _mul_u32(values, 0x7FEB352D)
    values = values ^ (values >> 15)
    values = _mul_u32(values, 0x846CA68B)
    return values ^ (values >> 16)


def counter_uniforms(seed: torch.Tensor, offset: torch.Tensor, num_draws: int) -> torch.Tensor:
    """``[rows, num_draws]`` float64 uniforms in (0, 1), hashed from each row's ``(seed, offset)``.

    The same key gives the same values on every rank and in every graph replay; a new offset
    gives fresh ones.
    """
    seed = seed.to(torch.int64).unsqueeze(-1)
    offset = offset.to(torch.int64).unsqueeze(-1)
    draws = torch.arange(num_draws, dtype=torch.int64, device=seed.device)
    key = _mix_u32((seed & _MASK32) ^ 0x5BD1E995)
    key = _mix_u32(key ^ ((seed >> 32) & _MASK32))
    key = _mix_u32(key ^ (offset & _MASK32))
    key = _mix_u32(key ^ ((offset >> 32) & _MASK32))
    bits = _mix_u32(_mix_u32(key ^ draws))
    return (bits.to(torch.float64) + 0.5) * 2.0**-32


def block_verify(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniforms: torch.Tensor,
    sample_emitted: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify ``[B, K]`` drafts; ``sample_emitted`` draws from ``[B, V]`` nonnegative weights.

    Returns the token-level kernel's layout: ``[B, K + 1]`` int32 tokens (accepted drafts, the emitted
    token, then -1) and ``[B]`` int32 counts including the emitted token. No host sync (graph-safe).
    """
    num_drafts = draft_tokens.shape[1]
    vocab_size = target_probs.shape[-1]
    tokens = draft_tokens.long().clamp(0, vocab_size - 1).unsqueeze(-1)
    p_x = target_probs[:, :num_drafts].gather(-1, tokens).squeeze(-1).float()
    q_x = draft_probs.gather(-1, tokens).squeeze(-1).float()
    # A draft its proposal gives no mass (or a non-finite row) is never accepted.
    log_ratio = torch.where(q_x > 0, p_x.log() - q_x.log(), torch.full_like(q_x, float("-inf")))
    log_ratio = torch.nan_to_num(log_ratio, nan=float("-inf"))
    log_w = torch.empty_like(log_ratio)
    running = torch.zeros_like(log_ratio[:, 0])
    for j in range(num_drafts):
        running = torch.clamp(running + log_ratio[:, j], max=0.0)
        log_w[:, j] = running
    w = log_w.exp()
    h = w.clone()
    if num_drafts > 1:
        residual = w[:, :-1, None] * target_probs[:, 1:num_drafts].float()
        residual.sub_(draft_probs[:, 1:].float()).clamp_min_(0.0)
        mass = residual.sum(-1)
        denom = mass + 1.0 - w[:, :-1]
        h[:, :-1] = torch.where(denom > 0, mass / denom, torch.ones_like(mass))

    positions = torch.arange(1, num_drafts + 1, dtype=torch.int64, device=draft_tokens.device)
    accepted_len = (positions * (uniforms.to(torch.float64) <= h.to(torch.float64))).amax(-1)
    row = accepted_len.view(-1, 1, 1)
    p_row = target_probs.gather(1, row.expand(-1, 1, vocab_size)).squeeze(1).float()
    q_row = draft_probs.gather(1, row.clamp(max=num_drafts - 1).expand(-1, 1, vocab_size))
    w_prev = torch.cat([torch.ones_like(w[:, :1]), w], dim=1).gather(1, accepted_len.view(-1, 1))
    is_bonus = (accepted_len == num_drafts).unsqueeze(-1)
    residual = torch.where(
        is_bonus, p_row, (w_prev * p_row - q_row.squeeze(1).float()).clamp_min(0.0)
    )
    # Fall back to the target row if rounding leaves the residual without mass.
    residual = torch.where(residual.sum(-1, keepdim=True) > 0, residual, p_row)
    emitted = sample_emitted(residual).long().clamp(0, vocab_size - 1)

    slots = torch.arange(num_drafts + 1, dtype=torch.int64, device=draft_tokens.device)
    drafts = torch.cat([draft_tokens.long(), torch.zeros_like(draft_tokens[:, :1]).long()], dim=1)
    accepted_tokens = torch.where(
        slots < accepted_len[:, None], drafts, torch.full_like(drafts, -1)
    )
    accepted_tokens = torch.where(slots == accepted_len[:, None], emitted[:, None], accepted_tokens)
    return accepted_tokens.to(torch.int32), (accepted_len + 1).to(torch.int32)
