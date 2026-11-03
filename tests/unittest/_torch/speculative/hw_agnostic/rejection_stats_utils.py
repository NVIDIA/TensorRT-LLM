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
"""Statistical helpers for the rejection-verification tests: toy prefix-conditioned models,
a token-level reference verifier and a chi-square check.

A prefix table holds one next-token distribution per prefix of length 0..block, rows laid
out by prefix length and then base-``vocab``.
"""

import math

import torch


def random_prefix_table(
    generator: torch.Generator,
    vocab: int,
    block: int,
    sharpness: float = 1.0,
    zero_frac: float = 0.0,
) -> torch.Tensor:
    """Random next-token distributions for every prefix; ``zero_frac`` of entries get no mass."""
    num_prefixes = sum(vocab**length for length in range(block + 1))
    logits = sharpness * torch.randn(num_prefixes, vocab, generator=generator)
    zero = torch.rand(num_prefixes, vocab, generator=generator) < zero_frac
    zero[:, 0] = False  # keep every row nonempty
    return torch.softmax(logits.masked_fill(zero, float("-inf")).double(), -1)


def prefix_ids(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    """``[N, m]`` prefixes -> their prefix-table row ids."""
    length = tokens.shape[1]
    ids = torch.zeros(tokens.shape[0], dtype=torch.int64)
    for m in range(length):
        ids = ids * vocab + tokens[:, m]
    return ids + sum(vocab**m for m in range(length))


def exact_sequence_probs(table: torch.Tensor, length: int) -> torch.Tensor:
    """Exact probability of every length-``length`` sequence under ``table``, in base-vocab order."""
    vocab = table.shape[-1]
    grid = torch.cartesian_prod(*[torch.arange(vocab)] * length)
    prob = torch.ones(grid.shape[0], dtype=torch.float64)
    for m in range(length):
        rows = table[prefix_ids(grid[:, :m], vocab)]
        prob = prob * rows.gather(-1, grid[:, m : m + 1]).squeeze(-1)
    return prob


def complete_and_count(
    table: torch.Tensor,
    accepted_tokens: torch.Tensor,
    num_accepted: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Emitted tokens, completed by sampling ``table`` to the block length, counted per sequence."""
    vocab = table.shape[-1]
    num, length = accepted_tokens.shape
    sequences = torch.zeros(num, length, dtype=torch.int64)
    for m in range(length):
        cont = torch.multinomial(table[prefix_ids(sequences[:, :m], vocab)], 1, generator=generator)
        keep = m < num_accepted.long()
        sequences[:, m] = torch.where(keep, accepted_tokens[:, m].long(), cont.squeeze(-1))
    ids = torch.zeros(num, dtype=torch.int64)
    for m in range(length):
        ids = ids * vocab + sequences[:, m]
    return torch.bincount(ids, minlength=vocab**length)


def chi_square(counts: torch.Tensor, probs: torch.Tensor) -> tuple[float, float]:
    """Pearson statistic over cells with expected count >= 5 (the rest pooled) vs its 1e-4 critical value."""
    assert counts[probs == 0].sum() == 0, "observations in zero-probability cells"
    total = counts.sum().item()
    expected = probs * total
    big = expected >= 5
    observed = torch.cat([counts[big].double(), counts[~big].double().sum().view(1)])
    expected = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = expected > 0
    stat = (((observed - expected) ** 2)[keep] / expected[keep]).sum().item()
    dof = int(keep.sum()) - 1
    z = 3.719  # one-sided 1e-4 normal quantile; Wilson-Hilferty approximation below
    return stat, dof * (1 - 2 / (9 * dof) + z * math.sqrt(2 / (9 * dof))) ** 3


def token_level_verify(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniforms: torch.Tensor,
    resample_uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level rejection sampling (Leviathan et al., 2023) with ``block_verify``'s layout.

    Each draft is accepted with probability min(1, p/q) in order; the first rejection
    resamples from max(p - q, 0) and full acceptance draws the bonus from the last target
    row, both by inverse CDF on ``resample_uniforms``. Slots past the emitted token are -1.
    """
    num, num_drafts = draft_tokens.shape
    vocab = target_probs.shape[-1]
    tokens_x = draft_tokens.long().unsqueeze(-1)
    ratio = (
        target_probs[:, :num_drafts].gather(-1, tokens_x).squeeze(-1).double()
        / draft_probs.gather(-1, tokens_x).squeeze(-1).double()
    )
    length = (uniforms <= ratio).long().cumprod(-1).sum(-1)
    rows = torch.arange(num)
    p_row = target_probs[rows, length]
    q_row = draft_probs[rows, length.clamp(max=num_drafts - 1)]
    weights = torch.where((length == num_drafts)[:, None], p_row, (p_row - q_row).clamp_min(0))
    cdf = weights.double().cumsum(-1)
    emitted = torch.searchsorted(cdf, resample_uniforms[:, None] * cdf[:, -1:], right=True)
    slots = torch.arange(num_drafts + 1)
    drafts = torch.cat([draft_tokens.long(), torch.zeros(num, 1, dtype=torch.int64)], 1)
    out = torch.full((num, num_drafts + 1), -1, dtype=torch.int64)
    out = torch.where(slots < length[:, None], drafts, out)
    out = torch.where(slots == length[:, None], emitted.clamp(max=vocab - 1), out)
    return out, length + 1
