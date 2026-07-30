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
"""DSpark confidence-scheduled verification: survival + per-request allocation.

Given the draft's calibrated per-position acceptance probabilities, decide how
many of the proposed tokens are worth sending to the target for verification.

The algorithm (DSpark, arXiv:2607.05147 §5; cross-checked against SGLang's
``dspark_schedule.py`` and vLLM PR #47808's ``adaptive_verification.py``):

  1. ``survival[r][j] = prod(conf[r][:j+1])`` -- the probability that draft
     position ``j`` of request ``r`` is reached *and* accepted. The confidence
     head predicts a *conditional* acceptance probability, which is exactly what
     makes this cumulative product the right prefix statistic.
  2. Rank every ``(request, position)`` candidate by survival, globally across
     the batch, and admit the best ``budget`` of them.

Step 2 needs no explicit prefix constraint: ``survival`` is non-increasing along
each request's positions, so any global top-k is automatically a per-request
*prefix*. That is why the allocation only has to *count* admitted candidates per
row rather than track which ones they were.

Nothing here decides acceptance -- only how many tokens get verified -- so
scheduling is lossless with respect to the target distribution.

Determinism note: ties are broken by ``(position, request)``, never by value.
Two TP ranks that compute bitwise-different confidences must still choose the
same verify lengths, or their batch shapes (and therefore their collectives)
diverge. See :func:`schedule_verify_lens_topk`.
"""

from dataclasses import dataclass

import torch

__all__ = [
    "DSparkScheduleConfig",
    "compute_survival",
    "schedule_verify_lens_topk",
]


@dataclass(frozen=True)
class DSparkScheduleConfig:
    """Bounds for the per-request verify length.

    Attributes:
        block_size: draft block length ``K`` (the number of speculative tokens
            the draft proposes per step).
        min_verify_len: floor on every request's verify length. Must stay >= 1:
            position 0 carries the bonus/anchor token, and a request that
            verifies nothing makes no progress at all. Budget is allocated
            *above* this floor.
        max_verify_len: cap on every request's verify length; 0 means
            ``block_size``. Never exceeds ``block_size`` -- a request cannot
            verify more tokens than the draft proposed.
        survival_eps: candidates whose survival falls below this are dropped
            before ranking. Purely a numerical floor, not a tuning knob: it
            keeps hopeless positions out of the ordering.
    """

    block_size: int
    min_verify_len: int = 1
    max_verify_len: int = 0
    survival_eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.min_verify_len < 1:
            raise ValueError(
                f"min_verify_len must be >= 1 (every request verifies its anchor "
                f"token), got {self.min_verify_len}"
            )
        if self.resolved_max_verify_len < self.min_verify_len:
            raise ValueError(
                f"max_verify_len ({self.resolved_max_verify_len}) < min_verify_len "
                f"({self.min_verify_len})"
            )
        if not 0.0 <= self.survival_eps < 1.0:
            raise ValueError(f"survival_eps must be in [0, 1), got {self.survival_eps}")

    @property
    def resolved_max_verify_len(self) -> int:
        """``max_verify_len`` with 0 meaning "the whole block", clamped to ``block_size``."""
        cap = self.max_verify_len or self.block_size
        return min(int(cap), int(self.block_size))

    @property
    def schedulable_per_request(self) -> int:
        """Positions a single request can win from the budget, above the floor."""
        return self.resolved_max_verify_len - self.min_verify_len


def compute_survival(confidence: torch.Tensor) -> torch.Tensor:
    """``[bs, K]`` conditional acceptance probabilities -> prefix survival.

    ``out[r][j] = prod_{i <= j} confidence[r][i]``, computed in fp32. The result
    is non-increasing along ``j``, which :func:`schedule_verify_lens_topk` relies
    on to keep its allocation prefix-shaped.
    """
    if confidence.dim() != 2:
        raise ValueError(f"confidence must be [bs, K], got shape {tuple(confidence.shape)}")
    return torch.cumprod(confidence.to(torch.float32), dim=1)


def schedule_verify_lens_topk(
    *,
    survival: torch.Tensor,
    budget: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    """Allocate ``budget`` verify tokens across the batch by global survival rank.

    Args:
        survival: ``[bs, K]`` prefix-survival probabilities from
            :func:`compute_survival`.
        budget: total verify tokens to hand out *above* the per-request floor.
        cfg: bounds; see :class:`DSparkScheduleConfig`.
    Returns:
        ``[bs]`` int32 verify lengths, each in
        ``[min_verify_len, resolved_max_verify_len]``, summing to at most
        ``bs * min_verify_len + budget``.

    Determinism: ranking uses a value-independent tie-break, so equal survivals
    always resolve the same way regardless of float noise or backend. Positions
    are ordered before requests, which biases ties toward *earlier* positions --
    the ones every downstream prefix depends on.
    """
    if survival.dim() != 2:
        raise ValueError(f"survival must be [bs, K], got shape {tuple(survival.shape)}")
    bs, num_positions = survival.shape
    if num_positions != cfg.block_size:
        raise ValueError(
            f"survival has {num_positions} positions but block_size is {cfg.block_size}"
        )

    device = survival.device
    floor = int(cfg.min_verify_len)
    schedulable = int(cfg.schedulable_per_request)
    verify_lens = torch.full((bs,), floor, dtype=torch.int32, device=device)
    budget = int(budget)
    if bs == 0 or schedulable <= 0 or budget <= 0:
        return verify_lens

    # Only positions above the floor are up for grabs: positions [0, floor) are
    # already granted to every request, so scoring them would let a request win
    # budget for something it is getting for free.
    candidates = survival[:, floor : floor + schedulable].to(torch.float32)
    flat = candidates.reshape(-1)
    eligible = flat >= cfg.survival_eps

    # Rank by survival, then break ties deterministically. Sorting the tie-break
    # key first and using a *stable* sort on the values makes the final order a
    # pure function of (survival values, position, request) -- no dependence on
    # the sort implementation's handling of equal keys.
    #
    # Ineligible candidates are scored -1 so they sort below every eligible one;
    # the ``& eligible`` below then drops them even when ``budget`` is larger
    # than the eligible count. That keeps the whole path free of any device->host
    # sync (no ``.item()``), so it stays usable inside a captured graph.
    scores = torch.where(eligible, flat, torch.full_like(flat, -1.0))
    tie_break = torch.arange(schedulable, device=device).repeat(bs) * bs + torch.arange(
        bs, device=device
    ).repeat_interleave(schedulable)
    order = torch.argsort(tie_break, stable=True)
    order = order[torch.argsort(scores[order], descending=True, stable=True)]

    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.numel(), device=device)
    selected = (rank < budget) & eligible

    # survival is non-increasing along positions, so the chosen set for a request
    # is necessarily a prefix -- counting per row recovers the full allocation.
    granted = selected.view(bs, schedulable).sum(dim=1).to(torch.int32)
    verify_lens += granted
    return torch.clamp(verify_lens, min=floor, max=cfg.resolved_max_verify_len)
