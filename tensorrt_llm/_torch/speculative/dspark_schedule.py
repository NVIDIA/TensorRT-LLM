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
"""DSpark confidence-scheduled verification (arXiv:2607.05147 §5).

``survival[r][j] = prod(conf[r][:j+1])`` ranks every ``(request, position)``
candidate globally; the best ``budget`` are admitted. Survival is
non-increasing along each request's positions, so any global top-k is
automatically a per-request prefix and the allocation only counts admitted
candidates per row. Only the number of verified tokens is decided here, never
acceptance, so scheduling is lossless w.r.t. the target distribution. Ties are
broken by ``(position, request)``, never by value: ranks with
bitwise-different confidences must still choose identical lengths or their
batch shapes (and collectives) diverge.
"""

from dataclasses import dataclass
from typing import Union

import torch

__all__ = [
    "DSparkScheduleConfig",
    "HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT",
    "NATIVE_UNIFORM_VERIFY_OUTPUT",
    "compute_survival",
    "schedule_verify_lens_topk",
]


# Raw logit for confidence rows that carry no measurement: sigmoid(30/T) ~ 1.0
# at any fitted temperature, so an unknown row is treated as "certainly accept"
# -- optimistic by design. Lives here because the worker (writes it) and the
# planner (counts it) cannot import each other.
NEUTRAL_CONFIDENCE_LOGIT = 30.0

# The DSpark forward publishes one of these markers when it cannot return a
# batch-aligned verify-length tensor. The sampler uses the marker to preserve
# the current iteration's policy across overlap with the next iteration.
HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT = "host_policy_windows_snapshot"
NATIVE_UNIFORM_VERIFY_OUTPUT = "native_uniform_verify"


@dataclass(frozen=True)
class DSparkScheduleConfig:
    """Bounds for the per-request verify length.

    Attributes:
        block_size: draft block length ``K``.
        min_verify_len: floor on every request's verify length; must stay >= 1
            (position 0 carries the bonus/anchor token). Budget is allocated
            *above* this floor.
        max_verify_len: cap on every request's verify length; 0 means
            ``block_size``; never exceeds ``block_size``.
        survival_eps: numerical floor (not a tuning knob); candidates below it
            are dropped before ranking.
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
    budget: Union[int, torch.Tensor],
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    """Allocate ``budget`` verify tokens across the batch by global survival rank.

    Args:
        survival: ``[bs, K]`` prefix-survival probabilities from
            :func:`compute_survival`.
        budget: total verify tokens to hand out *above* the per-request floor.
            A 0-d integer tensor is honoured without a host sync (the value
            stays device-resident through the ``rank < budget`` cut), which is
            what lets the whole function run inside a captured graph.
        cfg: bounds; see :class:`DSparkScheduleConfig`.
    Returns:
        ``[bs]`` int32 verify lengths, each in
        ``[min_verify_len, resolved_max_verify_len]``, summing to at most
        ``bs * min_verify_len + budget``.

    Determinism: ranking uses a value-independent tie-break (position before
    request), so equal survivals resolve identically regardless of float noise
    or backend.
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
    if not isinstance(budget, torch.Tensor):
        budget = int(budget)
        if budget <= 0:
            return verify_lens
    if bs == 0 or schedulable <= 0:
        return verify_lens

    # Positions [0, floor) are already granted to every request; only those
    # above compete for budget.
    candidates = survival[:, floor : floor + schedulable].to(torch.float32)
    flat = candidates.reshape(-1)
    eligible = flat >= cfg.survival_eps

    # Sort the tie-break key first, then stable-sort on the scores: the final
    # order is a pure function of (survival, position, request). Ineligible
    # candidates score -1 and are masked by ``& eligible`` below; no
    # ``.item()``/host sync anywhere, so the path stays capture-safe.
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
