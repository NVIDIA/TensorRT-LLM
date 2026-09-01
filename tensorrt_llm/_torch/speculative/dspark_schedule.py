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
import triton
import triton.language as tl

__all__ = [
    "DSparkScheduleConfig",
    "DSparkFusedScheduleError",
    "HOST_POLICY_WINDOWS_SNAPSHOT_OUTPUT",
    "NATIVE_UNIFORM_VERIFY_OUTPUT",
    "compute_survival",
    "schedule_verify_lens_topk_fused_fill",
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


class DSparkFusedScheduleError(RuntimeError):
    """A synchronous optional fused-scheduler dispatch failure.

    Input and graph-contract errors remain ordinary ``ValueError`` exceptions;
    callers may safely disable one fused G only for this wrapped launch error
    and rerun the established tensor implementation.
    """


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


@triton.jit(do_not_specialize=["num_real", "budget"])
def _schedule_topk_rank_kernel(
    survival_ptr,
    output_ptr,
    stride_row,
    stride_position,
    num_real,
    budget,
    survival_eps,
    FLOOR: tl.constexpr,
    COLS: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
    BLOCK_CP: tl.constexpr,
):
    request = tl.program_id(0)
    position = tl.arange(0, BLOCK_COLS)
    candidate_mask = position < COLS
    survival = tl.load(
        survival_ptr + request * stride_row + (position + FLOOR).to(tl.int64) * stride_position,
        mask=candidate_mask,
        other=0.0,
    ).to(tl.float32)
    valid = candidate_mask & (survival >= survival_eps)
    masked_survival = tl.where(valid, survival, float("-inf"))

    rank = tl.zeros([BLOCK_COLS], dtype=tl.int32)
    for peer_start in range(0, NUM_CANDIDATES, BLOCK_CP):
        peer = peer_start + tl.arange(0, BLOCK_CP)
        peer_mask = peer < NUM_CANDIDATES
        peer_request = peer // COLS
        peer_position = peer % COLS
        peer_survival = tl.load(
            survival_ptr
            + peer_request.to(tl.int64) * stride_row
            + (peer_position + FLOOR).to(tl.int64) * stride_position,
            mask=peer_mask,
            other=0.0,
        ).to(tl.float32)
        peer_valid = peer_mask & (peer_request < num_real) & (peer_survival >= survival_eps)
        peer_masked_survival = tl.where(peer_valid, peer_survival, float("-inf"))

        higher = peer_masked_survival[None, :] > masked_survival[:, None]
        equal = peer_masked_survival[None, :] == masked_survival[:, None]
        earlier = (peer_position[None, :] < position[:, None]) | (
            (peer_position[None, :] == position[:, None]) & (peer_request[None, :] < request)
        )
        before = (higher | (equal & earlier)) & peer_valid[None, :]
        rank += tl.sum(before.to(tl.int32), axis=1)

    selected = valid & (rank < budget)
    tl.store(output_ptr + request, tl.sum(selected.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=["num_real", "budget", "pad_len", "graph_num_tokens"])
def _schedule_topk_fill_kernel(
    output_ptr,
    num_rows,
    num_real,
    budget,
    pad_len,
    graph_num_tokens,
    TOKEN_FLOOR: tl.constexpr,
    MAX_TOKEN_LEN: tl.constexpr,
    HAS_CANDIDATES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.arange(0, BLOCK)
    row_mask = rows < num_rows
    is_real = row_mask & (rows < num_real)
    selected_extra = tl.load(
        output_ptr + rows,
        mask=is_real & (budget > 0) & HAS_CANDIDATES,
        other=0,
    ).to(tl.int32)
    token_len = tl.where(is_real, TOKEN_FLOOR + selected_extra, pad_len)

    # Reproduce fill_bucket_device's exact-pad real-row phase. The explicit
    # policy budget controls only the confidence-ranked allocation. Any
    # already-paid graph remainder is filled round-robin in request-index
    # order; fusion must not promote it into another confidence budget.
    spare = graph_num_tokens - tl.sum(tl.where(row_mask, token_len, 0), axis=0)
    for _ in range(MAX_TOKEN_LEN - 1):
        headroom = is_real & (token_len < MAX_TOKEN_LEN)
        in_cycle = tl.cumsum(headroom.to(tl.int32), axis=0) <= spare
        grant = headroom & in_cycle
        token_len += grant.to(tl.int32)
        spare -= tl.sum(grant.to(tl.int32), axis=0)
    tl.store(output_ptr + rows, token_len, mask=row_mask)


def _schedule_verify_lens_topk_fused_fill_triton(
    *,
    survival: torch.Tensor,
    budget: int,
    num_real: int,
    pad_len: int,
    graph_num_tokens: int,
    cfg: DSparkScheduleConfig,
) -> torch.Tensor:
    """Triton implementation of policy top-k plus exact round-robin fill."""
    num_rows = survival.shape[0]
    if num_rows == 0:
        return torch.empty(0, dtype=torch.int32, device=survival.device)
    if num_rows > 256:
        raise ValueError(f"fused top-k scheduler supports at most 256 rows, got {num_rows}")
    schedulable = int(cfg.schedulable_per_request)
    output = torch.empty(num_rows, dtype=torch.int32, device=survival.device)

    # Specialize rank work by the captured graph row count, never by the
    # step-varying real-row count. This keeps one compiled kernel per all-G
    # graph key; peer rows past num_real are masked inside the kernel.
    num_candidates = num_rows * schedulable
    ranked = budget > 0 and num_real > 0 and num_candidates > 0
    if ranked:
        block_cols = triton.next_power_of_2(schedulable)
        block_cp = 128
        try:
            _schedule_topk_rank_kernel[(num_real,)](
                survival,
                output,
                survival.stride(0),
                survival.stride(1),
                num_real,
                budget,
                float(cfg.survival_eps),
                FLOOR=int(cfg.min_verify_len),
                COLS=schedulable,
                NUM_CANDIDATES=num_candidates,
                BLOCK_COLS=block_cols,
                BLOCK_CP=block_cp,
            )
        except Exception as exc:
            raise DSparkFusedScheduleError("DSpark fused top-k rank launch failed") from exc
    finalize_block = triton.next_power_of_2(num_rows)
    try:
        _schedule_topk_fill_kernel[(1,)](
            output,
            num_rows,
            num_real,
            budget,
            pad_len,
            graph_num_tokens,
            TOKEN_FLOOR=int(cfg.min_verify_len) + 1,
            MAX_TOKEN_LEN=int(cfg.resolved_max_verify_len) + 1,
            HAS_CANDIDATES=schedulable > 0,
            BLOCK=finalize_block,
        )
    except Exception as exc:
        raise DSparkFusedScheduleError("DSpark fused round-robin fill launch failed") from exc
    return output


def schedule_verify_lens_topk_fused_fill(
    *,
    survival: torch.Tensor,
    budget: int,
    num_real: int,
    pad_len: int,
    cfg: DSparkScheduleConfig,
    graph_num_tokens: int,
) -> torch.Tensor:
    """Fuse the established top-k schedule and exact round-robin bucket fill.

    The explicit ``budget`` retains exactly the semantics of
    :func:`schedule_verify_lens_topk`. Pad rows receive exactly ``pad_len``
    tokens; any graph-token remainder is granted to real rows in index-order
    round-robin cycles, exactly as ``fill_bucket_device(..., pad_fill=...)``.
    CUDA uses two deterministic Triton launches. CPU executes the established
    tensor operations as the functional oracle.
    """
    if survival.dim() != 2:
        raise ValueError(f"survival must be [bs, K], got shape {tuple(survival.shape)}")
    num_rows, num_positions = survival.shape
    if num_positions != cfg.block_size:
        raise ValueError(
            f"survival has {num_positions} positions but block_size is {cfg.block_size}"
        )
    budget = int(budget)
    num_real = int(num_real)
    pad_len = int(pad_len)
    if cfg.survival_eps <= 0.0:
        raise ValueError(
            "fused top-k scheduling requires survival_eps > 0 so zeroed pad "
            "rows cannot compete for the legacy policy budget"
        )
    if not 0 <= num_real <= num_rows:
        raise ValueError(f"num_real must be in [0, {num_rows}], got {num_real}")
    max_token_len = int(cfg.resolved_max_verify_len) + 1
    if not 1 <= pad_len <= max_token_len:
        raise ValueError(f"pad_len must be in [1, {max_token_len}], got {pad_len}")

    graph_num_tokens = int(graph_num_tokens)
    token_floor = int(cfg.min_verify_len) + 1
    pad_tokens = (num_rows - num_real) * pad_len
    minimum_tokens = num_real * token_floor + pad_tokens
    maximum_tokens = num_real * max_token_len + pad_tokens
    scheduled_capacity = min(max(budget, 0), num_real * int(cfg.schedulable_per_request))
    if not minimum_tokens + scheduled_capacity <= graph_num_tokens <= maximum_tokens:
        raise ValueError(
            "fused schedule cannot realize the selected CUDA graph without "
            "changing policy semantics: "
            f"expected [{minimum_tokens + scheduled_capacity}, {maximum_tokens}] "
            f"tokens, graph requires {graph_num_tokens}"
        )

    if survival.is_cuda:
        return _schedule_verify_lens_topk_fused_fill_triton(
            survival=survival,
            budget=budget,
            num_real=num_real,
            pad_len=pad_len,
            graph_num_tokens=graph_num_tokens,
            cfg=cfg,
        )

    rows = torch.arange(num_rows, device=survival.device)
    real_survival = torch.where(
        (rows < num_real).unsqueeze(1),
        survival,
        torch.full_like(survival, -1.0),
    )
    scheduled = schedule_verify_lens_topk(
        survival=real_survival,
        budget=budget,
        cfg=cfg,
    )
    token_lens = torch.where(
        rows < num_real,
        scheduled + 1,
        torch.full_like(scheduled, pad_len),
    )
    spare = graph_num_tokens - int(token_lens.sum())
    for _ in range(max_token_len - 1):
        if spare <= 0:
            break
        headroom = (rows < num_real) & (token_lens < max_token_len)
        grant = headroom & (torch.cumsum(headroom.to(torch.int64), dim=0) <= spare)
        token_lens += grant.to(torch.int32)
        spare -= int(grant.sum())
    if spare != 0:
        raise ValueError(f"fused schedule left {spare} graph tokens after round-robin fill")
    return token_lens
