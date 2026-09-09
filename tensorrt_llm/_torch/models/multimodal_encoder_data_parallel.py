# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-independent multimodal encoder data-parallel execution.

Runtime item scheduling owns which cache-miss items run and their lifecycle.
This module owns deterministic rank placement, rank-local execution, and
ordered distributed reconstruction. Placement uses physical encoder input
tokens when item metadata provides them; output rows are tracked separately
for reconstruction.
"""

from dataclasses import dataclass
from typing import Callable, Sequence, TypeVar

import torch


@dataclass(frozen=True)
class EncoderDpItem:
    """An ordered atomic item passed to multimodal encoder data parallelism.

    The caller owns the item payload and lifecycle. The DP layer uses
    ``input_token_count`` as its initial placement-cost signal and
    ``output_row_count`` only to reconstruct the result in ordinal order.
    """

    ordinal: int
    input_token_count: int
    output_row_count: int


@dataclass(frozen=True)
class EncoderDpPlacement:
    """The rank and output location assigned to one encoder item."""

    item: EncoderDpItem
    rank: int
    global_row_start: int


@dataclass(frozen=True)
class EncoderDpPlan:
    """A deterministic placement of ordered encoder items onto DP ranks."""

    placements: tuple[EncoderDpPlacement, ...]
    rank_input_token_counts: tuple[int, ...]
    total_output_rows: int

    def local_placements(self, rank: int) -> tuple[EncoderDpPlacement, ...]:
        """Return placements for ``rank`` in global output order."""
        if rank < 0 or rank >= len(self.rank_input_token_counts):
            raise ValueError(
                f"Encoder DP rank must be in [0, {len(self.rank_input_token_counts)}), got {rank}."
            )
        return tuple(placement for placement in self.placements if placement.rank == rank)


def plan_encoder_dp_items(
    items: Sequence[EncoderDpItem],
    num_ranks: int,
) -> EncoderDpPlan:
    """Place atomic encoder items using deterministic LPT scheduling.

    Items are assigned largest-input-first to the rank with the smallest
    accumulated input-token count. Placements are returned in ordinal order so
    caller-provided preparation and encoder callbacks have one stable ordering
    contract independent of request or scheduler representation.
    """
    if num_ranks <= 0:
        raise ValueError(f"Encoder DP requires at least one rank, got {num_ranks}.")

    ordered_items = sorted(items, key=lambda item: item.ordinal)
    ordinals = [item.ordinal for item in ordered_items]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("Encoder DP item ordinals must be unique.")
    for item in ordered_items:
        if item.ordinal < 0:
            raise ValueError(f"Encoder DP item ordinal must be non-negative, got {item.ordinal}.")
        if item.input_token_count <= 0:
            raise ValueError(
                "Encoder DP item input_token_count must be positive, "
                f"got {item.input_token_count} for ordinal {item.ordinal}."
            )
        if item.output_row_count <= 0:
            raise ValueError(
                "Encoder DP item output_row_count must be positive, "
                f"got {item.output_row_count} for ordinal {item.ordinal}."
            )

    rank_input_token_counts = [0] * num_ranks
    item_ranks: dict[int, int] = {}
    for item in sorted(
        ordered_items,
        key=lambda candidate: (-candidate.input_token_count, candidate.ordinal),
    ):
        target_rank = min(
            range(num_ranks),
            key=lambda rank: (rank_input_token_counts[rank], rank),
        )
        item_ranks[item.ordinal] = target_rank
        rank_input_token_counts[target_rank] += item.input_token_count

    placements: list[EncoderDpPlacement] = []
    global_row_start = 0
    for item in ordered_items:
        placements.append(
            EncoderDpPlacement(
                item=item,
                rank=item_ranks[item.ordinal],
                global_row_start=global_row_start,
            )
        )
        global_row_start += item.output_row_count

    return EncoderDpPlan(
        placements=tuple(placements),
        rank_input_token_counts=tuple(rank_input_token_counts),
        total_output_rows=global_row_start,
    )


_LocalInputsT = TypeVar("_LocalInputsT")


def execute_encoder_dp_items(
    items: Sequence[EncoderDpItem],
    *,
    rank: int,
    num_ranks: int,
    prepare_local_inputs: Callable[[Sequence[EncoderDpItem]], _LocalInputsT],
    encode_local_inputs: Callable[[_LocalInputsT], torch.Tensor],
    allreduce: Callable[[torch.Tensor], torch.Tensor],
    output_dim: int,
    output_dtype: torch.dtype,
    output_device: torch.device,
) -> torch.Tensor:
    """Place, execute, and reconstruct atomic multimodal encoder items.

    Every rank independently derives the same plan. The callbacks see only the
    items assigned to the local rank, in global output order. A status
    collective precedes the output collective so a local preparation, encoder,
    validation, allocation, or copy failure cannot strand peer ranks.

    The return value is one dense tensor with item row blocks in ordinal order.
    The full-request path consumes it directly; runtime item scheduling splits
    it by ``output_row_count`` and feeds the existing per-item request-state
    ``record`` path.
    """
    local_error: Exception | None = None
    local_contribution: torch.Tensor | None = None
    plan: EncoderDpPlan | None = None

    try:
        plan = plan_encoder_dp_items(items, num_ranks)
        local_placements = plan.local_placements(rank)
        local_output: torch.Tensor | None = None
        if local_placements:
            local_items = [placement.item for placement in local_placements]
            local_inputs = prepare_local_inputs(local_items)
            local_output = encode_local_inputs(local_inputs)

            expected_local_rows = sum(item.output_row_count for item in local_items)
            if local_output.ndim != 2 or local_output.shape != (
                expected_local_rows,
                output_dim,
            ):
                raise ValueError(
                    "Multimodal encoder output shape does not match its local DP assignment: "
                    f"expected ({expected_local_rows}, {output_dim}), "
                    f"got {tuple(local_output.shape)}."
                )

        local_contribution = torch.zeros(
            (plan.total_output_rows, output_dim),
            dtype=output_dtype,
            device=output_device,
        )
        local_row_start = 0
        if local_output is not None:
            for placement in local_placements:
                row_count = placement.item.output_row_count
                local_contribution[
                    placement.global_row_start : placement.global_row_start + row_count
                ].copy_(local_output[local_row_start : local_row_start + row_count])
                local_row_start += row_count
    except Exception as error:
        # Collective safety requires converting every local failure into the
        # same status collective before any rank enters the output collective.
        local_error = error

    error_flag = torch.tensor(
        [local_error is not None],
        dtype=torch.int32,
        device=output_device,
    )
    any_error = allreduce(error_flag)
    if bool(any_error.item()):
        if local_error is not None:
            raise RuntimeError("Multimodal encoder data-parallel rank failed.") from local_error
        raise RuntimeError("Multimodal encoder data-parallel peer rank failed.")

    if local_contribution is None or plan is None:
        raise RuntimeError("Multimodal encoder data-parallel execution produced no local plan.")
    if plan.total_output_rows == 0:
        return local_contribution
    return allreduce(local_contribution)
