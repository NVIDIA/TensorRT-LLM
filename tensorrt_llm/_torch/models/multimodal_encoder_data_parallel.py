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

from contextlib import contextmanager
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import Callable, Iterator, Sequence, TypeVar

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from tensorrt_llm._torch.distributed import Distributed, allgather
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.mapping import Mapping


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
    """The rank assigned to one encoder item."""

    item: EncoderDpItem
    rank: int


@dataclass(frozen=True)
class EncoderDpPlan:
    """A deterministic placement of ordered encoder items onto DP ranks."""

    placements: tuple[EncoderDpPlacement, ...]
    rank_input_token_counts: tuple[int, ...]
    rank_output_row_counts: tuple[int, ...]
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
    rank_output_row_counts = [0] * num_ranks
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
        rank_output_row_counts[target_rank] += item.output_row_count

    placements: list[EncoderDpPlacement] = []
    for item in ordered_items:
        placements.append(
            EncoderDpPlacement(
                item=item,
                rank=item_ranks[item.ordinal],
            )
        )

    return EncoderDpPlan(
        placements=tuple(placements),
        rank_input_token_counts=tuple(rank_input_token_counts),
        rank_output_row_counts=tuple(rank_output_row_counts),
        total_output_rows=sum(rank_output_row_counts),
    )


class EncoderDpError(RuntimeError):
    """A coordinated Python failure, propagated before the next GPU collective."""


@contextmanager
def encoder_dp_error_guard(any_rank_failed: Callable[[bool], bool]) -> Iterator[None]:
    """Propagate rank-local Python failures before the next GPU collective.

    The callback exchanges CPU status and must not synchronize encoder CUDA
    work. It must return true on every rank if any rank reports a failure.
    """
    local_error: Exception | None = None
    try:
        yield
    except Exception as error:
        # Every rank must reach the status exchange even after a local failure.
        local_error = error

    if any_rank_failed(local_error is not None):
        if local_error is not None:
            raise EncoderDpError("Multimodal encoder data-parallel rank failed.") from local_error
        raise EncoderDpError("Multimodal encoder data-parallel peer rank failed.")


_LocalInputsT = TypeVar("_LocalInputsT")


def execute_encoder_dp_items(
    items: Sequence[EncoderDpItem],
    *,
    rank: int,
    num_ranks: int,
    prepare_local_inputs: Callable[[Sequence[EncoderDpItem]], _LocalInputsT],
    encode_local_inputs: Callable[[_LocalInputsT], torch.Tensor],
    any_rank_failed: Callable[[bool], bool],
    allgather: Callable[[torch.Tensor, list[int]], torch.Tensor],
    output_dim: int,
    output_dtype: torch.dtype,
    output_device: torch.device,
) -> torch.Tensor:
    """Place, execute, and reconstruct atomic multimodal encoder items.

    Every rank independently derives the same plan. The callbacks see only the
    items assigned to the local rank, in global output order. A CPU status
    exchange propagates local preparation and encoder failures before gathering
    compact per-rank outputs. ``allgather`` concatenates these ``[rows, dim]``
    tensors in rank order using the supplied per-rank row counts, including zero
    rows on empty ranks.

    The return value is one dense tensor with item row blocks in ordinal order.
    The full-request path consumes it directly; runtime item scheduling splits
    it by ``output_row_count`` and feeds the existing per-item request-state
    ``record`` path.
    """
    with encoder_dp_error_guard(any_rank_failed):
        plan = plan_encoder_dp_items(items, num_ranks)
        local_placements = plan.local_placements(rank)
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
            local_output = local_output.to(device=output_device, dtype=output_dtype).contiguous()
        else:
            local_output = torch.empty((0, output_dim), dtype=output_dtype, device=output_device)

    if plan.total_output_rows == 0:
        return local_output

    gathered = allgather(local_output, list(plan.rank_output_row_counts))
    if all(left.rank <= right.rank for left, right in pairwise(plan.placements)):
        # Rank order already matches prompt order; avoid another full-size copy.
        return gathered
    rank_row_starts = list(accumulate(plan.rank_output_row_counts, initial=0))
    ordered_outputs = []
    for placement in plan.placements:
        row_start = rank_row_starts[placement.rank]
        row_end = row_start + placement.item.output_row_count
        ordered_outputs.append(gathered[row_start:row_end])
        rank_row_starts[placement.rank] = row_end
    return torch.cat(ordered_outputs, dim=0)


def iter_encoder_rank_inputs(
    encoder_inputs: Sequence[tuple[MultimodalParams, list[int], str]],
    *,
    mapping: Mapping,
    any_rank_failed: Callable[[bool], bool],
) -> Iterator[tuple[int, list[tuple[MultimodalParams, list[int], str]]]]:
    """Share one ADP owner's prepared encoder batch at a time for encoder TP.

    Only CPU descriptors and metadata use object collectives. CUDA tensor
    payloads stay on the GPU and use the existing variable-size allgather.
    Processing owners in order bounds temporary GPU storage by one owner's
    encoder budget, rather than the sum of all ranks' budgets.
    """
    with encoder_dp_error_guard(any_rank_failed):
        leaves, spec = tree_flatten(
            [
                (param.multimodal_data, param.mm_item_order, lengths, modality)
                for param, lengths, modality in encoder_inputs
            ]
        )
        gpu_leaves: dict[torch.dtype, list[tuple[int, torch.Tensor]]] = {}
        descriptors = []
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                gpu_leaves.setdefault(leaf.dtype, []).append((index, leaf))
                descriptors.append((index, tuple(leaf.shape), leaf.dtype))
                leaves[index] = None
    rank_metadata = Distributed.get(mapping).tp_allgather((leaves, spec, descriptors))

    for owner, (owner_leaves, owner_spec, owner_descriptors) in enumerate(rank_metadata):
        if not owner_descriptors and not tree_unflatten(owner_leaves, owner_spec):
            continue
        shared_leaves = list(owner_leaves)
        by_dtype: dict[torch.dtype, list[tuple[int, tuple[int, ...]]]] = {}
        for index, shape, dtype in owner_descriptors:
            by_dtype.setdefault(dtype, []).append((index, shape))
        for dtype, entries in by_dtype.items():
            with encoder_dp_error_guard(any_rank_failed):
                total_elements = sum(torch.Size(shape).numel() for _, shape in entries)
                if mapping.tp_rank == owner:
                    payload = torch.cat([tensor.reshape(-1) for _, tensor in gpu_leaves[dtype]])
                else:
                    payload = torch.empty(0, dtype=dtype, device="cuda")
            if total_elements:
                payload = allgather(
                    payload,
                    mapping,
                    dim=0,
                    sizes=[
                        total_elements if rank == owner else 0 for rank in range(mapping.tp_size)
                    ],
                )
            offset = 0
            for index, shape in entries:
                numel = torch.Size(shape).numel()
                shared_leaves[index] = payload[offset : offset + numel].view(shape)
                offset += numel
        yield (
            owner,
            [
                (MultimodalParams(multimodal_data=data, mm_item_order=order), lengths, modality)
                for data, order, lengths, modality in tree_unflatten(shared_leaves, owner_spec)
            ],
        )
