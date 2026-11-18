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

import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from itertools import accumulate
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from tensorrt_llm._torch.distributed import Distributed, allgather
from tensorrt_llm._utils import mpi_disabled
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams
from tensorrt_llm.mapping import Mapping

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig


@dataclass(frozen=True)
class EncoderParallelLayout:
    """Where this rank's multimodal encoder runs inside the LLM TP group.

    Without ``encoder_data_parallel_size``, the encoder keeps the LLM mapping.
    An explicit size splits the LLM TP group into ``dp_size`` replicas of
    contiguous ranks; ``mapping`` is then this rank's replica.
    """

    mapping: Mapping
    dp_size: int = 1
    dp_rank: int = 0
    distributes_items: bool = False
    """One batch of items is split between the encoder replicas."""
    collective: bool = False
    """Attention-DP ranks share one encoder TP pass, so every rank must take part."""

    @property
    def tp_size(self) -> int:
        return self.mapping.tp_size

    @classmethod
    def from_config(cls, model_config: "ModelConfig") -> "EncoderParallelLayout":
        mapping = model_config.mapping
        mm_config = model_config.multimodal_config
        dp_size = None if mm_config is None else mm_config.encoder_data_parallel_size
        if dp_size is None:
            return cls(mapping)
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise NotImplementedError(
                "Explicit encoder parallelism requires pipeline parallel size 1 "
                "and context parallel size 1."
            )
        if mpi_disabled() and 1 < dp_size < mapping.tp_size:
            # The device mesh has one TP group, so a replica would get the whole group.
            raise NotImplementedError(
                "Encoder TP inside encoder DP replicas needs MPI; with the Ray "
                "orchestrator, use encoder_data_parallel_size 1 or the TP size."
            )
        encoder_mapping = mapping.split_tp(dp_size)
        shared_tp = mapping.enable_attention_dp and dp_size < mapping.tp_size
        return cls(
            mapping=encoder_mapping,
            dp_size=dp_size,
            dp_rank=mapping.tp_rank // encoder_mapping.tp_size,
            distributes_items=dp_size > 1 and (shared_tp or not mapping.enable_attention_dp),
            collective=shared_tp,
        )


def plan_encoder_dp_items(costs: Sequence[int], num_ranks: int) -> tuple[int, ...]:
    """Place items on ranks with deterministic LPT scheduling.

    Items go largest cost first to the rank with the smallest total cost.
    Ties go to the lower item index and then the lower rank. Returns the rank
    of each item, in item order.
    """
    rank_costs = [0] * num_ranks
    ranks = [0] * len(costs)
    for index in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min(range(num_ranks), key=lambda r: (rank_costs[r], r))
        ranks[index] = rank
        rank_costs[rank] += costs[index]
    return tuple(ranks)


class MultimodalEncoderContractError(ValueError):
    """A request-local MM encoder input or output contract violation."""


class EncoderDpError(RuntimeError):
    """A system error in shared encoder work. Every rank raises the same one."""


class EncoderDpStatus(IntEnum):
    """The status code each rank shares before the next encoder collective."""

    OK = 0
    REQUEST_ERROR = 1
    """Only the requests of the failed batch fail."""
    SYSTEM_ERROR = 2
    """Every rank raises the same ``EncoderDpError``."""

    @classmethod
    def of(cls, error: Exception | None) -> "EncoderDpStatus":
        """The status of this rank's error, or ``OK`` without one."""
        if error is None:
            return cls.OK
        if isinstance(error, MultimodalEncoderContractError):
            return cls.REQUEST_ERROR
        return cls.SYSTEM_ERROR


ExchangeStatus = Callable[[list[int]], Sequence[Sequence[int]]]
"""Gathers a small int vector from every TP rank, on the CPU. Row ``r`` is rank ``r``'s."""


def raise_shared_system_error(statuses: Sequence[int], local_error: Exception | None) -> None:
    """Raise the same ``EncoderDpError`` on every rank if any rank had a system error."""
    failed = [
        rank for rank, status in enumerate(statuses) if status == EncoderDpStatus.SYSTEM_ERROR
    ]
    if failed:
        raise EncoderDpError(f"Multimodal encoder failed on TP rank(s) {failed}.") from local_error


def release_error_frames(error: Exception) -> Exception:
    """Clear the local variables that ``error`` and its causes keep alive, and return it.

    An error kept until the end of an encoder step would otherwise keep its
    failed batch's buffers alive while later batches run. Line numbers stay.
    """
    pending: list[BaseException | None] = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        traceback.clear_frames(current.__traceback__)
        pending += (current.__cause__, current.__context__)
    return error


@contextmanager
def encoder_dp_error_guard(exchange_status: ExchangeStatus) -> Iterator[None]:
    """Share rank-local failures before the next collective.

    If any rank had a system error, every rank raises the same
    ``EncoderDpError``. Otherwise, if any rank had a request error, every rank
    raises a ``MultimodalEncoderContractError`` (its own one, or one that
    names the failed ranks), and the caller decides whose requests fail. An
    ``EncoderDpError`` from an inner guard is re-raised as is, because every
    rank already raises it. The exchange must not wait for encoder CUDA work.
    """
    local_error: Exception | None = None
    try:
        yield
    except EncoderDpError:
        raise
    except Exception as error:
        # Every rank must reach the status exchange even after a local failure.
        local_error = error

    statuses = [int(row[0]) for row in exchange_status([EncoderDpStatus.of(local_error)])]
    raise_shared_system_error(statuses, local_error)
    if local_error is not None:
        raise local_error
    failed = [rank for rank, status in enumerate(statuses) if status]
    if failed:
        raise MultimodalEncoderContractError(
            f"Multimodal encoder input failed on TP rank(s) {failed}."
        )


def execute_encoder_dp_items(
    costs: Sequence[int],
    row_counts: Sequence[int],
    *,
    layout: EncoderParallelLayout,
    prepare_local: Callable[[list[int]], Callable[[], torch.Tensor]],
    exchange_status: ExchangeStatus,
    allgather: Callable[[torch.Tensor, list[int]], torch.Tensor],
    empty_output: torch.Tensor,
    on_prepared: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, ...]:
    """Place, run, and reorder atomic multimodal encoder items.

    Every rank derives the same plan from ``costs`` (physical encoder input
    tokens). ``prepare_local`` gets the indices placed on this replica, in
    item order, and returns a function that encodes them into their output
    rows. When replicas use TP, every rank shares its preparation status before
    any rank encodes, so no rank starts encoder TP work while a replica peer
    has failed.
    A replica with no items sends ``empty_output``, a ``[0, dim]`` tensor with
    the output dtype and device. One status exchange before the output gather
    shares failures, including preparation failures for replicas without TP.
    ``allgather`` concatenates the
    per-replica outputs in replica order using the given row counts,
    including zero rows. When supplied, ``on_prepared`` runs on every rank
    after local input preparation and before encoding.

    Returns an output view for each item, in item order.
    """

    def prepare() -> tuple[tuple[int, ...], list[int], Callable[[], torch.Tensor] | None]:
        ranks = plan_encoder_dp_items(costs, layout.dp_size)
        local = [index for index, rank in enumerate(ranks) if rank == layout.dp_rank]
        encode_local = prepare_local(local) if local else None
        if on_prepared is not None:
            on_prepared()
        return ranks, local, encode_local

    if layout.tp_size > 1:
        with encoder_dp_error_guard(exchange_status):
            ranks, local, encode_local = prepare()

    with encoder_dp_error_guard(exchange_status):
        if layout.tp_size == 1:
            ranks, local, encode_local = prepare()
        if encode_local is None:
            local_output = empty_output
        else:
            local_output = encode_local()
            # Free the prepared inputs before the output gather.
            del encode_local
            expected_shape = (sum(row_counts[index] for index in local), empty_output.shape[1])
            if local_output.shape != expected_shape:
                raise MultimodalEncoderContractError(
                    "Multimodal encoder output shape does not match its local DP assignment: "
                    f"expected {expected_shape}, got {tuple(local_output.shape)}."
                )
            local_output = local_output.to(empty_output).contiguous()

    rank_row_counts = [0] * layout.dp_size
    for rank, rows in zip(ranks, row_counts, strict=True):
        rank_row_counts[rank] += rows
    gathered = allgather(local_output, rank_row_counts) if sum(rank_row_counts) else empty_output
    rank_row_starts = list(accumulate(rank_row_counts, initial=0))
    ordered_outputs = []
    for rank, rows in zip(ranks, row_counts, strict=True):
        ordered_outputs.append(gathered[rank_row_starts[rank] : rank_row_starts[rank] + rows])
        rank_row_starts[rank] += rows
    return tuple(ordered_outputs)


_PAYLOAD_ALIGNMENT = 256
"""Byte alignment of each tensor in a packed payload, so its view is aligned like an allocation."""

EncoderCost = tuple[int | None, int]
"""An encoder batch's ``(encoder input tokens, items)``. Tokens are ``None`` when not known."""


def group_encoder_owners(
    costs: Sequence[EncoderCost], max_tokens: int | None, max_items: int | None
) -> list[list[int]]:
    """Group consecutive owners while their batches fit one encoder step together.

    ``costs`` has one entry per TP rank. Owners without items are skipped. A
    group grows while its total tokens and items stay within ``max_tokens``
    and ``max_items``, the limits each rank's scheduler uses for its own
    batch. Without limits, or with unknown tokens, an owner is a group alone.
    """
    groups: list[list[int]] = []
    # The tokens and items the last group still has room for; None when it is closed.
    room: tuple[int, int] | None = None
    for owner, (tokens, items) in enumerate(costs):
        if not items:
            continue
        if room is not None and tokens is not None and tokens <= room[0] and items <= room[1]:
            groups[-1].append(owner)
            room = (room[0] - tokens, room[1] - items)
            continue
        groups.append([owner])
        room = (
            None
            if tokens is None or max_tokens is None or max_items is None
            else (max_tokens - tokens, max_items - items)
        )
    return groups


def _encoder_cost(
    encoder_inputs: Sequence[tuple[MultimodalParams, Sequence[int], str]],
) -> EncoderCost:
    tokens = 0
    for param, _, _ in encoder_inputs:
        metadata = param.item_metadata
        if metadata is None or metadata.encoder_token_lengths is None:
            tokens = None
            break
        tokens += sum(metadata.encoder_token_lengths)
    return tokens, sum(len(lengths) for _, lengths, _ in encoder_inputs)


def _payload_views(payload: torch.Tensor, descriptors: Sequence[tuple]) -> list[torch.Tensor]:
    """Typed views of the tensors packed in a ``uint8`` payload."""
    return [
        payload[offset : offset + torch.Size(shape).numel() * dtype.itemsize]
        .view(dtype)
        .view(shape)
        for _, shape, dtype, offset in descriptors
    ]


def _pack_payload(
    num_bytes: int, descriptors: Sequence[tuple], tensors: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Copy CUDA tensors into one ``uint8`` buffer at their descriptor offsets."""
    payload = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
    for view, tensor in zip(_payload_views(payload, descriptors), tensors, strict=True):
        view.copy_(tensor)
    return payload


def _shared_owner_inputs(
    owner_metadata: tuple, buffer: torch.Tensor | None
) -> list[tuple[MultimodalParams, Sequence[int], str]]:
    """Rebuild a peer owner's encoder inputs from its gathered metadata and packed bytes."""
    owner_leaves, owner_spec, owner_descriptors, _, _ = owner_metadata
    leaves = list(owner_leaves)
    if owner_descriptors:
        # Without a payload, every CUDA tensor of the group is empty.
        if buffer is None:
            buffer = torch.empty(0, dtype=torch.uint8, device="cuda")
        for (index, *_), view in zip(owner_descriptors, _payload_views(buffer, owner_descriptors)):
            leaves[index] = view
    return [
        (
            MultimodalParams(
                multimodal_data={modality: data, MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata}
            ),
            lengths,
            modality,
        )
        for data, metadata, lengths, modality in tree_unflatten(leaves, owner_spec)
    ]


def iter_encoder_rank_inputs(
    encoder_inputs: Sequence[tuple[MultimodalParams, Sequence[int], str]],
    *,
    mapping: Mapping,
    exchange_status: ExchangeStatus,
    max_tokens: int | None = None,
    max_items: int | None = None,
) -> Iterator[list[tuple[int, Sequence[tuple[MultimodalParams, Sequence[int], str]]]]]:
    """Share the ADP ranks' prepared encoder batches for one encoder TP pass.

    Only what the encoder reads is shared: the modality payload, the item
    metadata, the row counts and the modality. Other request data, such as
    full-prompt MRoPE positions, stays on its owner. CPU values use one object
    collective. Consecutive owners whose batches fit one encoder step
    together (see ``group_encoder_owners``) form a group. Each owner in a
    group packs its CUDA tensors into one byte buffer, so a group costs one
    status exchange and one variable-size allgather. The owner encodes its
    own local inputs.

    Yields one list of ``(owner, inputs)`` per group, in owner order. The
    caller must drop each group before asking for the next, so temporary GPU
    storage stays bounded by one encoder step.
    """
    with encoder_dp_error_guard(exchange_status):
        leaves, spec = tree_flatten(
            [
                (param.multimodal_data[modality], param.item_metadata, lengths, modality)
                for param, lengths, modality in encoder_inputs
            ]
        )
        local_cuda_leaves: list[torch.Tensor] = []
        descriptors = []
        num_bytes = 0
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                local_cuda_leaves.append(leaf)
                descriptors.append((index, tuple(leaf.shape), leaf.dtype, num_bytes))
                num_bytes += -(-leaf.numel() * leaf.element_size() // _PAYLOAD_ALIGNMENT) * (
                    _PAYLOAD_ALIGNMENT
                )
                leaves[index] = None
        cost = _encoder_cost(encoder_inputs)
    rank_metadata = Distributed.get(mapping).tp_allgather(
        (leaves, spec, descriptors, num_bytes, cost)
    )

    for owners in group_encoder_owners(
        [metadata[4] for metadata in rank_metadata], max_tokens, max_items
    ):
        sizes = [rank_metadata[rank][3] if rank in owners else 0 for rank in range(mapping.tp_size)]
        payload = None
        if sum(sizes):
            with encoder_dp_error_guard(exchange_status):
                payload = (
                    _pack_payload(num_bytes, descriptors, local_cuda_leaves)
                    if mapping.tp_rank in owners
                    else torch.empty(0, dtype=torch.uint8, device="cuda")
                )
            # The owner keeps its local tensors; its packed copy is freed here.
            payload = allgather(payload, mapping, dim=0, sizes=sizes)
        starts = accumulate((sizes[owner] for owner in owners), initial=0)
        group = [
            (
                owner,
                encoder_inputs
                if owner == mapping.tp_rank
                else _shared_owner_inputs(
                    rank_metadata[owner],
                    None if payload is None else payload[start : start + sizes[owner]],
                ),
            )
            for owner, start in zip(owners, starts)
        ]
        # The views keep what the group needs.
        del payload
        yield group
        # Free this group's buffers before the next group's are allocated.
        del group
