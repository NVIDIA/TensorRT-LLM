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
from enum import IntEnum
from itertools import accumulate, pairwise
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from tensorrt_llm._torch.distributed import Distributed, allgather
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
) -> torch.Tensor:
    """Place, run, and reorder atomic multimodal encoder items.

    Every rank derives the same plan from ``costs`` (physical encoder input
    tokens). ``prepare_local`` gets the indices placed on this replica, in
    item order, and returns a function that encodes them into their output
    rows. Every rank shares its preparation status before any rank encodes,
    so no rank starts encoder TP work while a peer of its replica has failed.
    A replica with no items sends ``empty_output``, a ``[0, dim]`` tensor with
    the output dtype and device. A second status exchange shares encoder
    failures before the output gather. ``allgather`` concatenates the
    per-replica outputs in replica order using the given row counts,
    including zero rows.

    Returns one dense tensor with each item's ``row_counts`` rows, in item order.
    """
    with encoder_dp_error_guard(exchange_status):
        ranks = plan_encoder_dp_items(costs, layout.dp_size)
        local = [index for index, rank in enumerate(ranks) if rank == layout.dp_rank]
        encode_local = prepare_local(local) if local else None
    with encoder_dp_error_guard(exchange_status):
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
    if not sum(rank_row_counts):
        return local_output

    gathered = allgather(local_output, rank_row_counts)
    if all(left <= right for left, right in pairwise(ranks)):
        # Rank order already matches item order; avoid another full-size copy.
        return gathered
    rank_row_starts = list(accumulate(rank_row_counts, initial=0))
    ordered_outputs = []
    for rank, rows in zip(ranks, row_counts, strict=True):
        ordered_outputs.append(gathered[rank_row_starts[rank] : rank_row_starts[rank] + rows])
        rank_row_starts[rank] += rows
    return torch.cat(ordered_outputs, dim=0)


def _owner_leaves(
    owner: int,
    owner_leaves: list,
    owner_descriptors: list[tuple[int, tuple[int, ...], torch.dtype]],
    local_cuda_leaves: dict[torch.dtype, list[torch.Tensor]],
    mapping: Mapping,
    exchange_status: ExchangeStatus,
) -> list:
    """Send or receive one owner's CUDA tensors, one allgather per dtype.

    The owner only sends: it keeps its local tensors and frees the packed copy
    right away. Other ranks get views into one received buffer per dtype.
    """
    leaves = list(owner_leaves)
    by_dtype: dict[torch.dtype, list[tuple[int, tuple[int, ...]]]] = {}
    for index, shape, dtype in owner_descriptors:
        by_dtype.setdefault(dtype, []).append((index, shape))
    for dtype, entries in by_dtype.items():
        with encoder_dp_error_guard(exchange_status):
            total_elements = sum(torch.Size(shape).numel() for _, shape in entries)
            if mapping.tp_rank == owner:
                payload = torch.cat([tensor.reshape(-1) for tensor in local_cuda_leaves[dtype]])
            else:
                payload = torch.empty(0, dtype=dtype, device="cuda")
        if total_elements:
            payload = allgather(
                payload,
                mapping,
                dim=0,
                sizes=[total_elements if rank == owner else 0 for rank in range(mapping.tp_size)],
            )
        if mapping.tp_rank == owner:
            continue
        offset = 0
        for index, shape in entries:
            numel = torch.Size(shape).numel()
            leaves[index] = payload[offset : offset + numel].view(shape)
            offset += numel
    return leaves


def iter_encoder_rank_inputs(
    encoder_inputs: Sequence[tuple[MultimodalParams, Sequence[int], str]],
    *,
    mapping: Mapping,
    exchange_status: ExchangeStatus,
) -> Iterator[tuple[int, Sequence[tuple[MultimodalParams, Sequence[int], str]]]]:
    """Share one ADP owner's prepared encoder batch at a time for encoder TP.

    Only what the encoder reads is shared: the modality payload, the item
    metadata, the row counts and the modality. Other request data, such as
    full-prompt MRoPE positions, stays on its owner. CPU values use one object
    collective; CUDA tensors stay on the GPU and use the variable-size
    allgather. The owner encodes its own local inputs. The caller must drop
    each yielded batch before asking for the next one, so temporary GPU
    storage stays bounded by one owner's encoder budget.
    """
    with encoder_dp_error_guard(exchange_status):
        leaves, spec = tree_flatten(
            [
                (param.multimodal_data[modality], param.item_metadata, lengths, modality)
                for param, lengths, modality in encoder_inputs
            ]
        )
        local_cuda_leaves: dict[torch.dtype, list[torch.Tensor]] = {}
        descriptors = []
        for index, leaf in enumerate(leaves):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                local_cuda_leaves.setdefault(leaf.dtype, []).append(leaf)
                descriptors.append((index, tuple(leaf.shape), leaf.dtype))
                leaves[index] = None
    rank_metadata = Distributed.get(mapping).tp_allgather((leaves, spec, descriptors))

    for owner, (owner_leaves, owner_spec, owner_descriptors) in enumerate(rank_metadata):
        if not owner_descriptors and not tree_unflatten(owner_leaves, owner_spec):
            continue
        shared_leaves = _owner_leaves(
            owner, owner_leaves, owner_descriptors, local_cuda_leaves, mapping, exchange_status
        )
        if owner == mapping.tp_rank:
            shared_inputs = encoder_inputs
        else:
            shared_inputs = [
                (
                    MultimodalParams(
                        multimodal_data={
                            modality: data,
                            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
                        }
                    ),
                    lengths,
                    modality,
                )
                for data, metadata, lengths, modality in tree_unflatten(shared_leaves, owner_spec)
            ]
        del shared_leaves
        yield owner, shared_inputs
        # Free this owner's buffers before the next owner's are allocated.
        del shared_inputs
