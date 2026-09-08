# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Atomic wrapper plan state and CUDA-stream lifetime management."""

from collections.abc import Callable
from dataclasses import dataclass
import functools
import _thread
from typing import Concatenate, ParamSpec, Protocol, TypeVar, cast

import torch

from flashinfer.utils import ceil_div

from .common import _SIGNED_INT32_MAX
from .compiler import _get_compiled_block_sparse
from .config import (
    _BlockSparseStaticProfile,
    _resolve_block_sparse_launch_spec,
)
from .prepared import _BlockSparseRouteLayout

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _PlanLockOwner(Protocol):
    """Structural type required by the plan-serialization decorator."""

    _plan_lock: _thread.LockType


_S = TypeVar("_S", bound=_PlanLockOwner)


def _serialize_plan(
    method: Callable[Concatenate[_S, _P], _R],
) -> Callable[Concatenate[_S, _P], _R]:
    """Serialize plan calls without making run wait for a replan."""

    @functools.wraps(method)
    def serialized(self: _S, /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        with self._plan_lock:
            return method(self, *args, **kwargs)

    return cast(Callable[Concatenate[_S, _P], _R], serialized)


@dataclass(frozen=True)
class _BlockSparsePlanState:
    """One complete launch state published by a block-sparse wrapper.

    Every state executes one ``prepare -> prepared-route attention`` adapter,
    including a pattern whose rows select every KV block. Caller BSR and token
    mask tensors belong to individual runs and are never retained here.

    Runtime geometry, dtypes, the compiled launch, and the readiness event are
    published together. ``run()`` therefore sees either the complete old state
    or the complete new one, never a mix. Plan-owned row offsets describe
    uniform capacity slices. Mutable route scratch, the optional unmasked-ABI
    dummy mask, and the event are also state-owned.
    Policy is immutable after publication; the cached compiled adapter is
    shared read-only.

    One revision owns one mutable route workspace. Ordered runs on one
    stream, or externally synchronized cross-stream runs, are valid. Unordered
    concurrent runs of the same revision are unsupported because their prepare
    launches would race. Different wrappers and published revisions own
    independent workspaces.

    Caller routing storage must outlive its queued run or captured graph.
    ``record_stream()`` extends allocator lifetime for eager runs only; it
    cannot prevent in-place modification or replace graph ownership.
    """

    device: torch.device
    batch_size: int
    seq_len_q: int
    seq_len_kv: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    q_block_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    output_dtype: torch.dtype
    use_kv_valid_bits: bool
    page_size: int | None

    # Only an unmasked specialization needs a shape-correct ABI placeholder.
    dummy_kv_valid_bits: torch.Tensor | None

    # Immutable row capacities and mutable per-run route payload.
    row_route_offsets: torch.Tensor
    route_workspace: torch.Tensor
    # Semantic row bound; unlike route capacity, this distinguishes
    # multiple semantic blocks packed into one prepared route.
    max_blocks_per_row: int

    policy: tuple[tuple[str, object], ...]
    compiled: Callable[..., object]

    # All plan-stream work happens-before run after waiting on this event.
    ready_event: torch.cuda.Event
    ready_stream_handle: int


def _allocate_dummy_kv_valid_bits(
    *,
    batch_size: int,
    seq_len_kv: int,
    device: torch.device,
) -> torch.Tensor:
    """Allocate the shape-correct placeholder required by the prepare ABI."""

    return torch.zeros(
        (batch_size, (seq_len_kv + 31) // 32),
        dtype=torch.uint32,
        device=device,
    )


def _allocate_route_storage(
    *,
    device: torch.device,
    route_layout: _BlockSparseRouteLayout,
    uniform_row_route_capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate uniform plan-owned route slices and mutable route scratch."""

    expected_capacity = route_layout.num_rows * uniform_row_route_capacity
    if expected_capacity != route_layout.route_metadata_capacity:
        raise RuntimeError("uniform row capacity does not match route layout")
    if expected_capacity > _SIGNED_INT32_MAX:
        raise OverflowError(
            "block-sparse route capacity must fit in signed int32: "
            f"got {expected_capacity}"
        )
    offsets_i64 = torch.arange(
        route_layout.num_rows + 1,
        dtype=torch.int64,
        device=device,
    ) * int(uniform_row_route_capacity)
    row_route_offsets = offsets_i64.to(torch.int32)
    route_workspace = torch.empty(
        route_layout.workspace_size_words,
        dtype=torch.int32,
        device=device,
    )
    return row_route_offsets, route_workspace


def _record_block_sparse_plan_ready_event(
    stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """Record the event that closes every plan-owned GPU operation."""

    # External events remain legal wait dependencies inside CUDA Graph capture.
    event = torch.cuda.Event(external=True)
    event.record(stream)
    return event


def _build_block_sparse_plan_state(
    static: _BlockSparseStaticProfile,
    *,
    device: torch.device,
    device_index: int,
    plan_stream: torch.cuda.Stream,
) -> _BlockSparsePlanState:
    """Build and close one complete state after storage validation."""

    assert static.max_blocks_per_row is not None
    max_row_route_capacity = ceil_div(
        static.max_blocks_per_row * static.kv_block_size,
        static.kv_route_size,
    )
    num_rows = (
        static.batch_size
        * static.num_kv_heads
        * ceil_div(static.seq_len_q, static.q_block_size)
    )
    route_layout = _BlockSparseRouteLayout.create(
        kv_route_size=static.kv_route_size,
        kv_block_size=static.kv_block_size,
        page_size=static.page_size,
        has_token_bits=static.use_kv_valid_bits,
        route_metadata_capacity=num_rows * max_row_route_capacity,
        num_rows=num_rows,
    )
    with torch.cuda.device(device_index), torch.cuda.stream(plan_stream):
        spec = _resolve_block_sparse_launch_spec(
            device_index=device_index,
            batch_size=static.batch_size,
            seq_len_q=static.seq_len_q,
            seq_len_kv=static.seq_len_kv,
            num_qo_heads=static.num_qo_heads,
            num_kv_heads=static.num_kv_heads,
            head_dim=static.head_dim,
            q_block_size=static.q_block_size,
            kv_block_size=static.kv_block_size,
            kv_route_size=static.kv_route_size,
            page_size=static.page_size,
            dtype_key=static.dtype_key,
            mask_type=static.mask_type,
            use_kv_valid_bits=static.use_kv_valid_bits,
            max_row_route_capacity=max_row_route_capacity,
        )
        policy = (
            *spec.policy,
            ("max_blocks_per_row", static.max_blocks_per_row),
        )
        compiled = _get_compiled_block_sparse(spec.compile_key)
        dummy_kv_valid_bits = (
            None
            if static.use_kv_valid_bits
            else _allocate_dummy_kv_valid_bits(
                batch_size=static.batch_size,
                seq_len_kv=static.seq_len_kv,
                device=device,
            )
        )
        row_route_offsets, route_workspace = _allocate_route_storage(
            device=device,
            route_layout=route_layout,
            uniform_row_route_capacity=max_row_route_capacity,
        )
        ready_event = _record_block_sparse_plan_ready_event(plan_stream)

    return _BlockSparsePlanState(
        device=device,
        batch_size=static.batch_size,
        seq_len_q=static.seq_len_q,
        seq_len_kv=static.seq_len_kv,
        num_qo_heads=static.num_qo_heads,
        num_kv_heads=static.num_kv_heads,
        head_dim=static.head_dim,
        q_block_size=static.q_block_size,
        q_dtype=static.q_dtype,
        kv_dtype=static.kv_dtype,
        output_dtype=static.output_dtype,
        use_kv_valid_bits=static.use_kv_valid_bits,
        page_size=static.page_size,
        dummy_kv_valid_bits=dummy_kv_valid_bits,
        row_route_offsets=row_route_offsets,
        route_workspace=route_workspace,
        max_blocks_per_row=static.max_blocks_per_row,
        policy=policy,
        compiled=compiled,
        ready_event=ready_event,
        ready_stream_handle=plan_stream.cuda_stream,
    )


def _wait_and_record_block_sparse_plan(
    state: _BlockSparsePlanState,
    stream: torch.cuda.Stream,
) -> None:
    """Acquire one state on ``stream`` and retain all plan-owned launch storage."""

    if stream.device != state.device:
        raise ValueError("run stream must share the planned CUDA device")
    if stream.cuda_stream != state.ready_stream_handle:
        stream.wait_event(state.ready_event)

    if state.dummy_kv_valid_bits is not None:
        state.dummy_kv_valid_bits.record_stream(stream)
    state.row_route_offsets.record_stream(stream)
    state.route_workspace.record_stream(stream)


__all__ = [
    "_BlockSparsePlanState",
    "_allocate_dummy_kv_valid_bits",
    "_allocate_route_storage",
    "_build_block_sparse_plan_state",
    "_record_block_sparse_plan_ready_event",
    "_serialize_plan",
    "_wait_and_record_block_sparse_plan",
]
