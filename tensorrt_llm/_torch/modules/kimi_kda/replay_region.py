# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA replay storage declarations and request-slot ownership."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    _compute_index_mapper_capacity,
    _read_guard_page_setting,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaStateLayout,
    _get_num_cuda_graph_padding_dummy_slots,
    _host_slots_to_device,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.regions import (
    AllocationUnit,
    HandOff,
    Plane,
    Region,
    RegionLayer,
    RegionLifeCycle,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.mapping import Mapping as ParallelMapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BufferConfig, DataRole, _KVCache

if TYPE_CHECKING:
    from .cache_manager import KDAReplayState

ReplayLayout = dict[DataRole, tuple[torch.dtype, list[int]]]
REPLAY_STATE_REGION = "replay_state"


class KDAReplayRegion:
    """Declare KDA's replay planes and bind their request-local views.

    Args:
        state: The selected KDA replay algorithm.
        context: Rank-local recurrent geometry, before pool allocation.
        layer_offsets: Global model layer to local manager layer mapping.
        max_living_caches: Bound on active, suspended and reserved caches.
        max_util_for_resume: Utilization ceiling at which a cache may resume.
    """

    def __init__(
        self,
        state: KDAReplayState,
        context: MambaStateLayout,
        layer_offsets: Mapping[int, int],
        max_living_caches: int,
        max_util_for_resume: float,
    ) -> None:
        self.state = state
        self.layouts = {
            layer: layout
            for layer in context.mamba_pp_layers
            if (layout := state.replay_layout(context, layer))
        }
        self.roles = frozenset(role for layout in self.layouts.values() for role in layout)
        self.region: Region | None = None
        if self.layouts and not state.replay_shares_state_slot:
            self.region = Region(
                name=REPLAY_STATE_REGION,
                life_cycle=RegionLifeCycle(AllocationUnit.REQUEST, HandOff.NEVER),
                layers=tuple(
                    RegionLayer(
                        model_layer=layer_offsets[layer],
                        planes=tuple(
                            Plane(role, dtype.itemsize * math.prod(shape))
                            for role, (dtype, shape) in layout.items()
                        ),
                    )
                    for layer, layout in self.layouts.items()
                ),
                # One replay slot per living cache: KDA runs at beam width 1. Beam search
                # would multiply the owners by the beam width in ``replay_slot_count``.
                fixed_slot_count=state.replay_slot_count(max_living_caches, max_util_for_resume),
            )
        self.group_id: int | None = None
        self._indices: DeviceCopyBuffer | None = None
        self._host_indices: torch.Tensor | None = None
        self._request_indices: dict[int, int] = {}
        self._initialized_requests: set[int] = set()
        self._last_index_key: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    @property
    def bytes_per_slot(self) -> int:
        return self.region.bytes_per_slot() if self.region is not None else 0

    def declare_regions(self) -> Sequence[Region]:
        return (self.region,) if self.region is not None else ()

    def shared_buffers(self, global_layer: int) -> list[BufferConfig]:
        """Buffers addressed by the recurrent slot in draft-records mode."""
        if not self.state.replay_shares_state_slot:
            return []
        return [
            BufferConfig(role=role, size=dtype.itemsize * math.prod(shape))
            for role, (dtype, shape) in self.layouts.get(global_layer, {}).items()
        ]

    def bind(
        self,
        get_buffer: Callable[[int, DataRole, torch.dtype, list[int]], torch.Tensor],
        layer_offsets: Mapping[int, int],
        group_ids: Sequence[int],
        batch_capacity: int,
        device: torch.device,
    ) -> None:
        """Bind allocated planes and stable int32 batch indices before model-state binding."""
        if len(group_ids) != int(self.region is not None):
            raise RuntimeError("KDA replay requires exactly one lifecycle for its declared region")
        self.group_id = int(group_ids[0]) if group_ids else None
        if self.group_id is not None:
            self._indices = DeviceCopyBuffer(
                torch.zeros(batch_capacity, dtype=torch.int32, device=device),
                pin_memory=prefer_pinned(),
            )
            self._host_indices = torch.zeros(
                batch_capacity, dtype=torch.int32, pin_memory=prefer_pinned()
            )
        if self.layouts:
            self.state.bind_replay_pool(
                [
                    {
                        role: get_buffer(layer_offsets[layer], role, dtype, list(shape))
                        for role, (dtype, shape) in layout.items()
                    }
                    for layer, layout in self.layouts.items()
                ]
            )

    def cuda_indices(self, state_indices: torch.Tensor) -> torch.Tensor:
        """Return the stable replay vector, or the shared recurrent vector."""
        return self._indices.tensor if self._indices is not None else state_indices

    def request_indices(self, request_ids: Sequence[int]) -> list[int]:
        return [self._request_indices[rid] for rid in request_ids]

    def prepare(
        self,
        requests: Sequence[LlmRequest],
        caches: Mapping[int, _KVCache],
        state_indices: torch.Tensor,
        host_state_indices: torch.Tensor,
        num_contexts: int,
        skip_unchanged: bool,
    ) -> None:
        """Refresh replay slots after allocation, retaining initialized pages across suspension."""
        slots = (
            [
                int(caches[req.py_request_id].get_ssm_block_base_index(self.group_id))
                for req in requests
            ]
            if self.group_id is not None
            else host_state_indices[: len(requests)].tolist()
        )
        if min(slots, default=0) < 0:
            raise RuntimeError("KDA replay request has no resident replay slot")
        key = (tuple(req.py_request_id for req in requests), tuple(slots))
        # Replay pages can move while the recurrent slots remain unchanged.
        if skip_unchanged and key == self._last_index_key:
            return
        self._last_index_key = key
        if self._indices is not None:
            self._host_indices.zero_()
            self._host_indices[: len(slots)] = torch.tensor(slots, dtype=torch.int32, device="cpu")
            self._indices.update(self._host_indices)
        for req, slot in zip(requests, slots):
            self._request_indices[req.py_request_id] = slot
        fresh = [
            slot
            for req, slot in zip(requests[num_contexts:], slots[num_contexts:])
            if req.py_request_id not in self._initialized_requests
        ]
        if fresh:
            self.state.reset_replay_slots(_host_slots_to_device(fresh, state_indices.device))
        # Context slots are reset separately, after both index vectors are ready.
        self._initialized_requests.update(req.py_request_id for req in requests)

    def reset_context(
        self, num_contexts: int, state_indices: torch.Tensor, host_state_indices: torch.Tensor
    ) -> None:
        if num_contexts:
            self.state.reset_slots(
                state_indices[:num_contexts].long(),
                host_state_indices[:num_contexts].tolist(),
                replay_slots=self.cuda_indices(state_indices)[:num_contexts].long(),
            )

    def update(self, batch: MambaAcceptanceBatch, state_indices: torch.Tensor) -> None:
        """Record acceptance using graph metadata's replay slots when supplied."""
        if self.group_id is None:
            self.state.update(batch)
            return
        indices = getattr(
            getattr(batch.attention_metadata, "mamba_metadata", None), "replay_state_indices", None
        )
        if indices is None:
            indices = self.cuda_indices(state_indices)
        start = batch.num_contexts
        self.state.update(batch, indices[start : start + batch.num_generations].to(torch.int32))

    def release(self, request_id: int) -> None:
        self._request_indices.pop(request_id, None)
        self._initialized_requests.discard(request_id)
        self._last_index_key = None

    def shutdown(self) -> None:
        self._indices = None
        self._host_indices = None
        self._request_indices.clear()
        self._initialized_requests.clear()
        self._last_index_key = None

    @staticmethod
    def estimate_fixed_bytes(
        model_config: object,
        mapping: ParallelMapping,
        max_batch_size: int,
        max_util_for_resume: float,
        **kwargs,
    ) -> int:
        """Bytes of the fixed replay pool, as the manager will allocate it.

        The pool holds ``replay_slot_count`` slots, the same count the region declares,
        times the bytes of one slot. Beam width is 1 for KDA; beam search would scale the
        count, and so this estimate, by the beam width.
        """
        from .cache_manager import KDAReplayState

        if kwargs.get("kda_draft_records", False):
            return 0
        bytes_per_slot = KDAReplayState.estimate_bytes_per_slot(model_config, mapping, **kwargs)
        if not bytes_per_slot:
            return 0
        reserved = _get_num_cuda_graph_padding_dummy_slots(
            kwargs.get("spec_config"), max_batch_size
        ) + int(mapping.enable_attention_dp)
        _, capacity = _compute_index_mapper_capacity(
            max_batch_size,
            mapping,
            is_disagg=kwargs.get("is_disagg", False),
            disable_overlap_scheduler=kwargs.get("disable_overlap_scheduler", False),
            num_reserved_index_slots=reserved,
            has_guard_page=_read_guard_page_setting()[1] is not None,
            gen_preemption_headroom=kwargs.get("gen_preemption_headroom", 0),
        )
        return KDAReplayState.replay_slot_count(capacity, max_util_for_resume) * bytes_per_slot


__all__ = ["KDAReplayRegion", "REPLAY_STATE_REGION", "ReplayLayout"]
