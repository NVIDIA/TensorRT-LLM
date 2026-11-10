# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive EPLB load balancer and its per-layer handle.

Statistics and slot routing run inside the forward, so CUDA graphs capture them and no host work
is needed per step.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist

from tensorrt_llm.logger import logger

from ...live_rows import current_word
from ..moe_load_balancer import BaseMoeLoadBalancer
from .policy import (
    ExpertGeometry,
    ExpertStatistics,
    TimeModel,
    active_slots,
    build_route_tables,
    statistics_buffers,
)
from .runtime import AdaptiveRuntime, DecisionConfig, DecisionProcess
from .transport import (
    PeerSlotTransport,
    find_backends,
    migrate_model_weights,
    peer_copy_gap,
    prepare_layers,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import MoeAdaptiveLoadBalancerConfig
    from tensorrt_llm.mapping import Mapping

    from .transport import AdaptiveEplbBackend


def time_layer_copy(
    transport: PeerSlotTransport, layer: int, geometry: ExpertGeometry, stream: torch.cuda.Stream
) -> float:
    """Milliseconds of the copies of one layer's install, at the slowest rank; collective over EP.

    Every rank copies into its free slots of ``layer`` as many experts as an install places on a
    GPU, each from the home slots of a rank half the EP group away. The ranks copy at the same
    time, once every rank's blocks hold their weights, and the same copies run once untimed
    first. No route table points at a free slot, so the copies change no result.
    """
    rank, size, homes = transport.ep_rank, geometry.ep_size, geometry.local_experts
    replicas = active_slots(geometry) // size - homes
    experts = [
        (layer, homes + j, (rank + size // 2 + j) % size, j % homes) for j in range(replicas)
    ]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    transport.allgather(None)
    for _ in range(2):
        transport.copy(experts, stream, start, end)
        end.synchronize()
    return max(transport.allgather(start.elapsed_time(end)))


NO_LAYER = "no MoE layer runs on an implementation that supports adaptive EPLB"
"""Why adaptive EPLB stays off on a model whose MoE layers all run outside the balancer."""


class AdaptiveMoeLayerLoadBalancer:
    """Handle of one MoE layer (a :class:`~..moe_load_balancer.MoeLayerLoadBalancer`): records its
    routes and maps experts to slots on the device, inside CUDA graphs. Its slot weights move whole
    after loading, so none of the host weight hooks of standard EPLB apply."""

    produces_replica_plan = False
    stages_routes = True
    # Set by the layer's MoE module when its kernel skips the rows past the engine's live-row
    # word: the statistics then skip them too, else they count every row the kernel computes.
    counts_live_rows = False

    def __init__(
        self,
        manager: AdaptiveMoeLoadBalancer,
        *,
        layer_idx: int,
        expert_count: int,
        slot_count_per_rank: int,
        repeated_count: int = 1,
    ) -> None:
        self.manager = manager
        self.layer_idx = int(layer_idx)
        self.expert_count = int(expert_count)
        self.num_slots = int(slot_count_per_rank) * manager.ep_size
        self.repeated_count = int(repeated_count)
        self.statistics: ExpertStatistics | None = None
        self._tables: list[torch.Tensor] = []
        self._slot_dtype = torch.int32
        self._route_staging: Callable[[int, int], object | None] | None = None
        self._pending: tuple[torch.Tensor, bool] | None = None
        self._route_into = None

    def get_repeat_count(self) -> int:
        return self.repeated_count

    def is_dynamic_routing(self) -> bool:
        """False: no host statistics, host weight migration or stage hooks."""
        return False

    def records_statistics(self) -> bool:
        """True: the MoE modules pass every forward's routed experts to the statistics hooks."""
        return True

    def need_load_shared_weights(self) -> bool:
        return False

    def get_load_expert_ids(self) -> list[int]:
        """The experts this rank loads from the checkpoint: its home experts."""
        local = self.expert_count // self.manager.ep_size
        start = self.manager.ep_rank * local
        return list(range(start, start + local))

    def home_layout(self) -> list[int]:
        """Every expert on its home slot (:meth:`~.policy.ExpertGeometry.home_layout`)."""
        geometry = ExpertGeometry(self.expert_count, self.num_slots, self.manager.ep_size)
        return geometry.home_layout().tolist()

    def set_initial_weight_assignments(self, initial_weight_assignments: Sequence[int]) -> None:
        """Nothing to set: every layer starts at its home layout."""

    def prepare_device_state(
        self,
        slot_dtype: torch.dtype,
        device: torch.device | str,
        route_staging: Callable[[int, int], object | None] | None = None,
    ) -> None:
        """Bind the layer's statistics and route tables on ``device`` before the first forward.

        ``slot_dtype`` is the dtype the backend reads slot ids in and ``route_staging`` its
        ``route_staging`` method, if it has one.
        """
        manager = self.manager
        flag = manager.recording_flag(device)
        self.statistics = ExpertStatistics.of(manager.statistics(device), self.layer_idx, flag)
        tables = manager.route_tables(device)[self.layer_idx]
        experts = self.expert_count
        slots = self.num_slots
        self._tables = tables.split([experts, experts, slots, slots])
        self._slot_dtype = slot_dtype
        self._route_staging = route_staging
        self._route_into = torch.ops.trtllm.adaptive_eplb_route_into

    def update_local_statistic(
        self,
        local_raw_expert_ids: torch.Tensor,
        is_first_stage: bool,
        is_last_stage: bool,
        allreduce: object = None,
    ) -> None:
        """Keep the routes of a chunk or repeat; the next :meth:`route` of the same ids adds them to
        the statistics.

        Forwards that are not recorded (context steps, warmup) record nothing, except under CUDA
        graph capture, where the device flag gates every replay.
        """
        statistics = self.statistics
        pending = self._pending
        self._pending = None
        if pending is not None:
            statistics.record(*pending, self._live_rows(pending[0]))
        if self.manager.recording or (
            statistics.rows.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            self._pending = (local_raw_expert_ids, is_first_stage)

    update_statistic_with_local_ids = update_local_statistic

    def route(
        self,
        token_selected_experts: torch.Tensor,
        offset_by_ep_rank: bool = False,
        *,
        overlap_fn: Callable[[], None] | None = None,
        token_final_scales: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map every routed expert to a slot, in one launch that also adds the kept routes to the
        statistics.

        Ids outside the experts map to ``num_slots``. With ``token_final_scales`` (``[tokens,
        top_k]`` float32) and a backend that stages routes, the launch writes the slots and weights
        into the backend's staging and returns a view of it, which the backend does not stage again.
        """
        ids = token_selected_experts
        pending = self._pending
        self._pending = None
        if pending is not None and pending[0] is not ids:
            self.statistics.record(*pending, self._live_rows(pending[0]))
            pending = None
        staging = None
        if (
            self._route_staging is not None
            and token_final_scales is not None
            and token_final_scales.dtype == torch.float32
            and token_final_scales.is_contiguous()
            and token_final_scales.shape == ids.shape
        ):
            staging = self._route_staging(*ids.shape)
        if staging is None:
            out = torch.empty_like(ids, dtype=self._slot_dtype)
            weights = (None, None)
        else:
            out = staging.topk_idx
            weights = (token_final_scales, staging.topk_weights)
        if pending is None:
            statistics_args = (None, None, None, True)
        else:
            statistics_args = (*self.statistics, pending[1])
        self._route_into(
            ids,
            *self._tables,
            self.manager.ep_rank,
            out,
            *weights,
            *statistics_args,
            self._live_rows(ids),
        )
        if overlap_fn is not None:
            overlap_fn()
        return out[: ids.shape[0]]

    def _live_rows(self, ids: torch.Tensor) -> torch.Tensor | None:
        """The engine's live-row word if the layer's kernel skips the rows past it."""
        return current_word(ids.device) if self.counts_live_rows else None


class AdaptiveMoeLoadBalancer(BaseMoeLoadBalancer):
    """Model-level adaptive EPLB balancer of rank ``ep_rank`` of ``ep_size``.

    Statistics and routes live on the device, so a forward outside ``MoeLoadBalancerIterContext``
    (the first replay after a graph capture) is recorded like any other.
    """

    def __init__(
        self, *, ep_rank: int, ep_size: int, config: MoeAdaptiveLoadBalancerConfig, mapping: Mapping
    ) -> None:
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.config = config
        self.mapping = mapping
        self.num_slots = int(config.num_slots)
        self.num_local_slots = self.num_slots // self.ep_size
        self.layers: list[AdaptiveMoeLayerLoadBalancer] = []
        self.next_layer_repeated_count: int | None = None
        self.in_iter = False
        self.is_shutdown = False
        self._previous_balancer: BaseMoeLoadBalancer | None = None
        self.enable_statistic = False
        self.enable_update_weights = False
        self.transport: PeerSlotTransport | None = None
        self.copy_stream: torch.cuda.Stream | None = None
        self.runtime: AdaptiveRuntime | None = None
        # Why adaptive EPLB stays off on this platform (:meth:`prepare_runtime`), if it does.
        self.off_reason: str | None = None
        # Whether the forwards now running record their routes.
        self.recording = False
        self._recording_flag: torch.Tensor | None = None
        self._route_tables: torch.Tensor | None = None
        self._statistics: dict[str, torch.Tensor] | None = None

    def is_static_routing(self) -> bool:
        return False

    def set_repeated_for_next_layer(self, repeated_count: int) -> None:
        if repeated_count <= 0:
            raise ValueError("repeat count must be positive")
        self.next_layer_repeated_count = int(repeated_count)

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: torch.cuda.Stream | None = None,
    ) -> AdaptiveMoeLayerLoadBalancer:
        """Register one MoE layer and return its handle; ``ValueError`` if it does not fit the config."""
        self.config.validate_expert_capacity(expert_count)
        if int(slot_count_per_rank) != self.num_local_slots:
            raise ValueError(
                f"layer slot count {slot_count_per_rank} differs from the configured "
                f"{self.num_local_slots} local slots"
            )
        layer = AdaptiveMoeLayerLoadBalancer(
            self,
            layer_idx=len(self.layers),
            expert_count=expert_count,
            slot_count_per_rank=slot_count_per_rank,
            repeated_count=self.next_layer_repeated_count or 1,
        )
        self.next_layer_repeated_count = None
        self.layers.append(layer)
        return layer

    def _nothing_before_runtime(self, *args: object, **kwargs: object) -> None:
        """Nothing to do: the slot weights move in :meth:`prepare_runtime`."""

    register_weight_slots_after_to_cuda = _nothing_before_runtime
    finalize_model = _nothing_before_runtime
    set_warm_up_iter_count = _nothing_before_runtime

    def prepare_runtime(self, model: torch.nn.Module) -> None:
        """Move the slot weights into MNNVL memory, time one layer's install copy
        (:func:`time_layer_copy`) and start the runtime under the backend's time model. Collective
        over EP; runs before the KV cache is sized.

        Where any EP rank cannot run adaptive EPLB (:meth:`platform_gap`) or balances no layer,
        every rank logs why and keeps it off: the layers serve their home layouts and record no
        statistics.
        """
        backends = find_backends(model, self.layers)
        device = torch.device("cuda", torch.cuda.current_device())
        prepare_layers(backends, self.layers, device)
        transport = PeerSlotTransport(self.mapping)
        # A rank that balances no layer votes too, so no rank waits for it.
        vote = self.platform_gap(backends[self.layers[0].layer_idx]) if self.layers else NO_LAYER
        gaps = transport.allgather(vote)
        stopped = [f"{gap} (EP rank {rank})" for rank, gap in enumerate(gaps) if gap is not None]
        if stopped:
            self.off_reason = stopped[0]
            logger.warning(
                f"adaptive EPLB is off: {self.off_reason}; the MoE layers keep their home layouts"
            )
            return
        first = self.layers[0].layer_idx
        self.transport = transport
        self.copy_stream = migrate_model_weights(backends, transport, device)
        time_model = backends[first].decode_time_model(
            transport.expert_bytes(first), torch.cuda.get_device_name()
        )
        geometry = ExpertGeometry(self.layers[0].expert_count, self.num_slots, self.ep_size)
        copy_ms = time_layer_copy(transport, first, geometry, self.copy_stream)
        logger.info(
            f"adaptive EPLB balances {len(self.layers)} MoE layers; one layer's install copy "
            f"takes {copy_ms:.3f} ms at start-up"
        )
        self.start_runtime(TimeModel(**time_model), self.mapping.gpus_per_node, copy_ms)

    def platform_gap(self, backend: AdaptiveEplbBackend) -> str | None:
        """Why adaptive EPLB cannot run on this rank, or ``None``: the backend's decode time model
        needs a calibration for this GPU, and the installs copy experts from every EP rank
        (:func:`~.transport.peer_copy_gap`)."""
        device_name = torch.cuda.get_device_name()
        if not backend.has_decode_time_model(device_name):
            return f"{type(backend).__name__} has no decode time model calibrated on {device_name}"
        return peer_copy_gap(self.mapping)

    def start_runtime(
        self, model: TimeModel, ranks_per_node: int, copy_ms: float | None = None
    ) -> None:
        """Start the windows, the decision process and the installs, the decisions taking
        ``copy_ms`` as one layer's copy time until an install is timed. Collective over EP."""
        if not dist.is_initialized():
            raise RuntimeError(
                "adaptive EPLB needs torch.distributed: its statistics gathers run on a gloo group "
                "of the EP ranks, and a MoE backend that supports it initializes the default group"
            )
        geometry = ExpertGeometry(self.layers[0].expert_count, self.num_slots, self.ep_size)
        homes = np.array([layer.home_layout() for layer in self.layers], dtype=np.int64)
        decision = DecisionProcess(
            DecisionConfig(geometry, model, homes, self.ep_rank, ranks_per_node, copy_ms=copy_ms)
        )
        # The runtime's worker thread gathers on a group of its own: a group the main thread also
        # uses would interleave their collectives differently on different ranks. Its window and
        # copy gathers alternate on every rank, so one group carries both.
        ranks = sorted(self.transport.allgather(dist.get_rank()))
        group = dist.new_group(ranks, backend="gloo", use_local_synchronization=True)

        def gather(send: torch.Tensor, rows: list[torch.Tensor]) -> None:
            dist.all_gather(rows, send, group=group)

        device = torch.device("cuda", torch.cuda.current_device())
        self.runtime = AdaptiveRuntime(
            self.layers,
            self.transport,
            self.copy_stream,
            decision,
            gather,
            device,
            self.route_tables(device),
            self.statistics(device),
        )

    def statistics(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        """Every layer's statistics in contiguous buffers (:func:`~.policy.statistics_buffers`)."""
        if self._statistics is None:
            self._statistics = statistics_buffers(
                len(self.layers), self.layers[0].expert_count, device
            )
        return self._statistics

    def route_tables(self, device: torch.device | str) -> torch.Tensor:
        """The route tables of all layers, one int32 row of ``2 * (experts + slots)`` per layer.

        Every layer starts with the table of its home layout.
        """
        if self._route_tables is None:
            layer = self.layers[0]
            home = build_route_tables(layer.home_layout(), None, layer.expert_count, self.num_slots)
            self._route_tables = torch.tensor(
                [home.packed()] * len(self.layers), dtype=torch.int32, device=device
            )
        return self._route_tables

    def recording_flag(self, device: torch.device | str) -> torch.Tensor:
        """The ``[1]`` int32 flag the statistics read: set for generation steps outside warmup."""
        if self._recording_flag is None:
            self._recording_flag = torch.zeros(1, dtype=torch.int32, device=device)
        return self._recording_flag

    def observe_scheduled_batch(
        self,
        num_generation_requests: int,
        generation_only: bool,
        is_warmup: bool,
        generation_progress: Callable[[int, int], tuple[int, int, int]],
    ) -> None:
        """Record the statistics of generation steps outside warmup only, and pass the generation
        requests' progress to the runtime when it asks for it."""
        recording = num_generation_requests > 0 and generation_only and not is_warmup
        if self.off_reason is not None or (
            self.runtime is not None and self.runtime.failure is not None
        ):
            recording = False  # nothing reads the statistics of adaptive EPLB that is off
        if recording != self.recording:
            self.recording = recording
            if self._recording_flag is not None:
                self._recording_flag.fill_(int(recording))
        if self.runtime is not None and self.runtime.wants_request_progress:
            self.runtime.add_request_progress(num_generation_requests, generation_progress)

    def iteration_stats(self) -> dict[str, int] | None:
        """The runtime's counters (:meth:`~.runtime.AdaptiveRuntime.counters`) once it runs."""
        return None if self.runtime is None else self.runtime.counters()

    def start_iter(self) -> None:
        if self.in_iter:
            raise RuntimeError("already in a MoE load-balancer iteration")
        self.in_iter = True
        if self.runtime is not None and self.enable_update_weights:
            if not torch.cuda.is_current_stream_capturing():
                self.runtime.boundary()

    def end_iter(self) -> None:
        if not self.in_iter:
            raise RuntimeError("not in a MoE load-balancer iteration")
        self.in_iter = False

    def shutdown(self, local_safe: bool = True) -> bool:
        """Stop the runtime once every EP rank is safe, else return ``False``. Collective over EP."""
        if self.is_shutdown:
            return True
        local_safe = bool(local_safe and not self.in_iter)
        if self.transport is not None:
            local_safe = all(self.transport.allgather(local_safe))
        if not local_safe:
            return False
        if self.runtime is not None:
            self.runtime.close()
        if self.transport is not None:
            self.copy_stream.synchronize()
        self.is_shutdown = True
        return True
