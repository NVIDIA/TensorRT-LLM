# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-R planner and relay copy for one per-layer dynamic EPLB layer.

Every layer call submits a fixed sequence on the shared copy stream (COPY)
and never reads device data on the host::

    MAIN: record input_ready
    COPY: wait input_ready -> HALO-R scheduler -> record plan_ready -> relay copy
    MAIN: wait plan_ready (first route consumer) ... MegaMoE ... record consumer_done
    COPY: wait consumer_done -> record copy_idle
    MAIN: wait copy_idle

MAIN is the stream that is current when the layer call starts; it may differ
between calls (a CUDA graph capture stream, for example) but not within one.
Each call therefore forks COPY from MAIN and joins it back, so the whole
call can be captured into a CUDA graph and replayed.

READY generations live on the device. The relay copy publishes the
scheduler's epoch into this rank's READY cell on every rank, and the next
scheduler launch of the same layer drains and zeroes the local READY row
before its EP count exchange. MegaMoE's gate is therefore handed the
constant generation 1, which a replayed graph reproduces exactly.

Helper bank reuse needs no host barrier: COPY of every rank waits for its
own consumer of layer i before it schedules layer i + 1, and the relay of a
later layer writes peers' helper slots only after that layer's EP count
exchange, which every rank enters after its own earlier consumers.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = ["HaloRSlotSchedulerGroup", "build_halo_r_slot_scheduler_group"]

#: Overloaded ranks offer this many of their hottest experts to the planner.
HALO_R_SEND_COUNT = 4
#: Extra weight of extending an already replicated expert over replicating a
#: new one; 0 ranks both by the routed tokens they move.
HALO_R_SEND_WEIGHT = 0
#: CTAs of the relay copy. The scheduler runs on `auxiliary_sms` CTAs, which
#: is also the SM count input quantization leaves free.
HALO_R_RELAY_CTAS = 16
#: The helper slot index is two bits wide.
HALO_R_MAX_HELPER_SLOTS = 4
#: One warp plans for every EP rank.
HALO_R_MAX_EP = 32
#: Exported experts are ordered by an index below 128.
HALO_R_MAX_EXPORTS = 128
#: The scheduler's CTA 0 plans while the other CTAs map routes.
HALO_R_MIN_SCHEDULER_CTAS = 2
#: Plan words of the shared per-layer dynamic EPLB layout (header + 6 fields).
HALO_R_PLAN_WORDS = 2 + 6 * 384
#: Per-broadcast route prefixes use the shared 384-entry field stride.
HALO_R_PREFIX_STRIDE = 384
#: Scheduler status words (see moeRebalanceHaloR.h).
HALO_R_STATUS_WORDS = 8
#: Status word of a grid rendezvous timeout.
HALO_R_STATUS_GRID_TIMEOUT = 1
#: The relay copy accepts at most this many weight planes.
HALO_R_MAX_PLANES = 8
#: The relay splits a weight plane into slices of this size.
HALO_R_RELAY_SLICE_BYTES = 8192
#: Wall-clock bound of each device wait in the scheduler, in nanoseconds. It
#: covers rank skew during startup and autotuning. The READY drain measures it
#: with %globaltimer, so it ends within 100 s, below the MegaMoE READY gate
#: timeout (120 s). The shared passes count SM cycles converted at the maximum
#: SM clock, so they wait at least this long; they end before the routes that
#: MegaMoE consumes are published, so they never overlap its gate.
HALO_R_SPIN_BUDGET_NS = 100_000_000_000
#: Symmetric count-exchange buffer: a 64-word header then two count slabs.
_SYMMETRIC_HEADER_INTS = 64
#: READY cells are uint64.
_READY_CELL_BYTES = 8
#: The MegaMoE gate waits for this generation; the device keeps the rest.
_READY_GENERATION = 1


def _check_driver(result: Any, what: str) -> Any:
    from cuda.bindings import driver

    error = result[0] if isinstance(result, tuple) else result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {error!r}")
    if isinstance(result, tuple) and len(result) > 1:
        return result[1]
    return None


def validate_halo_r_geometry(*, ep_size: int, home_experts: int, helper_slots: int) -> None:
    """Reject geometries the HALO-R kernels cannot plan."""
    if not 2 <= ep_size <= HALO_R_MAX_EP:
        raise ValueError(f"HALO-R supports 2 to {HALO_R_MAX_EP} EP ranks; got {ep_size}")
    if not 1 <= helper_slots <= HALO_R_MAX_HELPER_SLOTS:
        raise ValueError(
            f"HALO-R supports 1 to {HALO_R_MAX_HELPER_SLOTS} helper slots per rank; "
            f"got {helper_slots}"
        )
    if home_experts < HALO_R_SEND_COUNT:
        raise ValueError(
            f"HALO-R needs at least {HALO_R_SEND_COUNT} experts per rank; got {home_experts}"
        )
    if ep_size * HALO_R_SEND_COUNT > HALO_R_MAX_EXPORTS:
        raise ValueError(
            f"HALO-R exports at most {HALO_R_MAX_EXPORTS} experts; "
            f"{ep_size} ranks x {HALO_R_SEND_COUNT} exceed it"
        )
    if ep_size * home_experts > HALO_R_PREFIX_STRIDE:
        raise ValueError(
            f"HALO-R supports at most {HALO_R_PREFIX_STRIDE} experts; got {ep_size * home_experts}"
        )


class HaloRSlotSchedulerGroup:
    """One layer's HALO-R scheduler, relay copy and consumer ordering.

    Exposes the scheduler-group interface used by
    ``PerLayerDynamicMoeLayerLoadBalancer``: ``plan_schedule``,
    ``plan_finish``, ``wait_for_routes``, ``finish`` and ``discard_plan``.
    """

    def __init__(
        self,
        *,
        arena: Any,
        mapping: Any,
        ep_comm: Any,
        device: int,
        home_experts: int,
        helper_slots: int,
        topk: int,
        max_tokens_per_rank: int,
        copy_stream: torch.cuda.Stream,
        auxiliary_sms: int,
        layer_idx: Optional[int],
    ) -> None:
        from cuda.bindings import driver

        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.multirank_dist import (
            FabricSymmetricBuffer,
        )

        ep_size = int(mapping.moe_ep_size)
        ep_rank = int(mapping.moe_ep_rank)
        validate_halo_r_geometry(
            ep_size=ep_size, home_experts=int(home_experts), helper_slots=int(helper_slots)
        )
        if int(topk) <= 0 or int(max_tokens_per_rank) <= 0:
            raise ValueError("topk and max_tokens_per_rank must be positive")
        if copy_stream.device.index != int(device):
            raise ValueError("copy stream device does not match scheduler device")
        sm_count = int(torch.cuda.get_device_properties(int(device)).multi_processor_count)
        if not HALO_R_MIN_SCHEDULER_CTAS <= int(auxiliary_sms) <= sm_count:
            raise ValueError(
                f"HALO-R runs its scheduler on auxiliary_sms CTAs, which must be between "
                f"{HALO_R_MIN_SCHEDULER_CTAS} and the SM count {sm_count}; got {auxiliary_sms}"
            )
        if HALO_R_RELAY_CTAS > sm_count:
            raise ValueError(
                f"HALO-R's relay copy needs {HALO_R_RELAY_CTAS} CTAs; the device has {sm_count} SMs"
            )

        self.layer_idx = layer_idx
        self.device = int(device)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.home_experts = int(home_experts)
        self.helper_slots = int(helper_slots)
        self.experts = self.home_experts * ep_size
        self.topk = int(topk)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.route_capacity = self.max_tokens_per_rank * self.topk
        self.auxiliary_sms = int(auxiliary_sms)
        self.copy_stream = copy_stream
        self.scheduler_stream = copy_stream
        self.arena = arena
        self.plan_calls = 0
        self.symmetric = None
        self._closed = False
        self._validated_streams: set = set()
        self._call_stream_handle: Optional[int] = None
        self._pending_tokens: Optional[int] = None
        self._routes_waited = False
        self._planned = False
        self._part: Optional[Tuple[torch.Tensor, int]] = None

        # Register before any collective so the pool's collective shutdown
        # can close a partially initialized group.
        ep_comm.validate_geometry(expected_rank=ep_rank, expected_world=ep_size)
        self._ep_comm = ep_comm
        pool = getattr(arena.provider, "pool", None)
        if pool is None or pool.comm is not ep_comm:
            raise RuntimeError("rebalance arena and scheduler must use the same EP ProcessGroup")
        arena.provider.register_producer(self)

        self._driver = driver
        self._copy_handle = int(copy_stream.cuda_stream)

        with torch.cuda.device(self.device):
            self._allocate_buffers()
            self._events = {
                name: torch.cuda.Event()
                for name in ("input_ready", "plan_ready", "consumer_done", "copy_idle")
            }
            # Materialize every lazily created event once; the layer path uses
            # its cached driver handle.
            for event in self._events.values():
                event.record(copy_stream)
            self._event_handles = {name: int(e.cuda_event) for name, e in self._events.items()}

            self.symmetric = FabricSymmetricBuffer(
                4 * (_SYMMETRIC_HEADER_INTS + 2 * ep_size * self.experts), self.device
            )
            handles = list(ep_comm.allgather(self.symmetric.shareable))
            if len(handles) != ep_size or len(set(handles)) != ep_size:
                raise RuntimeError(
                    f"HALO-R requires {ep_size} unique EP fabric handles; received "
                    f"{len(handles)} handles with {len(set(handles))} unique values."
                )
            self.peer_bases.copy_(
                torch.tensor(
                    [
                        self.symmetric.ptr
                        if r == ep_rank
                        else self.symmetric.import_peer(handles[r])
                        for r in range(ep_size)
                    ],
                    dtype=torch.int64,
                )
            )
            self._bind_relay_tables(arena, pool)

            # Zero this layer's READY row before any peer can publish into it,
            # then run epoch 1 on every rank so later epochs find a full row.
            arena.terminal_flags_tensor().zero_()
            torch.cuda.synchronize(self.device)
            ep_comm.barrier()
            warmup_routes = torch.empty((0, self.topk), dtype=torch.int32, device=self.device)
            stream = torch.cuda.current_stream(self.device)
            if int(stream.cuda_stream) == self._copy_handle:
                raise RuntimeError("HALO-R warmup must not run on the COPY stream")
            self._submit(warmup_routes, 0, int(stream.cuda_stream))
            self._join(int(stream.cuda_stream), None)
            torch.cuda.synchronize(self.device)
            ep_comm.barrier()

        logger.debug(
            "[MegaMoECuteDsl] layer=%s HALO-R producer initialized: ep=%d helpers=%d scheduler_ctas=%d relay_ctas=%d",
            layer_idx,
            ep_size,
            self.helper_slots,
            self.auxiliary_sms,
            HALO_R_RELAY_CTAS,
        )

    def _allocate_buffers(self) -> None:
        def int32(*shape: int, fill: int = 0) -> torch.Tensor:
            return torch.full(shape, fill, dtype=torch.int32, device=self.device)

        warps = 512 // 32
        bins = self.experts + 1
        ctas = self.auxiliary_sms
        self.physical_slot_ids = int32(self.max_tokens_per_rank, self.topk, fill=-1)
        self.hot_expert_ids = int32(self.helper_slots, fill=-1)
        self.hot_expert_source_ranks = int32(self.helper_slots, fill=-1)
        self.status = int32(HALO_R_STATUS_WORDS)
        self.partial = int32(ctas * self.experts)
        self.route_aux = int32(
            max(ctas * warps * bins, 2 + (ctas - 1) * bins + self.route_capacity)
        )
        self.grid_sync = int32(4)
        self.plan_words = int32(HALO_R_PLAN_WORDS)
        self.route_prefix = int32(HALO_R_PREFIX_STRIDE * (self.ep_size + 1))
        self.peer_bases = torch.zeros(self.ep_size, dtype=torch.int64, device=self.device)
        self.relay_completed_ctas = int32(1)
        self.relay_launch_count = int32(1)

    def _bind_relay_tables(self, arena: Any, pool: Any) -> None:
        """Build the relay's source and destination address tables once."""
        bound = arena.bound.base
        live = arena.arena.base
        planes = len(bound.plane_bytes)
        if not 1 <= planes <= HALO_R_MAX_PLANES:
            raise ValueError(f"HALO-R relays 1 to {HALO_R_MAX_PLANES} weight planes; got {planes}")
        plane_bytes = [int(nbytes) for nbytes in bound.plane_bytes]
        slices = sum(-(-nbytes // HALO_R_RELAY_SLICE_BYTES) for nbytes in plane_bytes)
        if HALO_R_MAX_EXPORTS * slices >= 2**31:
            raise ValueError("HALO-R relay work items exceed int32 indexing")
        sources = [int(address) for address in bound.scatter_source_ptrs]
        if len(sources) != self.experts * planes:
            raise RuntimeError("HALO-R relay source table does not cover every expert plane")
        bank = int(arena.provider.layer.helper_set)
        peer_helpers = pool.peer_helper_bases()
        destinations = []
        for rank in range(self.ep_size):
            for index, (view, nbytes) in enumerate(zip(live.planes, plane_bytes)):
                name = bound.plane_names[index]
                if rank != self.ep_rank and name in pool.helper_bytes:
                    # Peers' unicast views map only home rows; use the mapped
                    # helper chunk, whose row 0 is the peer's slot `home`.
                    base = peer_helpers[(rank, bank, name)] - self.home_experts * nbytes
                else:
                    base = int(view.uc_views[rank].data_ptr())
                destinations.append(base)

        def int64(values: list) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64).to(self.device)

        self.relay_sources = int64(sources)
        self.relay_destinations = int64(destinations)
        self.relay_plane_bytes = int64(plane_bytes)
        self._ready_flags_ptr = int(bound.terminal_local_ptr)
        self._ready_flag_mc_ptr = int(bound.terminal_mc_ptr) + _READY_CELL_BYTES * self.ep_rank

    def close(self) -> None:
        """Release this layer after the owning pool has drained device work."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            if self.symmetric is not None:
                self.symmetric.close()
        self._closed = True
        self.symmetric = None
        self.arena = None
        self._call_stream_handle = None
        self._pending_tokens = None

    def _current_stream_handle(self) -> int:
        if self._closed:
            raise RuntimeError("HALO-R scheduler group is closed")
        if torch.cuda.current_device() != self.device:
            raise RuntimeError("HALO-R submission requires its bound CUDA device")
        stream = torch.cuda.current_stream(self.device)
        handle = int(stream.cuda_stream)
        if handle not in self._validated_streams:
            if handle == self._copy_handle or stream.priority <= self.copy_stream.priority:
                raise RuntimeError("MAIN must be distinct from the higher-priority COPY stream")
            self._validated_streams.add(handle)
        return handle

    def _call_stream(self) -> int:
        handle = self._current_stream_handle()
        if handle != self._call_stream_handle:
            raise RuntimeError("one HALO-R layer call must stay on one stream")
        return handle

    def _record(self, event: str, stream_handle: int) -> None:
        _check_driver(
            self._driver.cuEventRecord(self._event_handles[event], stream_handle),
            "cuEventRecord(HALO-R)",
        )

    def _wait(self, stream_handle: int, event: str) -> None:
        _check_driver(
            self._driver.cuStreamWaitEvent(stream_handle, self._event_handles[event], 0),
            "cuStreamWaitEvent(HALO-R)",
        )

    def _submit(self, routes: torch.Tensor, num_tokens: int, main_handle: int) -> None:
        """Fork COPY from MAIN and enqueue the scheduler and the relay copy."""
        if not torch.cuda.is_current_stream_capturing():
            # A captured route tensor lives in the graph's pool for the graph's lifetime.
            routes.record_stream(self.copy_stream)
        self._record("input_ready", main_handle)
        self._wait(self._copy_handle, "input_ready")
        with torch.cuda.stream(self.copy_stream):
            torch.ops.trtllm.moe_rebalance_halo_r(
                routes,
                self.physical_slot_ids,
                self.hot_expert_ids,
                self.hot_expert_source_ranks,
                self.peer_bases,
                self.status,
                self.partial,
                self.route_aux,
                self.grid_sync,
                self.plan_words,
                self.route_prefix,
                self._ready_flags_ptr,
                self.ep_size,
                self.experts,
                self.helper_slots,
                HALO_R_SEND_COUNT,
                HALO_R_SEND_WEIGHT,
                self.ep_rank,
                self.route_capacity,
                self.auxiliary_sms,
                HALO_R_SPIN_BUDGET_NS,
                num_tokens * self.topk,
            )
            self._record("plan_ready", self._copy_handle)
            torch.ops.trtllm.moe_rebalance_relay(
                self.plan_words,
                self.relay_sources,
                self.relay_destinations,
                self.relay_plane_bytes,
                self.status,
                self.relay_completed_ctas,
                self.relay_launch_count,
                self._ready_flag_mc_ptr,
                self.ep_size,
                self.experts,
                self.ep_rank,
                HALO_R_RELAY_CTAS,
            )

    def _join(self, main_handle: int, completion_event: Optional[torch.cuda.Event]) -> None:
        """Order COPY after the consumer, then join COPY back into MAIN."""
        if completion_event is None:
            self._record("consumer_done", main_handle)
            self._wait(self._copy_handle, "consumer_done")
        else:
            handle = int(completion_event.cuda_event)
            if handle <= 0:
                raise ValueError("completion_event must expose a CUDA event")
            _check_driver(
                self._driver.cuStreamWaitEvent(self._copy_handle, handle, 0),
                "cuStreamWaitEvent(HALO-R consumer)",
            )
        self._record("copy_idle", self._copy_handle)
        self._wait(main_handle, "copy_idle")

    def plan_schedule(self, logical_expert_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Submit contiguous CUDA int32 routes [tokens, topk] without staging.

        Every EP rank must submit each layer call, including ranks without
        tokens: the scheduler's count exchange is collective.
        """
        handle = self._current_stream_handle()
        if self._pending_tokens is not None or self._planned:
            raise RuntimeError("Previous HALO-R plan has no paired finish")
        if (
            not isinstance(logical_expert_ids, torch.Tensor)
            or logical_expert_ids.ndim != 2
            or logical_expert_ids.shape[1] != self.topk
            or logical_expert_ids.dtype != torch.int32
            or not logical_expert_ids.is_contiguous()
            or logical_expert_ids.device != self.physical_slot_ids.device
        ):
            raise ValueError("Routes must be contiguous CUDA int32 [tokens, topk] on this device")
        num_tokens = int(logical_expert_ids.shape[0])
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(f"Route chunk {num_tokens} exceeds {self.max_tokens_per_rank}")
        self._call_stream_handle = handle
        self._submit(logical_expert_ids, num_tokens, handle)
        self._pending_tokens = num_tokens
        self._routes_waited = False
        part = (self.physical_slot_ids, num_tokens)
        self._part = part
        return part

    def plan_finish(
        self, part: Tuple[torch.Tensor, int], *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """Lend routes until finish(); optionally defer their device dependency."""
        self._call_stream()
        if self._pending_tokens is None or part is not self._part:
            raise RuntimeError("plan_finish requires this group's pending plan")
        if not defer_wait:
            self.wait_for_routes()
        physical, num_tokens = part
        self.plan_calls += 1
        self._pending_tokens = None
        self._planned = True
        return physical[:num_tokens], _READY_GENERATION

    def wait_for_routes(self) -> None:
        """Order the first MAIN route read after the scheduler, without a host wait."""
        handle = self._call_stream()
        if not self._routes_waited:
            self._wait(handle, "plan_ready")
            self._routes_waited = True

    def plan(
        self, logical_expert_ids: torch.Tensor, *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        return self.plan_finish(self.plan_schedule(logical_expert_ids), defer_wait=defer_wait)

    def discard_plan(self, part: Tuple[torch.Tensor, int]) -> None:
        """Release an enqueued layer call that has no MegaMoE consumer."""
        self.plan_finish(part, defer_wait=False)
        self.finish()

    def finish(self, completion_event: Optional[torch.cuda.Event] = None) -> None:
        """Queue the consumer wait before the next layer call and join COPY."""
        handle = self._call_stream()
        if not self._planned or not self._routes_waited:
            raise RuntimeError("finish must pair exactly once with a completed plan")
        self._join(handle, completion_event)
        self._planned = False
        self._call_stream_handle = None
        self._part = None


def build_halo_r_slot_scheduler_group(
    *,
    arena: Any,
    manager: Any,
    mapping: Any,
    ep_process_group: Any,
    home_experts: int,
    helper_slots: int,
    topk: int,
    max_tokens_per_rank: int,
    layer_idx: Optional[int],
) -> HaloRSlotSchedulerGroup:
    """Build one layer's HALO-R group from its manager's resources."""
    device = int(mapping.local_rank)
    return HaloRSlotSchedulerGroup(
        arena=arena,
        mapping=mapping,
        ep_comm=manager.get_ep_comm(ep_process_group),
        device=device,
        home_experts=int(home_experts),
        helper_slots=int(helper_slots),
        topk=int(topk),
        max_tokens_per_rank=int(max_tokens_per_rank),
        copy_stream=manager.get_copy_stream(device),
        auxiliary_sms=int(manager.auxiliary_sms),
        layer_idx=layer_idx,
    )
