# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-R planner and relay copy for one per-layer dynamic EPLB layer.

HALO-R is a second producer behind HALO-Q's scheduler-group contract
(``planner.RebalanceSlotSchedulerGroup``): the same generations, route hold,
deferred route wait, lent routes, lease and spin bound, over the same CUDA
scheduler buffers. Only the copy endpoint differs. Its one kernel runs on the
``auxiliary_sms`` CTAs on the shared copy stream (COPY):

    MAIN: ... routes -> record input_ready
    COPY: wait input_ready -> HALO-R: READY drain -> scheduler -> route release -> relay -> READY 1
    MAIN: wait launched (nominally, every HALO-R CTA has begun); wait route counter >= auxiliary_sms, write 0
          (first route consumer) -> MegaMoE
    MAIN: (finish) wait copy_idle, recorded on COPY

The scheduler role plans the layer's replicas and writes every route's
physical slot. Each CTA then adds one to the group's route counter and copies
its share of the relay: every receiver of an expert pulls its share of the
expert from the owner and writes it into the helper slot of every receiver.
The last CTA publishes READY into this rank's cell on every rank, which
MegaMoE's gate waits for. A failing CTA traps before it adds to the route
counter, so MegaMoE never reads routes of a failed launch.

Every value a layer call hands the device is constant: READY is always 1,
because the next launch of the layer waits for every rank's READY and zeroes
its row before its count exchange; the route wait targets ``auxiliary_sms``
and MAIN resets the counter once it passes. A breakable prefill CUDA graph can
therefore capture whole layer calls and replay them in any interleaving with
eager calls: MAIN may change between layer calls (the eager warmup and the
capture of the breakable runner run on its capture stream), and a change
orders the new MAIN after the old one; COPY joins MAIN in every ``finish``.

Helper bank reuse needs no host barrier: every launch follows the MAIN work
queued before it, which includes this rank's consumer of the previous
generation, and its relay writes peers' helper slots only after the all-rank
count exchange, which every rank enters after its own consumers.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from .planner import RebalanceSlotSchedulerGroup, SchedulerSpinBound, _LiveBankLeaseProvider

__all__ = [
    "HaloRRelayEndpoint",
    "HaloRSlotSchedulerGroup",
    "build_halo_r_slot_scheduler_group",
    "validate_halo_r_geometry",
]

#: Overloaded ranks offer this many of their hottest experts to the planner.
HALO_R_SEND_COUNT = 4
#: Extra weight of extending an already replicated expert over replicating a
#: new one; 0 ranks both by the routed tokens they move.
HALO_R_SEND_WEIGHT = 0
#: The helper slot index is two bits wide.
HALO_R_MAX_HELPER_SLOTS = 4
#: One warp plans for every EP rank.
HALO_R_MAX_EP = 32
#: Exported experts are ordered by an index below 128.
HALO_R_MAX_EXPORTS = 128
#: CTA 0 plans while the other CTAs map routes.
HALO_R_MIN_AUXILIARY_SMS = 2
#: Per-broadcast route prefixes use the shared 384-entry field stride.
HALO_R_PREFIX_STRIDE = 384
#: The relay copy accepts at most this many weight planes.
HALO_R_MAX_PLANES = 8
#: The relay splits a weight plane into slices of this size.
HALO_R_RELAY_SLICE_BYTES = 32768
#: Diagnostic counters: launches, receiver copies and receiver skips.
HALO_R_COUNTERS = 3
#: READY cells are uint64.
_READY_CELL_BYTES = 8
#: Every launch publishes this READY generation; see the module docstring.
_READY_GENERATION = 1


def validate_halo_r_geometry(
    *, ep_size: int, home_experts: int, helper_slots: int, auxiliary_sms: int
) -> None:
    """Reject geometries the HALO-R kernel cannot plan."""
    if not 2 <= ep_size <= HALO_R_MAX_EP or ep_size % 2:
        raise ValueError(
            f"HALO-R supports an even number of 2 to {HALO_R_MAX_EP} EP ranks; got {ep_size}"
        )
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
    if auxiliary_sms < HALO_R_MIN_AUXILIARY_SMS:
        raise ValueError(
            f"HALO-R runs on auxiliary_sms CTAs, at least {HALO_R_MIN_AUXILIARY_SMS}; "
            f"got {auxiliary_sms}"
        )


def _relay_tables(arena: Any, pool: Any, *, ep_rank: int, home_experts: int) -> tuple:
    """The relay's source and destination address tables, plane sizes and READY cell."""
    # HALO-R's helper sets have no multicast object, so its provider hands out
    # the base arena without HALO levels; a hierarchical arena has it as ``base``.
    bound = getattr(arena.bound, "base", arena.bound)
    live = getattr(arena.arena, "base", arena.arena)
    planes = len(bound.plane_bytes)
    if not 1 <= planes <= HALO_R_MAX_PLANES:
        raise ValueError(f"HALO-R relays 1 to {HALO_R_MAX_PLANES} weight planes; got {planes}")
    plane_bytes = [int(nbytes) for nbytes in bound.plane_bytes]
    slices = sum(-(-nbytes // HALO_R_RELAY_SLICE_BYTES) for nbytes in plane_bytes)
    if HALO_R_MAX_EXPORTS * slices >= 2**31:
        raise ValueError("HALO-R relay work items exceed int32 indexing")
    sources = [int(address) for address in bound.scatter_source_ptrs]
    experts = len(sources) // planes
    if len(sources) != experts * planes or experts % home_experts:
        raise RuntimeError("HALO-R relay source table does not cover every expert plane")
    # The layer's own helper set with per-layer helper slots, else one of the
    # two shared banks; mapping it is collective on first use.
    peer_helpers = pool.peer_helper_bases(int(arena.provider.layer.helper_set))
    destinations = []
    for rank in range(experts // home_experts):
        for index, (view, nbytes) in enumerate(zip(live.planes, plane_bytes)):
            name = bound.plane_names[index]
            if rank != ep_rank and name in pool.helper_bytes:
                # Peers' unicast views map only home rows; use the mapped
                # helper chunk, whose row 0 is the peer's slot `home`.
                base = peer_helpers[(rank, name)] - home_experts * nbytes
            else:
                base = int(view.uc_views[rank].data_ptr())
            destinations.append(base)
    flag_mc = int(bound.terminal_mc_ptr) + _READY_CELL_BYTES * ep_rank
    return sources, destinations, plane_bytes, int(bound.terminal_local_ptr), flag_mc


class HaloRRelayEndpoint:
    """One layer's native HALO-R context: the copy endpoint of its scheduler group.

    It has the surface of ``HierarchicalSamiWeightBroadcast`` that
    ``RebalanceSlotSchedulerGroup`` and ``_LiveBankLeaseProvider`` call, and
    mirrors its lease bookkeeping: a launch may overwrite the helper slots only
    after the bound reuse authority has proven, for the previous generation,
    that every rank's consumer precedes the next launch. An exception in any
    native call aborts the handoff for good; the owner then rebuilds the group.
    """

    live_bank_count = 1

    def __init__(
        self,
        *,
        scheduler: Any,
        route_counter: torch.Tensor,
        arena: Any,
        pool: Any,
        home_experts: int,
        copy_stream: torch.cuda.Stream,
        helper_residency: bool,
        first_run_spin_cycles: int,
    ) -> None:
        from cuda.bindings import driver

        from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph

        self._driver = driver
        # The bound of each breakable bucket's first eager launch; see submit_fused.
        self._in_breakable_cuda_graph = is_in_breakable_cuda_graph
        self._first_run_spin_cycles = int(first_run_spin_cycles)
        self._breakable_token_counts: set[int] = set()
        self.device = int(scheduler.device.index)
        self._copy_stream = copy_stream
        self._copy_handle = int(copy_stream.cuda_stream)
        scheduler.bind_gpu_direct(self._copy_handle)
        (
            out_slots,
            out_ids,
            _out_levels,
            out_owners,
            peer_bases,
            status,
            partial,
            route_aux,
            grid_sync,
            plan,
            route_prefix,
            ep,
            experts,
            helpers,
            local_rank,
            topk,
            route_capacity,
            ctas,
            _threads,
        ) = scheduler.fused_launch_args()
        sources, destinations, plane_bytes, ready_row, flag_mc = _relay_tables(
            arena, pool, ep_rank=local_rank, home_experts=int(home_experts)
        )
        if len(destinations) != ep * len(plane_bytes):
            raise RuntimeError("HALO-R relay destination table does not cover every EP rank")

        def int64(values: list) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64).to(self.device)

        self.relay_sources = int64(sources)
        self.relay_destinations = int64(destinations)
        self.relay_plane_bytes = int64(plane_bytes)
        self.completed_ctas = torch.zeros(1, dtype=torch.int32, device=self.device)
        self.counters = torch.zeros(HALO_R_COUNTERS, dtype=torch.int64, device=self.device)
        # The expert each rank's helper slot holds, identical on every rank:
        # every rank starts at -1 (unknown) and applies the same plans. Only a
        # layer's own helper slots can hold it, so shared banks keep no table.
        self.residency: Optional[torch.Tensor] = None
        if helper_residency:
            self.residency = torch.full((ep, helpers), -1, dtype=torch.int32, device=self.device)
            torch.cuda.synchronize(self.device)
        # Binds every argument the per-layer launches share and prepares the
        # launch, so a layer call is one native call on the context's handle.
        ctx = torch.ops.trtllm.moe_rebalance_halo_r_create(
            out_slots,
            out_ids,
            out_owners,
            peer_bases,
            status,
            partial,
            route_aux,
            grid_sync,
            plan,
            route_prefix,
            route_counter,
            self.relay_sources,
            self.relay_destinations,
            self.relay_plane_bytes,
            self.completed_ctas,
            self.counters,
            self.residency,
            ready_row,
            flag_mc,
            ep,
            experts,
            helpers,
            HALO_R_SEND_COUNT,
            HALO_R_SEND_WEIGHT,
            local_rank,
            topk,
            route_capacity,
            ctas,
            self._copy_handle,
        )
        try:
            # The per-layer op takes this handle of ctx; passing the
            # custom-class object costs microseconds per call.
            handle = int(torch.ops.trtllm.moe_rebalance_halo_r_handle(ctx))
        except BaseException:
            torch.ops.trtllm.moe_rebalance_halo_r_destroy(ctx)
            raise
        self.ctx = ctx
        self._handle = handle
        self._outputs = scheduler.outputs

        self._lock = threading.Lock()
        self._generation = 1
        self._reuse_authority: Optional[object] = None
        self._active_generation = 0
        self._local_release_generation = 0
        self._collective_safe_generation = 0
        self._handoff_aborted = False
        self._launch_captured = False

    def _require_operational(self) -> None:
        if self._handoff_aborted or self.ctx is None:
            raise RuntimeError("HALO-R relay handoff aborted; quiesced group rebuild required")

    def _validate_reuse_authority(self, authority: object) -> None:
        if authority is None:
            raise TypeError("generation reuse authority cannot be None")
        if getattr(authority, "collective_safe", False) is not True:
            raise ValueError("generation reuse authority lacks collective safety")
        if getattr(authority, "producer_reuse_guarded", False) is not True:
            raise ValueError("generation reuse authority does not guard producer reuse")
        if getattr(authority, "live_bank_count", None) != self.live_bank_count:
            raise ValueError("generation reuse authority must target one live bank")
        if getattr(authority, "bound_copy_module", None) is not self:
            raise ValueError("generation reuse authority is not bound to this HALO-R endpoint")

    def bind_generation_reuse_authority(self, authority: object) -> None:
        """Bind the exact collective single-bank lease provider once."""
        self._validate_reuse_authority(authority)
        with self._lock:
            self._require_operational()
            if self._reuse_authority is not None:
                raise RuntimeError("generation reuse authority is single-assignment")
            if self._active_generation:
                raise RuntimeError("generation reuse authority must bind before submit")
            self._reuse_authority = authority

    def release_generation_after(
        self, generation: int, completion_event: object | None = None
    ) -> None:
        """Register this rank's consumer of ``generation``.

        The consumer is ``completion_event``, or without it the work already
        queued on the current stream, which the next launch follows; only an
        event queues a device wait. The bound reuse authority must still prove
        the all-rank ordering before the next launch.
        """
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        event_handle = 0
        if completion_event is not None:
            event_handle = int(getattr(completion_event, "cuda_event", completion_event))
            if event_handle <= 0:
                raise ValueError("completion_event must expose a positive CUDA event")
        with self._lock:
            self._require_operational()
            if generation != self._active_generation:
                raise ValueError("generation is not the active live bank")
            if self._local_release_generation >= generation:
                raise RuntimeError("generation already has a local release event")
            if completion_event is not None:
                try:
                    (error,) = self._driver.cuStreamWaitEvent(self._copy_handle, event_handle, 0)
                    if error != self._driver.CUresult.CUDA_SUCCESS:
                        raise RuntimeError(f"cuStreamWaitEvent(HALO-R consumer) failed: {error!r}")
                except BaseException:
                    self._handoff_aborted = True
                    raise
            self._local_release_generation = generation

    def mark_collective_reuse_safe(self, authority: object, generation: int) -> None:
        """Accept the bound provider's all-rank proof for one generation."""
        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        with self._lock:
            self._require_operational()
            # bind_generation_reuse_authority() validated this authority once,
            # so identity suffices per layer call.
            if authority is not self._reuse_authority:
                raise PermissionError("generation reuse authority identity mismatch")
            if generation != self._active_generation:
                raise ValueError("generation is not the active live bank")
            if self._local_release_generation != generation:
                raise RuntimeError("local consumer release is not registered")
            if self._collective_safe_generation >= generation:
                raise RuntimeError("generation is already collectively reusable")
            self._collective_safe_generation = generation

    def _require_live_bank_reusable(self) -> None:
        if self._reuse_authority is None:
            raise RuntimeError("single live bank has no bound collective reuse authority")
        if self._active_generation and self._collective_safe_generation < self._active_generation:
            raise RuntimeError("single live bank is still leased by the previous generation")

    def submit_fused(
        self, outputs: object, routes: torch.Tensor, route_target: int, *, spin_cycles: int
    ) -> int:
        """Enqueue the HALO-R kernel on COPY after the work queued on the current stream.

        The current stream must have produced ``routes``. ``route_target`` is
        not used: the kernel only adds one per CTA to the bound route counter
        and never reads it. ``spin_cycles`` bounds every cross-rank and
        cross-CTA wait of the launch. Returns the launched generation, which
        is also the READY value the launch publishes.

        Inside a breakable CUDA graph region, the first eager launch at each
        routed-token count (a bucket's warmup before its capture) waits at
        least the warmup bound. The breakable runner runs that pass under the
        serving timeouts, so that the captured graphs keep them, and a peer's
        one-time work on the breakable path (JIT compilation, module loading)
        can delay a bucket's first run by minutes. A captured launch keeps
        ``spin_cycles``, which its replays reuse, and does not count as a
        run: its kernel runs only on replay. Launches outside a breakable
        region keep ``spin_cycles``.
        """
        del route_target
        rows = int(routes.shape[0])
        with self._lock:
            self._require_operational()
            if outputs is not self._outputs:
                raise RuntimeError("outputs are not from the bound CUDA scheduler")
            self._require_live_bank_reusable()
            self._launch_captured = torch.cuda.is_current_stream_capturing()
            first_run = (
                not self._launch_captured
                and rows not in self._breakable_token_counts
                and self._in_breakable_cuda_graph()
            )
            if first_run:
                spin_cycles = max(int(spin_cycles), self._first_run_spin_cycles)
            try:
                torch.ops.trtllm.moe_rebalance_halo_r_submit_fused(
                    self._handle, routes, spin_cycles
                )
                if first_run:
                    self._breakable_token_counts.add(rows)
                self._active_generation = self._generation
            except BaseException:
                self._handoff_aborted = True
                raise
            generation = self._generation
            self._generation = generation + 1
            return generation

    def launch_capture_ended(self) -> bool:
        """Whether the last launch was captured into a graph whose capture has ended.

        Such a launch runs only when that graph replays, and a failed capture
        discards the graph, so an eager route wait or join for it would wait
        for a kernel that never runs. A layer call stays within one capture,
        so this holds only when a failure ends the capture mid-call.
        """
        return self._launch_captured and not torch.cuda.is_current_stream_capturing()

    def abandon_captured_launch(self) -> None:
        """Abort the handoff after a launch whose capture ended within its layer call."""
        with self._lock:
            self._handoff_aborted = True

    def wait_routes(self, stream_handle: int) -> None:
        """Queue MAIN's route wait on ``stream_handle``: launched, counter >= CTAs, then reset it to 0."""
        with self._lock:
            self._require_operational()
            try:
                torch.ops.trtllm.moe_rebalance_halo_r_wait_routes(self._handle, stream_handle)
            except BaseException:
                self._handoff_aborted = True
                raise

    def join(self) -> None:
        """Order the current stream (MAIN) after everything queued on COPY so far."""
        with self._lock:
            self._require_operational()
            try:
                torch.ops.trtllm.moe_rebalance_halo_r_join(self._handle)
            except BaseException:
                self._handoff_aborted = True
                raise

    def relay_counts(self) -> Tuple[int, int, int]:
        """Return ``(launches, receiver_copies, receiver_skips)`` once COPY is idle.

        The counts cover every launch of the layer, eager and replayed, and
        sum each plan's receivers over its broadcasts. This diagnostic waits
        for the copy stream only, and replayed launches run on the graph's
        own streams: synchronize the device first to count them. It stays out
        of the serving path.
        """
        launches, copies, skips = torch.ops.trtllm.moe_rebalance_halo_r_relay_counts(self.ctx)
        return int(launches), int(copies), int(skips)

    def forget_residency(self) -> None:
        """Mark every helper slot's expert unknown, so the next launch copies every receiver.

        For a weight reload or rebind, between layer calls. It waits for the
        device before and after the reset, so that no earlier launch, eager
        or replayed, can still update the table and every later one reads the
        reset table. The group checks that every rank resets at the same
        point of its layer calls, which keeps the tables identical.
        """
        with self._lock:
            self._require_operational()
            if self.residency is None:
                return
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                self.residency.fill_(-1)
                torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Release the native context after all device work is drained."""
        with self._lock, torch.cuda.device(self.device):
            if self.ctx is None:
                return
            self._handle = 0
            torch.ops.trtllm.moe_rebalance_halo_r_destroy(self.ctx)
            self.ctx = None
            self._outputs = None


class HaloRSlotSchedulerGroup(RebalanceSlotSchedulerGroup):
    """One layer's HALO-R producer behind HALO-Q's scheduler-group contract.

    HALO-Q's ``__init__`` builds the TMA copy endpoint, so this one sets the
    same attributes in the same collective order and builds a
    ``HaloRRelayEndpoint`` instead. For CUDA graphs it overrides four more:
    ``plan_finish`` lends the constant READY generation, ``_wait_route_epoch``
    waits for the constant route target and resets the counter,
    ``_check_owner`` lets MAIN change between layer calls behind a MAIN-to-MAIN
    fence, and ``finish`` joins COPY into MAIN. Every other method is HALO-Q's.
    It adds ``forget_helper_residency`` for weight reloads.
    """

    def forget_helper_residency(self) -> None:
        """Forget which expert each helper slot holds after the expert weights were rewritten.

        The helper slots may still hold copies of the old weights, which the
        next launch would otherwise skip. Every rank's launches derive the
        same need sets from their own tables, so a reset on some ranks only
        would leave helper slices that no rank copies. This is therefore an
        EP collective: every rank must call it for the same layer between the
        same two layer calls, as weight loading does. The ranks compare the
        layer and the count of its resets and raise if any differs. A rank
        that skips the call leaves its peers waiting here, so its own next
        launch traps at the count exchange.
        """
        if self._closed:
            raise RuntimeError("Rebalance scheduler group is closed")
        if self._plan_part is not None or self._generation != self._finished_generation:
            raise RuntimeError("helper residency can be reset only between layer calls")
        if self.broadcaster.residency is not None:
            self._residency_resets += 1
            local = (self.layer_idx, self._residency_resets)
            resets = list(self._ep_comm.allgather(local))
            if any(tuple(entry) != local for entry in resets):
                raise RuntimeError(
                    "HALO-R helper residency resets differ across EP ranks: "
                    f"(layer, resets) per rank {resets}"
                )
        self.broadcaster.forget_residency()

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
        helper_residency: bool,
        spin_bound: SchedulerSpinBound,
        layer_idx: Optional[int],
    ) -> None:
        from cuda.bindings import driver

        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler import (
            CudaPhysicalSlotScheduler,
            CudaSchedulerConfig,
        )
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.multirank_dist import (
            FabricSymmetricBuffer,
        )
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.runtime import (
            symmetric_buffer_ints,
        )

        ep_size = int(mapping.moe_ep_size)
        ep_rank = int(mapping.moe_ep_rank)
        validate_halo_r_geometry(
            ep_size=ep_size,
            home_experts=int(home_experts),
            helper_slots=int(helper_slots),
            auxiliary_sms=int(auxiliary_sms),
        )
        self.layer_idx = layer_idx
        self.device = int(device)
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.home_experts = int(home_experts)
        self.helper_slots = int(helper_slots)
        self.topk = int(topk)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self.auxiliary_sms = int(auxiliary_sms)
        self._spin_bound = spin_bound
        if copy_stream.device.index != self.device:
            raise ValueError("copy stream device does not match scheduler device")
        self.arena = arena
        self.plan_calls = 0
        self._plan_part: Optional[Tuple[Any, int]] = None
        self._generation = 0
        self._pending_release: Optional[Tuple[int, Any]] = None
        self._finished_generation = 0
        self._scheduled_generation = 0
        self._route_wait_generation = 0
        self._routes_in_flight: Optional[torch.Tensor] = None
        self._lent_rows = -1
        self._lent_routes: Optional[torch.Tensor] = None
        self._execution_stream_handle: Optional[int] = None
        # Every stream validated as MAIN, kept alive with its handle.
        self._main_streams: Dict[int, torch.cuda.Stream] = {}
        self._residency_resets = 0
        self._closed = False
        self.scheduler = None
        self.symmetric = None
        self.broadcaster = None
        self.lease = None

        # Register before any collective so the pool's collective shutdown
        # can close a partially initialized group.
        ep_comm.validate_geometry(expected_rank=ep_rank, expected_world=ep_size)
        self._ep_comm = ep_comm
        pool = getattr(arena.provider, "pool", None)
        if pool is None or pool.comm is not ep_comm:
            raise RuntimeError("rebalance arena and scheduler must use the same EP ProcessGroup")
        arena.provider.register_producer(self)

        logical_expert_count = self.home_experts * ep_size
        # HALO-R's scheduler and relay run in the same CTAs, one per auxiliary SM.
        cfg = CudaSchedulerConfig(
            ep_size=ep_size,
            logical_expert_count=logical_expert_count,
            extra_slots_per_rank=self.helper_slots,
            max_tokens_per_rank=self.max_tokens_per_rank,
            topk=self.topk,
            local_rank=ep_rank,
            ctas=self.auxiliary_sms,
        )
        cfg.validate()
        self.cfg = cfg
        self.scheduler = CudaPhysicalSlotScheduler(cfg, device=f"cuda:{self.device}")

        self.copy_stream = copy_stream
        self.scheduler_stream = self.copy_stream
        self._stream_handle = int(self.copy_stream.cuda_stream)
        self._driver = driver
        self._wait_value_geq = driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ
        self._cuda_success = driver.CUresult.CUDA_SUCCESS
        error, cu_device = driver.cuDeviceGet(self.device)
        supported = 0
        if error == driver.CUresult.CUDA_SUCCESS:
            error, supported = driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                cu_device,
            )
        if error != driver.CUresult.CUDA_SUCCESS or not supported:
            raise RuntimeError("MoE rebalance requires 64-bit CUDA stream memory operations")
        # Each CTA of a launch adds one once its routes are written.
        self._route_epoch = torch.zeros(1, dtype=torch.int64, device=f"cuda:{self.device}")
        self._route_epoch_ptr = int(self._route_epoch.data_ptr())
        self._route_target = 0

        # Every peer address must be connected before the first launch, whose
        # count exchange writes into every peer's buffer.
        self.symmetric = FabricSymmetricBuffer(
            4 * symmetric_buffer_ints(ep_size, logical_expert_count), self.device
        )
        handles = list(ep_comm.allgather(self.symmetric.shareable))
        if len(handles) != ep_size or len(set(handles)) != ep_size:
            raise RuntimeError(
                f"MoE rebalance requires {ep_size} unique EP fabric handles; "
                f"received {len(handles)} handles with {len(set(handles))} unique values."
            )
        self.scheduler.connect_peer_bases(
            [
                self.symmetric.ptr if r == ep_rank else self.symmetric.import_peer(handles[r])
                for r in range(ep_size)
            ]
        )
        ep_comm.barrier()

        self.broadcaster = HaloRRelayEndpoint(
            scheduler=self.scheduler,
            route_counter=self._route_epoch,
            arena=arena,
            pool=pool,
            home_experts=self.home_experts,
            copy_stream=self.copy_stream,
            helper_residency=bool(helper_residency),
            first_run_spin_cycles=int(spin_bound.warmup_cycles),
        )

        # Set this layer's READY row to 1, as a launch before the first would
        # leave it, so that every launch drains it before its count exchange,
        # whatever its epoch. The device synchronization keeps a late write
        # from erasing a peer's first READY; the barrier extends it to every
        # rank.
        with torch.cuda.device(self.device):
            arena.terminal_flags_tensor().view(torch.int64).fill_(_READY_GENERATION)
            torch.cuda.synchronize(self.device)
        ep_comm.barrier()

        # The previous consumer precedes the next launch's EP count exchange.
        self.lease = _LiveBankLeaseProvider(self)
        self.broadcaster.bind_generation_reuse_authority(self.lease)

        logger.info(
            "[MegaMoECuteDsl] layer=%s planner=halo_r producer initialized: ep=%d helpers=%d ctas=%d",
            layer_idx,
            ep_size,
            self.helper_slots,
            self.auxiliary_sms,
        )

    def _check_owner(self) -> None:
        """Bind the current stream as MAIN for this layer call.

        HALO-Q binds MAIN once for life. The breakable graph runner warms up
        and captures on its own capture stream and replays on the execution
        stream, so MAIN may change between layer calls, never within one. A
        change orders the new MAIN after the work queued on the old one, whose
        last route-counter reset the next route wait must follow. A change
        into a stream that is capturing raises: an event recorded outside the
        capture cannot be waited on inside it.

        The host sees only layer calls, not graph replays: after a capture,
        MAIN stays the capture stream while the graph replays on the execution
        stream. A later eager call on the capture stream is therefore no
        change and gets no fence, so the caller must order it after the
        replays, as the breakable runner does before every warmup on its
        capture stream (``wait_stream``). Without that order, its route wait
        could pass on a replayed launch's routes.
        """
        if self._closed:
            raise RuntimeError("Rebalance scheduler group is closed")
        if torch._C._cuda_getDevice() != self.device:
            raise RuntimeError("Rebalance submission requires its bound CUDA device")
        handle = torch._C._cuda_getCurrentRawStream(self.device)
        if handle == self._execution_stream_handle:
            return
        if self._plan_part is not None or self._generation != self._finished_generation:
            raise RuntimeError("one HALO-R layer call must stay on one MAIN stream")
        streams = self._main_streams
        stream = streams.get(handle)
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
            if handle == self._stream_handle or stream.priority <= self.copy_stream.priority:
                raise RuntimeError("MAIN must be distinct from the higher-priority COPY stream")
            streams[handle] = stream
        previous = streams.get(self._execution_stream_handle)
        if previous is not None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("HALO-R cannot switch MAIN into a stream that is capturing")
            fence = torch.cuda.Event()
            fence.record(previous)
            stream.wait_event(fence)
        self._execution_stream_handle = handle

    def _wait_route_epoch(self, stream_handle: int) -> None:
        # A constant target and a reset after it, in one batched memory
        # operation, so a captured route wait replays exactly.
        if self.broadcaster.launch_capture_ended():
            # A failure ended the capture of this layer call (the plan is being
            # released by the abort path): the kernel never runs, so its route
            # release never comes. finish() then aborts the endpoint.
            return
        self.broadcaster.wait_routes(stream_handle)

    def plan_finish(
        self, part: Tuple[Any, int], *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """Lend routes until finish() with the constant READY generation."""
        physical, _ = super().plan_finish(part, defer_wait=defer_wait)
        return physical, _READY_GENERATION

    def finish(self, completion_event: torch.cuda.Event | None = None) -> None:
        """Release this generation as HALO-Q does, then join COPY into MAIN.

        A capture must join COPY, which the launch forked natively, and an
        eager call's kernel tail must precede a replay of the same layer.
        MAIN waits only for the tail after READY: MegaMoE, queued before,
        already waited for READY.

        If a failure ended the capture of this layer call, the captured
        kernel never runs: COPY holds nothing to join, the route wait was
        skipped, and the endpoint is aborted, so the group must be rebuilt
        before its next launch.
        """
        capture_ended = self.broadcaster.launch_capture_ended()
        super().finish(completion_event)
        if capture_ended:
            self.broadcaster.abandon_captured_launch()
            return
        self.broadcaster.join()


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
    """Build one layer's HALO-R group from its manager's resources.

    It takes the keywords of ``build_rebalance_slot_scheduler_group`` and the
    same manager resources.
    """
    device = int(mapping.local_rank)
    return HaloRSlotSchedulerGroup(
        arena=arena,
        mapping=mapping,
        ep_comm=manager.get_ep_comm(ep_process_group),
        home_experts=int(home_experts),
        helper_slots=int(helper_slots),
        max_tokens_per_rank=int(max_tokens_per_rank),
        topk=int(topk),
        device=device,
        copy_stream=manager.get_copy_stream(device),
        auxiliary_sms=int(manager.auxiliary_sms),
        helper_residency=bool(manager.helper_residency),
        spin_bound=manager.get_scheduler_spin_bound(),
        layer_idx=layer_idx,
    )
