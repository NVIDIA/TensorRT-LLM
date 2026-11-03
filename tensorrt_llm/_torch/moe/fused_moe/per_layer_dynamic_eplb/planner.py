# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HALO-Q and in-switch TMA weight-copy integration on one MAIN stream.

One highest-priority stream is shared across layers. plan_schedule() enqueues
one kernel that runs HALO-Q and then the TMA copy of its plan on the auxiliary
SMs, before returning to the shared-expert hook. Each CTA of that kernel raises
a device route epoch once its routes are written. plan_finish() can lend routes
before they exist; the first route consumer calls wait_for_routes(), a stream
memory wait on that epoch that holds no SM. MegaMoE's system-acquire READY gate
protects helper reads while the copy finishes concurrently.

For each rank and generation g:
    MAIN: MegaMoE(g) -> ... -> routes(g+1) -> input_ready(g+1)
    COPY: wait input_ready(g+1) -> [HALO-Q(g+1) EP exchange -> TMA(g+1)]
Every launch orders COPY after the MAIN work queued before it, which includes
the previous consumer MegaMoE(g), so finish() queues no device work of its own.
All ranks enter the bounded EP exchange after their own previous consumer, so no
next-generation overwrite can precede any previous-generation consumer. No host
collective or auxiliary copy submitter is needed in the serving path.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "RebalanceSlotSchedulerGroup",
    "SchedulerSpinBound",
    "build_rebalance_slot_scheduler_group",
]


class SchedulerSpinBound:
    """Bound of the HALO-Q scheduler's cross-rank waits, chosen per launch.

    HALO-Q exchanges route counts with every EP rank inside the kernel and
    traps once a peer is later than this bound. During warmup, one-time JIT
    compilation and autotuning can delay a healthy rank by minutes. As a MoE
    communication timeout proxy, the bound follows every warmup transition:
    warmup gets the warmup timeout, and serving the serving timeout or the
    scheduler's native ``SPIN_CYCLES``. Launches while ``is_tuning`` returns
    True, such as AutoTuner profiling outside warmup, also get the warmup
    timeout. One bound serves every layer of a model.

    Args:
        warmup_seconds: The MoE communication warmup timeout.
        is_tuning: Returns whether kernels are being profiled.
    """

    name = "per-layer dynamic EPLB HALO-Q"

    def __init__(self, *, warmup_seconds: int, is_tuning: Callable[[], bool]) -> None:
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.cuda_scheduler.runtime import (
            spin_cycles_for_timeout,
        )

        self._spin_cycles_for_timeout = spin_cycles_for_timeout
        self._is_tuning = is_tuning
        self.warmup_cycles = spin_cycles_for_timeout(warmup_seconds)
        self._timeout_cycles = spin_cycles_for_timeout(None)

    def set_timeout_seconds(self, seconds: Optional[int]) -> None:
        """Apply a timeout to the launches issued after this call; ``None`` restores the default."""
        self._timeout_cycles = self._spin_cycles_for_timeout(seconds)

    def cycles(self) -> int:
        """Return the bound of the next launch."""
        if self._is_tuning():
            return self.warmup_cycles
        return self._timeout_cycles


class _LiveBankLeaseProvider:
    """Authorize ordered reuse, not a claim of completed GPU work on the host.

    release_generation_after() registers the local consumer; the next fused
    HALO-Q + TMA launch orders COPY after it on MAIN. prove() is legal right
    before that launch. The launch's bounded all-rank count exchange precedes its plan,
    and its copy follows the plan in the same CTAs. Therefore a delayed peer
    consumer blocks every rank's overwrite without a host-side barrier.
    """

    collective_safe = True
    producer_reuse_guarded = True
    live_bank_count = 1

    def __init__(self, group: RebalanceSlotSchedulerGroup) -> None:
        self.bound_copy_module = group.broadcaster
        self._group = group
        self.proved_generation = 0

    def prove(self, generation: int) -> None:
        group = self._group
        if (
            generation != group._finished_generation
            or group._pending_release is None
            or group._pending_release[0] != generation
            or group._scheduled_generation != generation + 1
            or group.scheduler_stream is not group.copy_stream
        ):
            raise RuntimeError("Live-bank reuse lacks the consumer/HALO-Q ordering chain")
        self.bound_copy_module.mark_collective_reuse_safe(self, generation)
        self.proved_generation = generation


class RebalanceSlotSchedulerGroup:
    """One layer's scheduler, copy endpoint, and paired consumer generations."""

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
        from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami import (
            HierarchicalSamiWeightBroadcast,
        )

        ep_size = int(mapping.moe_ep_size)
        ep_rank = int(mapping.moe_ep_rank)
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
        # Routes of the latest launch, held until MAIN waits for its route epoch.
        self._routes_in_flight: Optional[torch.Tensor] = None
        # The lent physical routes view and its row count: every layer of a
        # forward plans the same number of tokens, so one view serves them.
        self._lent_rows = -1
        self._lent_routes: Optional[torch.Tensor] = None
        self._execution_stream_handle: Optional[int] = None
        self._closed = False
        self.scheduler = None
        self.symmetric = None
        self.broadcaster = None
        self.lease = None

        # Keep a partially initialized producer reachable by the arena's
        # collective shutdown path. Constructor failures must not attempt a
        # rank-local teardown while peers may still be initializing.
        ep_comm.validate_geometry(expected_rank=ep_rank, expected_world=ep_size)
        self._ep_comm = ep_comm
        pool = getattr(arena.provider, "pool", None)
        if pool is None or pool.comm is not ep_comm:
            raise RuntimeError("rebalance arena and scheduler must use the same EP ProcessGroup")
        arena.provider.register_producer(self)

        logical_expert_count = self.home_experts * ep_size

        # Use the EP rank, not the CUDA ordinal. HALO-Q and the TMA copy run in
        # the same CTAs, one per auxiliary SM.
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

        # All layers share the model manager's highest-priority copy stream.
        self.copy_stream = copy_stream
        self.scheduler_stream = self.copy_stream
        self._stream_handle = int(self.copy_stream.cuda_stream)
        self._driver = driver
        # The route wait runs once per layer; resolve its driver constants once.
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
        # CTAs of the fused launches whose routes are written, and its value
        # after the latest launch. Only those launches change it.
        self._route_epoch = torch.zeros(1, dtype=torch.int64, device=f"cuda:{self.device}")
        self._route_epoch_ptr = int(self._route_epoch.data_ptr())
        self._route_target = 0

        # Every peer address must be connected before the first fused launch;
        # HALO-Q performs an all-rank exchange on the COPY stream.
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

        # connect_peer_bases() synchronized this device, so its zeroed exchange
        # buffer and peer mappings are complete. The barrier extends that to every
        # rank before any rank binds its copy and can launch the fused kernel,
        # whose HALO-Q exchange writes into its peers' buffers.
        ep_comm.barrier()

        # The provider must return the existing arena with identical bundle geometry.
        # A second allocation would separate the loader aliases from the copy target.
        bundle = arena.bundle
        self.broadcaster = HierarchicalSamiWeightBroadcast(
            comm=ep_comm,
            device=self.device,
            bundle=bundle,
            helper_count=self.helper_slots,
            global_expert_count=logical_expert_count,
            arena_provider=arena.provider,
            tma_sm_count=self.auxiliary_sms,
            tma_warps=7,
            tma_route="plan",
            helper_residency=bool(helper_residency),
        )
        self.broadcaster.bind_scheduler(self.scheduler, self.scheduler_stream, self._route_epoch)

        # The previous consumer precedes the next HALO-Q EP rendezvous.
        self.lease = _LiveBankLeaseProvider(self)
        self.broadcaster.bind_generation_reuse_authority(self.lease)

        logger.debug("[MegaMoECuteDsl] layer=%s rebalance producer initialized", layer_idx)

    def close(self) -> None:
        """Release this layer after the owning pool has drained device work."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            if self.broadcaster is not None:
                self.broadcaster.close()
            if self.symmetric is not None:
                self.symmetric.close()
        self._closed = True
        self._execution_stream_handle = None
        self._plan_part = None
        self._pending_release = None
        self._routes_in_flight = None
        self._lent_rows = -1
        self._lent_routes = None
        self.lease = None
        self.broadcaster = None
        self.scheduler = None
        self.symmetric = None
        self.arena = None

    def _check_owner(self) -> None:
        if self._closed:
            raise RuntimeError("Rebalance scheduler group is closed")
        # Runs several times per layer: read the raw device and stream instead
        # of building a torch.cuda.Stream, and return early for the bound MAIN.
        if torch._C._cuda_getDevice() != self.device:
            raise RuntimeError("Rebalance submission requires its bound CUDA device")
        handle = torch._C._cuda_getCurrentRawStream(self.device)
        if handle == self._execution_stream_handle:
            return
        if self._execution_stream_handle is not None:
            raise RuntimeError(
                "Rebalance MAIN stream cannot change; every executor of a model "
                "engine must submit on its one execution stream"
            )
        stream = torch.cuda.current_stream(self.device)
        if handle == self._stream_handle or stream.priority <= self.copy_stream.priority:
            raise RuntimeError("MAIN must be distinct from the higher-priority COPY stream")
        self._execution_stream_handle = handle

    def _wait_route_epoch(self, stream_handle: int) -> None:
        (error,) = self._driver.cuStreamWaitValue64(
            stream_handle, self._route_epoch_ptr, self._route_target, self._wait_value_geq
        )
        if error != self._cuda_success:
            raise RuntimeError(f"cuStreamWaitValue64(rebalance) failed: {error!r}")

    def plan_schedule(self, logical_expert_ids: torch.Tensor) -> Tuple[Any, int]:
        """Submit contiguous CUDA int32 routes [tokens, topk] without staging.

        The route producer must allocate and write this tensor on MAIN and must
        not overwrite it until HALO-Q finishes. The group holds the tensor until
        wait_for_routes() orders MAIN after the route epoch. Every HALO-Q CTA
        reads its routes before it raises that epoch, so the caching allocator
        can only reuse the block for MAIN work queued after that wait. This needs
        no COPY-stream allocator event, which every later allocation would poll
        while COPY runs behind MAIN. Empty ranks still submit and join the EP
        exchange.
        """
        self._check_owner()
        if self._plan_part is not None or self._generation != self._finished_generation:
            raise RuntimeError("Previous rebalance plan has no paired finish")
        if self._routes_in_flight is not None:
            # Only a failed submission leaves routes that MAIN never waited for.
            raise RuntimeError("Routes of an earlier HALO-Q launch were never ordered on MAIN")
        if (
            not isinstance(logical_expert_ids, torch.Tensor)
            or logical_expert_ids.ndim != 2
            or logical_expert_ids.shape[1] != self.topk
            or logical_expert_ids.dtype != torch.int32
            or not logical_expert_ids.is_contiguous()
            # -1 for a CPU tensor; cheaper than building and comparing torch.device objects.
            or logical_expert_ids.get_device() != self.device
        ):
            raise ValueError("Routes must be contiguous CUDA int32 [tokens, topk] on this device")
        num_tokens = int(logical_expert_ids.shape[0])
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(f"Route chunk {num_tokens} exceeds {self.max_tokens_per_rank}")

        outputs = self.scheduler.outputs
        self._scheduled_generation = self._generation + 1
        if self._pending_release is not None:
            self.lease.prove(self._pending_release[0])
            self._pending_release = None
        route_target = self._route_target + self.auxiliary_sms
        self._routes_in_flight = logical_expert_ids
        # The launch follows the MAIN work queued so far, which wrote the
        # routes; no host synchronization is needed.
        generation = self.broadcaster.submit_fused(
            outputs,
            logical_expert_ids,
            route_target,
            spin_cycles=self._spin_bound.cycles(),
        )
        if generation != self._scheduled_generation:
            raise RuntimeError("TMA and HALO-Q generation mismatch")
        self._route_target = route_target
        self._generation = generation
        part = (outputs, num_tokens)
        self._plan_part = part
        return part

    def plan_finish(
        self, part: Tuple[Any, int], *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """Lend routes until finish(); optionally defer their device dependency.

        It queues no device work itself: plan_schedule() checked the owner of
        this pending plan, and the route wait checks it again.
        """
        if part is not self._plan_part or part is None:
            raise RuntimeError("plan_finish requires this group's pending plan")
        outputs, num_tokens = part
        if not defer_wait:
            self.wait_for_routes()
        physical = self._lent_routes
        if num_tokens != self._lent_rows:
            # A view of the scheduler's fixed output buffer; slicing it is
            # the cost, so reuse it while the token count repeats.
            physical = self._lent_routes = outputs.physical_slot_ids[:num_tokens]
            self._lent_rows = num_tokens
        self.plan_calls += 1
        self._plan_part = None
        return physical, self._generation

    def wait_for_routes(self) -> None:
        """Order the first MAIN route read after HALO-Q, without a host wait.

        The first call of a generation checks the owner and queues the stream
        wait. Later calls of that generation return at once: their consumers
        are already ordered after it on MAIN.
        """
        generation = self._generation
        if generation > 0 and generation == self._route_wait_generation and not self._closed:
            return
        self._check_owner()
        if generation <= 0:
            raise RuntimeError("No scheduler generation is available")
        self._wait_route_epoch(self._execution_stream_handle)
        self._route_wait_generation = generation
        # MAIN work queued from here on runs after every HALO-Q read of the routes.
        self._routes_in_flight = None

    def plan(
        self, logical_expert_ids: torch.Tensor, *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        return self.plan_finish(self.plan_schedule(logical_expert_ids), defer_wait=defer_wait)

    def discard_plan(self, part: Tuple[Any, int]) -> None:
        """Release an enqueued generation that has no MegaMoE consumer."""
        self.plan_finish(part, defer_wait=False)
        self.finish()

    def finish(self, completion_event: torch.cuda.Event | None = None) -> None:
        """Release this generation to the next-generation producer.

        The consumer is ``completion_event``, or without it the MAIN work
        queued so far: the MegaMoE launch that read this generation, if any.
        The next fused launch, submitted on MAIN, orders COPY after it.
        """
        self._check_owner()
        if (
            self._plan_part is not None
            or self._generation <= 0
            or self._generation != self.plan_calls
            or self._route_wait_generation != self._generation
            or self._generation != self._finished_generation + 1
        ):
            raise RuntimeError("finish must pair exactly once with a completed plan")
        self.broadcaster.release_generation_after(self._generation, completion_event)
        self._pending_release = (self._generation, completion_event)
        self._finished_generation = self._generation


def build_rebalance_slot_scheduler_group(
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
) -> RebalanceSlotSchedulerGroup:
    """Build one layer's scheduler/copy group from its manager's resources.

    The manager supplies the model-owned EP communicator, copy stream,
    auxiliary SM budget, helper-slot mode and scheduler spin bound; the
    communicator is resolved before the stream.
    """
    device = int(mapping.local_rank)
    return RebalanceSlotSchedulerGroup(
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
