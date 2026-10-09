# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-main-thread HALO-Q and in-switch TMA weight-copy integration.

One highest-priority stream is shared across layers. plan_schedule() enqueues
HALO-Q, records route completion, and enqueues TMA copy before returning to the
shared-expert hook. plan_finish() can lend routes before the route event;
the first route consumer calls wait_for_routes(). MegaMoE's
system-acquire READY gate protects helper reads while TMA finishes concurrently.

For each rank and generation g:
    MAIN: MegaMoE(g) -> consumer_done(g)
    COPY: wait consumer_done(g) -> HALO-Q(g+1) EP exchange -> TMA(g+1)
All ranks enter the bounded EP exchange after their own previous consumer, so no
next-generation overwrite can precede any previous-generation consumer. No host
collective or auxiliary copy submitter is needed in the serving path.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = ["RebalanceSlotSchedulerGroupV2", "build_rebalance_slot_scheduler_group_v2"]

# Shared-expert persistent grids use the same capacity hint as the copy grid.
TMA_COPY_SM_COUNT = 8


def configured_halo_q_sm_count() -> int:
    value = os.environ.get("TRTLLM_MOE_REBALANCE_CTAS_V2", "").strip()
    count = int(value) if value else TMA_COPY_SM_COUNT
    if count <= 0:
        raise ValueError("HALO-Q SM count must be positive")
    return count


def rebalance_auxiliary_sm_count() -> int:
    # HALO-Q and TMA copy share one stream and cannot run concurrently.
    return max(configured_halo_q_sm_count(), TMA_COPY_SM_COUNT)


_COPY_STREAM: Optional[torch.cuda.Stream] = None


def _global_copy_stream(device: int) -> torch.cuda.Stream:
    """Return the single highest-priority copy stream owned by this process."""
    global _COPY_STREAM
    if _COPY_STREAM is None:
        _, priority = torch.cuda.Stream.priority_range()
        _COPY_STREAM = torch.cuda.Stream(device=device, priority=priority)
    elif _COPY_STREAM.device.index != device:
        raise RuntimeError("Rebalance copy stream cannot span CUDA devices")
    return _COPY_STREAM


def _pin_halo_q_arch(device: int) -> str:
    """Select the local GPU architecture before the first HALO-Q JIT compile.

    The Torch-free scheduler defaults to sm_100. An explicit
    MEGAMOE_HALO_Q_ARCH override takes precedence over local capability.
    """
    env = os.environ.get("MEGAMOE_HALO_Q_ARCH", "").strip()
    if env:
        return env
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    os.environ["MEGAMOE_HALO_Q_ARCH"] = arch
    return arch


class _V2LiveBankLeaseProvider:
    """Authorize ordered reuse, not a claim of completed GPU work on the host.

    release_generation_after() enqueues the local consumer wait on COPY.
    prove() is legal only after the next HALO-Q launch on that same stream.
    Its all-rank count exchange is bounded and precedes plan publication; TMA
    waits for that publication and executes after HALO-Q. Therefore a delayed
    peer consumer blocks every rank's overwrite without a host-side barrier.
    """

    collective_safe = True
    producer_reuse_guarded = True
    live_bank_count = 1

    def __init__(self, group: RebalanceSlotSchedulerGroupV2) -> None:
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


class RebalanceSlotSchedulerGroupV2:
    """One layer's scheduler, copy endpoint, and paired consumer generations."""

    def __init__(
        self,
        *,
        arena: Any,
        mapping: Any,
        device: int,
        home_experts: int,
        helper_slots: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        max_tokens_per_rank: int,
        layer_idx: Optional[int],
    ) -> None:
        from cuda.bindings import driver

        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.cuda_scheduler import (
            CudaPhysicalSlotScheduler,
            CudaSchedulerConfig,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.cuda_scheduler.multirank_dist import (
            FabricSymmetricBuffer,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.cuda_scheduler.runtime import (
            symmetric_buffer_ints,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami import (
            HierarchicalSamiWeightBroadcast,
        )
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami.geometry import (
            BundleLayout,
        )

        from .rebalance_live_arena_v2 import _ep_mpi_comm, _EpComm

        # Before the first CudaPhysicalSlotScheduler is built: the nvcc JIT
        # reads the target once, at first compile, for the whole process.
        _halo_q_arch = _pin_halo_q_arch(int(device))

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
        self.arena = arena
        self.plan_calls = 0
        self._plan_part: Optional[Tuple[Any, int]] = None
        self._generation = 0
        self._pending_release: Optional[Tuple[int, Any]] = None
        self._finished_generation = 0
        self._scheduled_generation = 0
        self._route_wait_generation = 0
        self._owner_thread_id: Optional[int] = None
        self._execution_stream_handle: Optional[int] = None
        self._closed = False
        self.scheduler = None
        self.symmetric = None
        self.broadcaster = None
        self.lease = None

        # Keep a partially initialized producer reachable by the arena's
        # collective shutdown path. Constructor failures must not attempt a
        # rank-local teardown while peers may still be initializing.
        ep_comm = _ep_mpi_comm(mapping)
        self._ep_comm = ep_comm
        arena.provider.register_producer(self)

        logical_expert_count = self.home_experts * ep_size

        # Use the EP rank, not the CUDA ordinal. HALO-Q shares the configured
        # copy-stream launch budget with TMA.
        cfg = CudaSchedulerConfig(
            ep_size=ep_size,
            logical_expert_count=logical_expert_count,
            extra_slots_per_rank=self.helper_slots,
            max_tokens_per_rank=self.max_tokens_per_rank,
            topk=self.topk,
            local_rank=ep_rank,
            ctas=configured_halo_q_sm_count(),
        )
        cfg.validate()
        self.cfg = cfg
        self.scheduler = CudaPhysicalSlotScheduler(cfg, device=f"cuda:{self.device}")

        # All layers share one highest-priority copy stream on this device.
        self.copy_stream = _global_copy_stream(self.device)
        self.scheduler_stream = self.copy_stream
        self._stream_handle = int(self.copy_stream.cuda_stream)
        self._driver = driver
        self._input_ready = torch.cuda.Event()
        self._plan_ready = torch.cuda.Event()
        self._consumer_done = torch.cuda.Event()
        # Materialize each lazily created event once. The steady path uses its
        # cached driver handle, without per-plan stream guards or allocation.
        with torch.cuda.device(self.device):
            for event in (self._input_ready, self._plan_ready, self._consumer_done):
                event.record(self.copy_stream)
        self._input_ready_handle = int(self._input_ready.cuda_event)
        self._plan_ready_handle = int(self._plan_ready.cuda_event)
        self._consumer_done_handle = int(self._consumer_done.cuda_event)

        # Every peer address must be connected before either warmup or submit;
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

        # Warm all ranks before binding the plan channel: a bound launch would
        # leave a pending publication that must be consumed by the broadcaster.
        # The scheduler input is already initialized and peer allocations synchronized.
        self.scheduler.launch(self.scheduler_stream)
        torch.cuda.synchronize(self.device)
        ep_comm.Barrier()

        # The provider must return the existing arena with identical bundle geometry.
        # A second allocation would separate the loader aliases from the copy target.
        bundle = BundleLayout.create(hidden=int(hidden_size), intermediate=int(intermediate_size))
        self.broadcaster = HierarchicalSamiWeightBroadcast(
            comm=_EpComm(ep_comm),
            device=self.device,
            bundle=bundle,
            helper_count=self.helper_slots,
            global_expert_count=logical_expert_count,
            arena_provider=arena.provider,
            copy_backend="tma",
            tma_sm_count=TMA_COPY_SM_COUNT,
            tma_warps=7,
            tma_route="plan",
            tma_plan_mode="gpu_direct",
        )
        self.broadcaster.bind_scheduler(self.scheduler, self.scheduler_stream)

        # A prior consumer event precedes the next HALO-Q EP rendezvous.
        self.lease = _V2LiveBankLeaseProvider(self)
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
        self._owner_thread_id = None
        self._execution_stream_handle = None
        self._plan_part = None
        self._pending_release = None
        self.lease = None
        self.broadcaster = None
        self.scheduler = None
        self.symmetric = None
        self.arena = None

    @property
    def has_submission_owner(self) -> bool:
        """Whether a CPU thread currently owns MAIN submissions."""
        return self._owner_thread_id is not None

    def _check_owner(self) -> None:
        if self._closed:
            raise RuntimeError("Rebalance scheduler group is closed")
        thread_id = threading.get_ident()
        if torch.cuda.current_device() != self.device:
            raise RuntimeError("Rebalance submission requires its bound CUDA device")
        stream = torch.cuda.current_stream(self.device)
        handle = int(stream.cuda_stream)
        if self._owner_thread_id is None:
            if (
                self._execution_stream_handle is not None
                and handle != self._execution_stream_handle
            ):
                raise RuntimeError("Rebalance MAIN stream cannot change during owner handoff")
            if handle == self._stream_handle or stream.priority <= self.copy_stream.priority:
                raise RuntimeError("MAIN must be distinct from the higher-priority COPY stream")
            self._owner_thread_id = thread_id
            self._execution_stream_handle = handle
        elif thread_id != self._owner_thread_id or handle != self._execution_stream_handle:
            raise RuntimeError("Rebalance must be submitted by one MAIN thread and stream")

    def release_warmup_owner_after_sync(self, execution_stream_handle: int) -> None:
        """Release the startup CPU owner after the caller drains this device.

        Keep the fixed MAIN stream, generation counters, and pending lease
        release. The first serving submission binds its CPU owner normally.
        """
        if self._closed:
            raise RuntimeError("Rebalance scheduler group is closed")
        if self._owner_thread_id != threading.get_ident():
            raise RuntimeError("Only the warmup submitter may release its owner")
        if self._execution_stream_handle != int(execution_stream_handle):
            raise RuntimeError("Warmup owner handoff requires the bound MAIN stream")
        if (
            self._plan_part is not None
            or self._generation != self._finished_generation
            or self._generation != self.plan_calls
        ):
            raise RuntimeError("Warmup owner handoff requires all plans to be finished")
        self._owner_thread_id = None

    def _record_event(self, event_handle: int, stream_handle: int) -> None:
        (error,) = self._driver.cuEventRecord(event_handle, stream_handle)
        if error != self._driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuEventRecord(rebalance) failed: {error!r}")

    def _wait_event(self, stream_handle: int, event_handle: int) -> None:
        (error,) = self._driver.cuStreamWaitEvent(stream_handle, event_handle, 0)
        if error != self._driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamWaitEvent(rebalance) failed: {error!r}")

    def plan_schedule(self, logical_expert_ids: torch.Tensor) -> Tuple[Any, int]:
        """Submit contiguous CUDA int32 routes [tokens, topk] without staging.

        The route producer must run on MAIN and must not overwrite this tensor
        until HALO-Q finishes. record_stream protects allocator reuse while COPY
        consumes the input. Empty ranks still submit and join the EP exchange.
        """
        self._check_owner()
        if self._plan_part is not None or self._generation != self._finished_generation:
            raise RuntimeError("Previous rebalance plan has no paired finish")
        if (
            not isinstance(logical_expert_ids, torch.Tensor)
            or logical_expert_ids.ndim != 2
            or logical_expert_ids.shape[1] != self.topk
            or logical_expert_ids.dtype != torch.int32
            or not logical_expert_ids.is_contiguous()
            or logical_expert_ids.device != self.scheduler.device
        ):
            raise ValueError("Routes must be contiguous CUDA int32 [tokens, topk] on this device")
        num_tokens = int(logical_expert_ids.shape[0])
        if num_tokens > self.max_tokens_per_rank:
            raise RuntimeError(f"Route chunk {num_tokens} exceeds {self.max_tokens_per_rank}")

        # Both waits capture their event's current record before a later plan
        # can re-record it. No host synchronization or temporary event is needed.
        logical_expert_ids.record_stream(self.copy_stream)
        self._record_event(self._input_ready_handle, self._execution_stream_handle)
        self._wait_event(self._stream_handle, self._input_ready_handle)
        outputs = self.scheduler.submit(logical_expert_ids, self._stream_handle)
        self._scheduled_generation = self._generation + 1
        self._record_event(self._plan_ready_handle, self._stream_handle)
        part = (outputs, num_tokens)
        self._plan_part = part
        if self._pending_release is not None:
            self.lease.prove(self._pending_release[0])
            self._pending_release = None
        self._submit_copy(outputs)
        return part

    def _submit_copy(self, outputs: object) -> None:
        """Consume the HALO-Q plan and enqueue its TMA kernel on COPY."""
        ticket = self.broadcaster.submit(outputs)
        if int(ticket.generation) != self._scheduled_generation:
            raise RuntimeError("TMA and HALO-Q generation mismatch")
        self._generation = int(ticket.generation)

    def plan_finish(
        self, part: Tuple[Any, int], *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        """Lend routes until finish(); optionally defer their device dependency."""
        self._check_owner()
        if part is not self._plan_part or part is None:
            raise RuntimeError("plan_finish requires this group's pending plan")
        outputs, num_tokens = part
        if not defer_wait:
            self.wait_for_routes()
        physical = outputs.physical_slot_ids[:num_tokens]
        self.plan_calls += 1
        self._plan_part = None
        return physical, self._generation

    def wait_for_routes(self) -> None:
        """Order the first MAIN route read after HALO-Q, without a host wait."""
        self._check_owner()
        if self._generation <= 0:
            raise RuntimeError("No scheduler generation is available")
        if self._route_wait_generation != self._generation:
            with torch.cuda.nvtx.range("rebal/route_wait"):
                self._wait_event(self._execution_stream_handle, self._plan_ready_handle)
            self._route_wait_generation = self._generation

    def plan(
        self, logical_expert_ids: torch.Tensor, *, defer_wait: bool = False
    ) -> Tuple[torch.Tensor, int]:
        return self.plan_finish(self.plan_schedule(logical_expert_ids), defer_wait=defer_wait)

    def discard_plan(self, part: Tuple[Any, int]) -> None:
        """Release an enqueued generation that has no MegaMoE consumer."""
        self.plan_finish(part, defer_wait=False)
        self.finish()

    def finish(self, completion_event: torch.cuda.Event | None = None) -> None:
        """Queue the previous consumer wait before any next-generation producer."""
        self._check_owner()
        if (
            self._plan_part is not None
            or self._generation <= 0
            or self._generation != self.plan_calls
            or self._route_wait_generation != self._generation
            or self._generation != self._finished_generation + 1
        ):
            raise RuntimeError("finish must pair exactly once with a completed plan")
        if completion_event is None:
            completion_event = self._consumer_done
            self._record_event(self._consumer_done_handle, self._execution_stream_handle)
        self.broadcaster.release_generation_after(self._generation, completion_event)
        self._pending_release = (self._generation, completion_event)
        self._finished_generation = self._generation


def build_rebalance_slot_scheduler_group_v2(backend: Any) -> RebalanceSlotSchedulerGroupV2:
    """Build the layer's scheduler/copy group after its live arena exists."""
    arena = getattr(backend, "_rebalance_arena", None)
    if arena is None:
        raise RuntimeError(
            "MoE rebalance requires create_weights to allocate the live arena "
            "before building the producer group."
        )
    home = getattr(backend, "_rebalance_home_experts", None)
    if type(home) is not int or home <= 0:
        raise RuntimeError(
            f"MoE rebalance requires a positive resident expert count; got {home!r}."
        )
    return RebalanceSlotSchedulerGroupV2(
        arena=arena,
        mapping=backend.mapping,
        home_experts=home,
        helper_slots=int(backend._rebalance_slots_active),
        hidden_size=int(backend.hidden_size),
        intermediate_size=int(backend.intermediate_size_per_partition),
        max_tokens_per_rank=int(backend._maxt_buckets[-1]),
        topk=int(backend.routing_method.experts_per_token),
        device=int(backend.mapping.local_rank),
        layer_idx=getattr(backend, "layer_idx", None),
    )


_PLAN_GAP_HOOK_ATTR = "_rebalance_plan_gap_hook"


def _resident_slot_ids(
    token_selected_slots: torch.Tensor,
    *,
    home_experts: int,
    helper_slots: int,
) -> torch.Tensor:
    """Map logical IDs to the widened resident slots while preserving padding."""
    ids = token_selected_slots
    slot_count = home_experts + helper_slots
    logical = ids.long()
    owner = torch.div(logical, home_experts, rounding_mode="floor")
    physical = owner * slot_count + (logical - owner * home_experts)
    return torch.where(logical < 0, logical, physical).to(ids.dtype)


def apply_rebalance_scheduler(
    moe: Any,
    token_selected_slots: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Produce physical routes and enqueue the matching helper-weight copy."""
    backend = getattr(moe, "backend", None)
    helper_slots = getattr(backend, "_rebalance_slots_active", 0)
    if type(helper_slots) is not int or helper_slots <= 0:
        return token_selected_slots
    if getattr(moe, "layer_load_balancer", None) is not None:
        raise RuntimeError(
            "MoE rebalance helper slots and EPLB cannot both remap the "
            "routing tensor: layer_load_balancer is not None while S="
            f"{helper_slots} > 0 (layer_idx={getattr(moe, 'layer_idx', None)})."
        )

    group = getattr(backend, "_rebalance_scheduler_group", None)
    if group is None:
        raise RuntimeError(
            f"MoE rebalance helper slots are live (S={helper_slots}, "
            f"layer_idx={getattr(moe, 'layer_idx', None)}) but the scheduler "
            "group is absent. Set TRTLLM_MOE_REBALANCE_DISABLE=1 to run "
            "without helper slots."
        )
    if not backend.is_rebalance_active():
        home_experts = getattr(backend, "_rebalance_home_experts", None)
        if type(home_experts) is not int or home_experts <= 0:
            raise RuntimeError(
                "MoE rebalance bypass requires a positive resident expert count; "
                f"got {home_experts!r}."
            )
        backend._rebalance_plan_ran = False
        return _resident_slot_ids(
            token_selected_slots,
            home_experts=home_experts,
            helper_slots=helper_slots,
        )

    backend._rebalance_plan_ran = True
    gap_hook = getattr(moe, _PLAN_GAP_HOOK_ATTR, None)
    gap_hook_owner = moe
    backend_gap_hook = getattr(backend, _PLAN_GAP_HOOK_ATTR, None)
    if backend_gap_hook is not None:
        if gap_hook is not None:
            raise RuntimeError(
                "MoE rebalance found shared-expert hooks on both wrapper and backend"
            )
        gap_hook, gap_hook_owner = backend_gap_hook, backend
    if gap_hook is None:
        physical_slot_ids, generation = group.plan(token_selected_slots, defer_wait=True)
    else:
        part = group.plan_schedule(token_selected_slots)
        setattr(gap_hook_owner, _PLAN_GAP_HOOK_ATTR, None)
        try:
            gap_hook()
        except BaseException as error:  # noqa: BLE001
            try:
                group.discard_plan(part)
                backend._rebalance_plan_ran = False
                cleanup_note = "the unused rebalance generation was safely released."
            except BaseException as cleanup_error:  # noqa: BLE001
                cleanup_note = (
                    "rebalance generation cleanup failed and the group requires teardown: "
                    f"{cleanup_error!r}"
                )
            notes = getattr(error, "__notes__", None)
            if type(notes) is not list:
                notes = []
                error.__notes__ = notes
            notes.append("Shared-expert hook failed after HALO-Q/TMA enqueue; " + cleanup_note)
            raise
        physical_slot_ids, generation = group.plan_finish(part, defer_wait=True)
    backend._rebalance_generation = int(generation)
    return physical_slot_ids


def bind_live_weight_planes(quant_method: Any, module: Any) -> None:
    """Alias MegaMoE's seven live parameters onto its rebalance arena."""
    arena = getattr(module, "_rebalance_arena", None)
    if arena is None:
        return
    from ....cute_dsl_kernels.megamoe_scheduler_v2.integrations.megamoe.direct_live_weight_bridge import (
        CANONICAL_WEIGHT_PLANE_NAMES,
    )

    if tuple(arena.plane_names) != CANONICAL_WEIGHT_PLANE_NAMES:
        raise RuntimeError("MoE rebalance arena plane order is not canonical")
    rebound = False
    for name, arena_view, alias in zip(
        arena.plane_names, arena.local_plane_views, arena.tekit_alias_views
    ):
        old = getattr(module, name, None)
        if old is None:
            raise RuntimeError(f"MoE rebalance live weight plane {name!r} was not registered")
        if (
            tuple(alias.shape) != tuple(old.shape)
            or tuple(alias.stride()) != tuple(old.stride())
            or alias.dtype != old.dtype
        ):
            raise RuntimeError(f"MoE rebalance arena alias for {name!r} has an incompatible layout")
        if not alias.is_cuda or int(alias.data_ptr()) != int(arena_view.data_ptr()):
            raise RuntimeError(f"MoE rebalance arena alias for {name!r} is not its allocator view")
        if old.is_meta:
            setattr(
                module,
                name,
                torch.nn.Parameter(alias, requires_grad=old.requires_grad),
            )
            rebound = True
        else:
            try:
                old.data = alias
            except RuntimeError as error:
                raise RuntimeError(
                    f"MoE rebalance failed to bind live plane {name!r}: {error}"
                ) from error
    if rebound:
        quant_method.setup_quant_scales(module)
    norm_const = getattr(module, "fc1_norm_const", None)
    if norm_const is not None and arena.home_experts < norm_const.data.shape[0]:
        norm_const.data[int(arena.home_experts) :].fill_(1.0)
