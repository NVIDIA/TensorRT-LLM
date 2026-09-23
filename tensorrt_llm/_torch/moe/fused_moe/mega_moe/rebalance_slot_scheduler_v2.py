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
        ep_process_group: Any = None,
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

        logical_expert_count = self.home_experts * ep_size

        # Use the EP rank, not the CUDA ordinal. HALO-Q defaults to the same
        # eight-SM budget as TMA; an override may reduce that budget.
        ctas_env = os.environ.get("TRTLLM_MOE_REBALANCE_CTAS_V2", "").strip()
        cfg = CudaSchedulerConfig(
            ep_size=ep_size,
            logical_expert_count=logical_expert_count,
            extra_slots_per_rank=self.helper_slots,
            max_tokens_per_rank=self.max_tokens_per_rank,
            topk=self.topk,
            local_rank=ep_rank,
            ctas=(int(ctas_env) if ctas_env else TMA_COPY_SM_COUNT),
            # HALO-Q is the supported scheduler algorithm.
            algorithm=os.environ.get("TRTLLM_MOE_REBALANCE_ALGO", "halo_q"),
        )
        cfg.validate()
        if cfg.ctas > TMA_COPY_SM_COUNT:
            raise ValueError("HALO-Q CTAs exceed the shared-expert SM reservation")
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

        ep_comm = _ep_mpi_comm(mapping)
        self._ep_comm = ep_comm

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

        logger.info(
            f"[MegaMoECuteDsl] layer={layer_idx} rebalance producer group up: "
            f"HALO-Q algorithm={cfg.algorithm} ctas={cfg.ctas} "
            f"arch={_halo_q_arch} + "
            f"TMA in-switch copy sm={TMA_COPY_SM_COUNT} pairs=7 "
            f"route=plan gpu-direct single-main levels={len(arena.provider.group_sizes)} "
            f"groups={arena.provider.group_sizes}, EP={ep_size} rank={ep_rank} "
            f"H={home_experts} S={helper_slots} maxT={max_tokens_per_rank}"
        )

    def _check_owner(self) -> None:
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
        # Reuse: COPY waits for the consumer event, then the next HALO-Q EP exchange.
        ep_process_group=getattr(backend, "_ep_pg", None),
    )
