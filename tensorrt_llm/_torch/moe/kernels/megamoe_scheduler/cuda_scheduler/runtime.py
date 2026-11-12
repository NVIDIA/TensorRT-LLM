# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host runtime for the HALO-Q scheduler, launched with its GPU-direct TMA copy as one kernel."""

from __future__ import annotations

from dataclasses import dataclass
import threading

import torch

from ..geometry import MAX_EP, MIN_EP, hierarchy_group_sizes


MAX_EXPERTS = 384
MAX_CTAS = 128
MAX_BROADCASTS = MAX_EXPERTS
# Nominal clock64 rate that converts a timeout in seconds to SM cycles, as the
# MoE all-to-all kernels assume.
ASSUMED_CLOCK_HZ = 2_000_000_000
# Default bound, in SM clock cycles, of the scheduler's grid barrier and
# cross-rank waits; a timeout records a status flag. Each launch takes its bound
# as an argument, so callers can allow more peer skew while ranks warm up. The
# default, 300 s at ASSUMED_CLOCK_HZ, matches NVLinkOneSided's serving timeout,
# so a transient stall on one rank does not trap every EP rank;
# TRTLLM_MOE_COMM_TIMEOUT_SEC can set a shorter bound.
SPIN_CYCLES = 600_000_000_000


def spin_cycles_for_timeout(seconds: int | None) -> int:
    """Spin bound for a timeout in seconds; ``None`` selects ``SPIN_CYCLES``."""
    if seconds is None:
        return SPIN_CYCLES
    if type(seconds) is not int or seconds <= 0:
        raise ValueError("scheduler timeout must be a positive exact int of seconds")
    return seconds * ASSUMED_CLOCK_HZ


def recommend_cuda_scheduler_ctas(
    ep_size: int,
    max_tokens_per_rank: int,
    topk: int = 6,
    threads: int = 512,
) -> int:
    """Return the bounded pure-CUDA CTA policy without importing CuTe."""

    for name, value in {
        "ep_size": ep_size,
        "max_tokens_per_rank": max_tokens_per_rank,
        "topk": topk,
        "threads": threads,
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive exact int")
    if threads != 512:
        raise ValueError("the pure-CUDA scheduler currently requires threads=512")
    routes = max_tokens_per_rank * topk
    if ep_size == 8:
        # Bound scheduler occupancy so independent same-stream work retains
        # launch capacity.
        workers = min(31, max(1, (routes + 3 * threads - 1) // (3 * threads)))
        return 1 + workers
    target_rounds = max(3, (96 + (ep_size + 6) // 2) // (ep_size + 6))
    workers = max(
        1, (max_tokens_per_rank * topk) // (threads * target_rounds)
    )
    workers = min(workers, MAX_CTAS - 1)
    rounds = (routes + workers * threads - 1) // (workers * threads)
    workers = (routes + rounds * threads - 1) // (rounds * threads)
    return 1 + workers


def symmetric_buffer_ints(ep_size: int, expert_count: int) -> int:
    # Two reset-free arrival generations plus a local epoch and two count slabs.
    return 64 + 2 * ep_size * expert_count


@dataclass(frozen=True)
class CudaSchedulerConfig:
    ep_size: int
    logical_expert_count: int
    extra_slots_per_rank: int
    max_tokens_per_rank: int
    topk: int = 6
    local_rank: int = 0
    threads: int = 512
    ctas: int | None = None

    def __post_init__(self) -> None:
        if self.ctas is None:
            object.__setattr__(
                self,
                "ctas",
                recommend_cuda_scheduler_ctas(
                    self.ep_size,
                    self.max_tokens_per_rank,
                    self.topk,
                    self.threads,
                ),
            )
        self.validate()

    @property
    def home_experts_per_rank(self) -> int:
        return self.logical_expert_count // self.ep_size

    @property
    def local_physical_slot_count(self) -> int:
        return self.home_experts_per_rank + self.extra_slots_per_rank

    @property
    def route_count(self) -> int:
        return self.max_tokens_per_rank * self.topk

    @property
    def world_size(self) -> int:
        return self.ep_size

    @property
    def global_expert_count(self) -> int:
        return self.logical_expert_count

    @property
    def helper_count(self) -> int:
        return self.extra_slots_per_rank

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return hierarchy_group_sizes(self.ep_size)

    def validate(self) -> None:
        values = {
            "ep_size": self.ep_size,
            "logical_expert_count": self.logical_expert_count,
            "extra_slots_per_rank": self.extra_slots_per_rank,
            "max_tokens_per_rank": self.max_tokens_per_rank,
            "topk": self.topk,
            "local_rank": self.local_rank,
            "threads": self.threads,
            "ctas": self.ctas,
        }
        if any(type(value) is not int for value in values.values()):
            raise ValueError("all CUDA scheduler dimensions must be exact ints")
        if not MIN_EP <= self.ep_size <= MAX_EP or self.ep_size % 2:
            raise ValueError(
                f"pure-CUDA scheduler supports even EP in [{MIN_EP},{MAX_EP}]")
        if not 0 <= self.local_rank < self.ep_size:
            raise ValueError("local_rank must be in [0,ep_size)")
        if not 1 <= self.logical_expert_count <= MAX_EXPERTS:
            raise ValueError(f"logical_expert_count must be in [1,{MAX_EXPERTS}]")
        if self.logical_expert_count % self.ep_size:
            raise ValueError("logical_expert_count must be EP-divisible")
        if self.extra_slots_per_rank <= 0:
            raise ValueError("extra_slots_per_rank must be positive")
        if self.ep_size * self.local_physical_slot_count > 2**31 - 1:
            raise ValueError("global physical slot addresses must fit int32")
        if self.max_tokens_per_rank <= 0 or self.topk <= 0:
            raise ValueError("token and topk dimensions must be positive")
        if self.ep_size * self.route_count > 2**31 - 1:
            raise ValueError("global route count must fit int32 histogram totals")
        if self.threads != 512:
            raise ValueError("the pure-CUDA scheduler currently requires threads=512")
        if not 1 <= self.ctas <= MAX_CTAS:
            raise ValueError(f"ctas must be in [1,{MAX_CTAS}]")


class CudaSchedulerOutputs:
    """The exact four-tensor hierarchical scheduler ABI."""

    __slots__ = (
        "physical_slot_ids",
        "hot_expert_ids",
        "hot_expert_group_level",
        "hot_expert_source_ranks",
    )

    def __init__(self, config: CudaSchedulerConfig, device: torch.device):
        self.physical_slot_ids = torch.empty(
            (config.max_tokens_per_rank, config.topk),
            dtype=torch.int32,
            device=device,
        )
        shape = (config.extra_slots_per_rank,)
        self.hot_expert_ids = torch.empty(shape, dtype=torch.int32, device=device)
        self.hot_expert_group_level = torch.empty(
            shape, dtype=torch.int32, device=device
        )
        self.hot_expert_source_ranks = torch.empty(
            shape, dtype=torch.int32, device=device
        )


def _stream_handle(stream: object) -> int:
    if isinstance(stream, bool):
        raise TypeError("stream must be a CUDA stream handle")
    handle = getattr(stream, "cuda_stream", stream)
    try:
        value = int(handle)
    except (TypeError, ValueError) as error:
        raise TypeError("stream must be a CUDA stream handle") from error
    if value < 0:
        raise ValueError("stream handle cannot be negative")
    return value


class CudaPhysicalSlotScheduler:
    """Preallocated HALO-Q outputs and workspace for the fused scheduler + copy kernel.

    The GPU-direct copy endpoint binds it once (``bind_gpu_direct``), binds
    the scheduler's arguments from ``fused_launch_args`` into its native
    context, and then launches it with the copy
    (``moe_rebalance_tma_submit_fused``), passing the routes, the route
    target and the spin bound.
    """

    def __init__(self, config: CudaSchedulerConfig, device: str | torch.device = "cuda"):
        config.validate()
        scheduler_device = torch.device(device)
        if scheduler_device.type != "cuda":
            raise ValueError("CudaPhysicalSlotScheduler requires a CUDA device")
        if scheduler_device.index is None:
            scheduler_device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(scheduler_device)
        if config.ctas > properties.multi_processor_count:
            raise ValueError("ctas cannot exceed the CUDA device SM count")

        self.cfg = config
        self.device = scheduler_device
        self.outputs = CudaSchedulerOutputs(config, scheduler_device)
        self.out = self.outputs
        self.status = torch.zeros(8, dtype=torch.int32, device=scheduler_device)
        # These tensors and the config values are fixed for the endpoint's
        # lifetime, so the copy binds them once; see fused_launch_args().
        self.partial = torch.zeros(
            config.ctas * config.logical_expert_count,
            dtype=torch.int32,
            device=scheduler_device,
        )
        warps = config.threads // 32
        bins = config.logical_expert_count + 1
        route_aux_ints = config.ctas * warps * bins
        if config.ctas > 1:
            route_aux_ints = max(
                route_aux_ints,
                2 + (config.ctas - 1) * bins + config.route_count,
            )
        self.route_aux = torch.zeros(
            route_aux_ints,
            dtype=torch.int32,
            device=scheduler_device,
        )
        # The main barrier uses [arrival, epoch]; word 2 publishes route readiness.
        self.grid_sync = torch.zeros(4, dtype=torch.int32, device=scheduler_device)
        # Each expert is broadcast at most once. The private plan's fixed
        # field stride is bounded by E, independently of requested slot capacity.
        self.max_broadcasts = MAX_BROADCASTS
        self.plan_workspace = torch.full(
            (2 + 6 * self.max_broadcasts,),
            -1,
            dtype=torch.int32,
            device=scheduler_device,
        )
        self.route_prefix = torch.zeros(
            self.max_broadcasts * (config.ep_size + 1),
            dtype=torch.int32,
            device=scheduler_device,
        )
        self.sym = torch.zeros(
            symmetric_buffer_ints(
                config.ep_size, config.logical_expert_count
            ),
            dtype=torch.int32,
            device=scheduler_device,
        )
        self.peer_base = torch.full(
            (config.ep_size,),
            int(self.sym.data_ptr()),
            dtype=torch.int64,
            device=scheduler_device,
        )
        self._connected = config.ep_size == 1
        self._lock = threading.Lock()
        # The bound copy's static stream: every fused launch shares the
        # workspace and reusable barrier epochs, so all of them use it.
        self._bound_stream: int | None = None

    def connect_peer_bases(self, peer_bases: object) -> None:
        if self._connected:
            raise RuntimeError("peer base binding is single-assignment")

        bases = tuple(peer_bases)  # type: ignore[arg-type]
        if len(bases) != self.cfg.ep_size or any(
            type(pointer) is not int or pointer <= 0 for pointer in bases
        ):
            raise ValueError("peer_bases must contain EP positive exact addresses")
        self.peer_base.copy_(
            torch.tensor(bases, dtype=torch.int64, device=self.device)
        )
        torch.cuda.synchronize(self.device)
        self._connected = True

    def bind_gpu_direct(self, stream: object) -> None:
        """Bind the GPU-direct copy that launches this scheduler with its copy.

        The copy reads the outputs and plan workspace in place, in the same
        kernel, and every fused launch uses this static stream. No additional
        plan allocation or GPU/host transfer is performed.
        """
        with self._lock:
            if self._bound_stream is not None:
                raise RuntimeError("GPU-direct binding is single-assignment")
            if not self._connected:
                raise RuntimeError("EP peer bases are not connected")
            self._bound_stream = _stream_handle(stream)

    def fused_launch_args(self) -> tuple:
        """Scheduler arguments that every fused launch of the bound copy
        shares, ``out_slots`` through ``threads`` of
        moe_rebalance_tma_bind_fused.

        A launch adds its route target, its spin bound and its routes: a
        caller-owned CUDA int32[T,K] tensor, read without input staging. T may
        be zero and may differ across ranks, up to max_tokens_per_rank. The
        kernel never reads outside those T rows and writes -1 to the unused
        output tail. All ranks, including empty
        ranks, must launch each generation in the same collective order. The
        caller keeps the routes alive and unmodified until the scheduler has
        read them; the kernel reports invalid expert values through
        check_status().
        """
        if self._bound_stream is None:
            raise RuntimeError("a fused launch requires a bound GPU-direct copy")
        return (
            self.outputs.physical_slot_ids,
            self.outputs.hot_expert_ids,
            self.outputs.hot_expert_group_level,
            self.outputs.hot_expert_source_ranks,
            self.peer_base,
            self.status,
            self.partial,
            self.route_aux,
            self.grid_sync,
            self.plan_workspace,
            self.route_prefix,
            self.cfg.ep_size,
            self.cfg.logical_expert_count,
            self.cfg.extra_slots_per_rank,
            self.cfg.local_rank,
            self.cfg.topk,
            self.cfg.route_count,
            self.cfg.ctas,
            self.cfg.threads,
        )

    def check_status(self) -> None:
        """Raise if any generation faulted since the last `reset_status()`.

        Each flag holds the unsigned modular epoch that wrote it, not a bare 1.
        Zero means never set; an error at wrapped epoch zero stores UINT32_MAX.
        That lets the per-generation status clear stay out of the submission path.
        """
        torch.cuda.synchronize(self.device)
        flags = [
            flag & 0xFFFFFFFF for flag in self.status[:6].tolist()
        ]  # one D2H, not six
        labels = (
            "cross-rank histogram rendezvous timeout",
            "CTA grid barrier timeout",
            "plan publication timeout",
            "invalid logical expert ID",
            "HALO helper coloring failure",
            "HALO quota/host invariant failure",
        )
        failures = [
            f"{label} (generation {flag})"
            for label, flag in zip(labels, flags)
            if flag
        ]
        if failures:
            raise RuntimeError("; ".join(failures))

    def reset_status(self) -> None:
        """Acknowledge and clear faults. Not needed per generation."""
        self.status.zero_()


__all__ = [
    "ASSUMED_CLOCK_HZ",
    "SPIN_CYCLES",
    "CudaPhysicalSlotScheduler",
    "CudaSchedulerConfig",
    "CudaSchedulerOutputs",
    "recommend_cuda_scheduler_ctas",
    "spin_cycles_for_timeout",
    "symmetric_buffer_ints",
]
