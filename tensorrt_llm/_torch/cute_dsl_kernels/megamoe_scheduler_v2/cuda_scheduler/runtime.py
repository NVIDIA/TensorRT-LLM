"""Host runtime for the pure-CUDA GAR-N and HALO-Q schedulers."""

from __future__ import annotations

from dataclasses import dataclass
import threading

import torch

from ..geometry import MAX_EP, MIN_EP, hierarchy_group_sizes


MAX_EXPERTS = 384
MAX_CTAS = 128
MAX_BROADCASTS = MAX_EXPERTS
SPIN_CYCLES = 8_000_000_000

def _plan_channel_word_count(*, abi_version: int, channel_words: int | None,
                             route_features: int, ep: int, helpers: int,
                             algorithm: str) -> int:
    """Validate an explicitly negotiated private publication allocation."""
    if type(abi_version) is not int or type(route_features) is not int:
        raise ValueError("plan channel ABI and features must be exact integers")
    if abi_version != 6:
        raise ValueError("CUDA scheduler requires PlanChannel ABI6")
    if route_features not in (0, 1) or (route_features and algorithm != "halo_q"):
        raise ValueError("external-owner routes require HALO-Q capability")
    if type(ep) is not int or not MIN_EP <= ep <= MAX_EP or ep % 2:
        raise ValueError("plan channel requires even EP in [2,32]")
    if type(helpers) is not int or helpers <= 0:
        raise ValueError("helper count must be a positive exact int")
    stride = (helpers + 3) // 4 * 4
    expected = 4 + 7 * stride
    if expected > 2**31 - 1:
        raise ValueError("plan channel word count does not fit its int32 ABI")
    if channel_words is None:
        channel_words = expected
    if type(channel_words) is not int or channel_words != expected:
        raise ValueError("plan channel allocation does not match its explicit ABI")
    return channel_words


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
    algorithm: str = "halo_q"
    enable_pdl: bool = True

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
        if self.algorithm not in ("legacy", "halo_q"):
            raise ValueError("algorithm must be 'legacy' or 'halo_q'")
        if type(self.enable_pdl) is not bool:
            raise ValueError("enable_pdl must be bool")


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
    """Preallocated pure-CUDA scheduler with no CuTe JIT executor."""

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
        self.logical_expert_ids = torch.full(
            (config.max_tokens_per_rank, config.topk),
            -1,
            dtype=torch.int32,
            device=scheduler_device,
        )
        self.status = torch.zeros(8, dtype=torch.int32, device=scheduler_device)
        # Everything the kernel takes between `routes` and `stream` is fixed for
        # the endpoint's lifetime.  Cache it so a submission does not rebuild 13
        # data_ptr() calls and ~11 config lookups every generation.
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
                1 + (config.ctas - 1) * bins + config.route_count,
            )
        self.route_aux = torch.zeros(
            route_aux_ints,
            dtype=torch.int32,
            device=scheduler_device,
        )
        # Two reset-free arrival counters followed by their completion epochs.
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
        self._plan_channel_lock = threading.Lock()
        self._plan_channel_ptr = 0
        self._plan_channel_abi_version = 0
        self._plan_channel_words = 0
        self._plan_channel_route_features = 0
        self._plan_channel_bound = False
        self._gpu_direct_bound = False
        self._plan_channel_pending = False
        self._plan_channel_stream_handle: int | None = None
        self._plan_channel_torch_stream: torch.cuda.Stream | None = None
        self._launch_stream_handle: int | None = None

    def _staging_stream(self) -> torch.cuda.Stream:
        current = torch.cuda.current_stream(self.device)
        bound = self._plan_channel_torch_stream
        if bound is not None and _stream_handle(current) != self._plan_channel_stream_handle:
            bound.wait_stream(current)
            return bound
        return current

    def stage(self, logical_expert_ids: torch.Tensor, *, validate_values: bool = False) -> None:
        expected = (self.cfg.max_tokens_per_rank, self.cfg.topk)
        if not isinstance(logical_expert_ids, torch.Tensor):
            raise TypeError("logical_expert_ids must be a torch.Tensor")
        if tuple(logical_expert_ids.shape) != expected:
            raise ValueError(f"logical_expert_ids must have shape {expected}")
        if logical_expert_ids.dtype != torch.int32 or not logical_expert_ids.is_contiguous():
            raise ValueError("logical_expert_ids must be contiguous torch.int32")
        if not logical_expert_ids.is_cuda or logical_expert_ids.device != self.device:
            raise ValueError("logical_expert_ids must be on the scheduler CUDA device")
        if validate_values:
            invalid = (logical_expert_ids < -1) | (
                logical_expert_ids >= self.cfg.logical_expert_count
            )
            if bool(invalid.any().item()):
                raise ValueError("logical expert IDs must be -1 or in [0,E)")
        with torch.cuda.device(self.device):
            stream = self._staging_stream()
            with torch.cuda.stream(stream):
                # No status clear: the kernel writes the failing generation into
                # each flag rather than a bare 1, so a stale flag is
                # distinguishable from a fresh one and clearing is only needed to
                # acknowledge a fault (`reset_status()`).
                self.logical_expert_ids.copy_(logical_expert_ids)
                logical_expert_ids.record_stream(stream)

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

    def bind_plan_channel(
        self,
        device_ptr: int,
        stream: object,
        *,
        abi_version: int,
        channel_words: int | None = None,
        route_features: int = 0,
    ) -> None:
        with self._plan_channel_lock:
            words = _plan_channel_word_count(
                abi_version=abi_version, channel_words=channel_words,
                route_features=route_features, ep=self.cfg.ep_size,
                helpers=self.cfg.extra_slots_per_rank, algorithm=self.cfg.algorithm,
            )
            if self._plan_channel_bound:
                raise RuntimeError("plan channel binding is single-assignment")
            if type(device_ptr) is not int or device_ptr <= 0 or device_ptr % 16:
                raise ValueError("device_ptr must be a positive 16-byte aligned address")
            handle = _stream_handle(stream)
            with torch.cuda.device(self.device):
                self._plan_channel_torch_stream = torch.cuda.ExternalStream(
                    handle, device=self.device
                )
            self._plan_channel_ptr = device_ptr
            self._plan_channel_abi_version = abi_version
            self._plan_channel_words = words
            self._plan_channel_route_features = route_features
            self._plan_channel_stream_handle = handle
            self._plan_channel_bound = True

    def bind_gpu_direct(self, stream: object) -> None:
        """Bind a copy consumer to the existing outputs/workspace, without publication.

        The consumer must enqueue on this static stream, then call
        ``release_plan_channel`` before the next scheduler submission.
        No additional plan allocation or GPU/host transfer is performed.
        """
        with self._plan_channel_lock:
            if self._plan_channel_bound:
                raise RuntimeError("plan channel binding is single-assignment")
            handle = _stream_handle(stream)
            if self._launch_stream_handle is not None and handle != self._launch_stream_handle:
                raise RuntimeError("GPU-direct binding requires the existing static stream")
            with torch.cuda.device(self.device):
                self._plan_channel_torch_stream = torch.cuda.ExternalStream(handle, device=self.device)
            self._plan_channel_ptr = 0
            self._plan_channel_abi_version = 0
            self._plan_channel_words = 0
            self._plan_channel_route_features = 0
            self._plan_channel_stream_handle = handle
            self._plan_channel_bound = True
            self._gpu_direct_bound = True

    def release_plan_channel(self) -> None:
        """Allow another publication after its copy has been enqueued.

        GPU-direct readers and scheduler writers use the same static stream,
        so stream order protects the single plan allocation without a host
        completion wait. This does not release the weight bank's consumer lease.
        """
        with self._plan_channel_lock:
            if not self._plan_channel_bound or not self._plan_channel_pending:
                raise RuntimeError("plan channel has no pending publication")
            self._plan_channel_pending = False

    def launch(self, stream: object | None = None) -> CudaSchedulerOutputs:
        if not self._connected:
            raise RuntimeError("EP peer bases are not connected")
        with self._plan_channel_lock:
            if self._plan_channel_bound:
                if self._plan_channel_pending:
                    raise RuntimeError("plan publication is still pending")
                handle = self._plan_channel_stream_handle
                if stream is not None and _stream_handle(stream) != handle:
                    raise RuntimeError("bound plan channel requires its static stream")
            else:
                handle = _stream_handle(
                    torch.cuda.current_stream(self.device) if stream is None else stream
                )
            if self._launch_stream_handle is None:
                self._launch_stream_handle = handle
            elif handle != self._launch_stream_handle:
                raise RuntimeError(
                    "CUDA scheduler uses one static stream because its workspace "
                    "and reset-free generations are shared"
                )
            self._launch_op(
                self.logical_expert_ids,
                int(handle),
                valid_route_count=self.cfg.route_count,
            )
            if self._plan_channel_bound:
                self._plan_channel_pending = True
        return self.outputs

    def _launch_op(
        self,
        routes: torch.Tensor,
        stream_handle: int,
        *,
        valid_route_count: int,
    ) -> None:
        stream = self._plan_channel_torch_stream
        if stream is None or _stream_handle(stream) != stream_handle:
            stream = torch.cuda.ExternalStream(stream_handle, device=self.device)
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            torch.ops.trtllm.moe_rebalance_halo_q(
                routes,
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
                self._plan_channel_ptr,
                self.cfg.ep_size,
                self.cfg.logical_expert_count,
                self.cfg.extra_slots_per_rank,
                self.cfg.local_rank,
                self.cfg.route_count,
                self.cfg.ctas,
                self.cfg.threads,
                0 if self.cfg.algorithm == "legacy" else 1,
                self.cfg.enable_pdl,
                SPIN_CYCLES,
                self._plan_channel_abi_version,
                self._plan_channel_words,
                self._plan_channel_route_features,
                valid_route_count,
            )

    def submit(
        self, routes: torch.Tensor, stream_handle: int, *,
        valid_tokens: int | None = None,
    ) -> CudaSchedulerOutputs:
        """Read a caller-owned CUDA int32[T,K] tensor without input staging.

        T may be zero and may differ across ranks, up to max_tokens_per_rank.
        By default all T rows are valid; valid_tokens can select a shorter
        prefix of an existing allocation. The kernel never reads outside that
        prefix and writes -1 to the unused output tail. All ranks, including
        empty ranks, must submit each generation in the same collective order.

        The caller must order input production before this static stream and
        keep routes alive and unmodified until the scheduler completes. This
        includes allocator lifetime protection for temporary cross-stream
        inputs. Metadata checks below do not launch GPU work or synchronize;
        invalid expert values are reported by the kernel through check_status().

        Graph capture fixes both the input pointer and valid_tokens. For replay
        with changing pointers/lengths use eager submit; stage()+launch() remains
        available for a fixed-capacity graph input buffer.
        """
        with self._plan_channel_lock:
            if self._plan_channel_bound:
                if self._plan_channel_pending:
                    raise RuntimeError("plan publication is still pending")
                if stream_handle != self._plan_channel_stream_handle:
                    raise RuntimeError("bound plan channel requires its static stream")
            if self._launch_stream_handle is None:
                self._launch_stream_handle = stream_handle
            elif stream_handle != self._launch_stream_handle:
                raise RuntimeError(
                    "CUDA scheduler uses one static stream because its workspace "
                    "and reset-free generations are shared"
                )
            self._submit_routes(routes, stream_handle, valid_tokens=valid_tokens)
            if self._plan_channel_bound:
                self._plan_channel_pending = True
        return self.outputs

    def _submit_routes(
        self, routes: torch.Tensor, stream_handle: int, *,
        valid_tokens: int | None = None,
    ) -> None:
        if not isinstance(routes, torch.Tensor):
            raise TypeError("routes must be a torch.Tensor")
        if (routes.ndim != 2 or routes.shape[1] != self.cfg.topk
                or routes.shape[0] > self.cfg.max_tokens_per_rank):
            raise ValueError("routes must have shape [T,topk] with 0 <= T <= max_tokens_per_rank")
        if routes.dtype != torch.int32 or not routes.is_contiguous():
            raise ValueError("routes must be contiguous torch.int32")
        if not routes.is_cuda or routes.device != self.device:
            raise ValueError("routes must be on the scheduler CUDA device")
        if valid_tokens is None:
            valid_tokens = routes.shape[0]
        if type(valid_tokens) is not int or not 0 <= valid_tokens <= routes.shape[0]:
            raise ValueError("valid_tokens must be an int in [0,T]")
        self._launch_op(
            routes,
            stream_handle,
            valid_route_count=valid_tokens * self.cfg.topk,
        )

    def run(
        self,
        logical_expert_ids: torch.Tensor,
        stream: object | None = None,
        *,
        validate_values: bool = False,
    ) -> CudaSchedulerOutputs:
        if stream is not None and not self._plan_channel_bound:
            with torch.cuda.device(self.device):
                external = torch.cuda.ExternalStream(
                    _stream_handle(stream), device=self.device
                )
                with torch.cuda.stream(external):
                    self.stage(logical_expert_ids, validate_values=validate_values)
        else:
            self.stage(logical_expert_ids, validate_values=validate_values)
        return self.launch(stream)

    def check_status(self) -> None:
        """Raise if any generation faulted since the last `reset_status()`.

        Each flag holds the generation that wrote it, not a bare 1, so the
        failing generation is named and a stale flag from an earlier generation
        is not mistaken for a fresh one. Generations are 1-based, so 0 means
        never set. That is what lets the per-generation status clear be removed
        from the submission path.
        """
        torch.cuda.synchronize(self.device)
        flags = self.status[:6].tolist()  # one D2H, not six
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
    "CudaPhysicalSlotScheduler",
    "CudaSchedulerConfig",
    "CudaSchedulerOutputs",
    "recommend_cuda_scheduler_ctas",
    "symmetric_buffer_ints",
]
