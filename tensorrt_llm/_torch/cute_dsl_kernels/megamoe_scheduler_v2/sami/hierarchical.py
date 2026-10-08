"""Hierarchical HALO-Q live arena and low-overhead SAMI submission path."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import os
import threading
from typing import Protocol, runtime_checkable

import torch
from cuda.bindings import driver as cuda

from ..native import load_hierarchical_native
from .arena import BoundLiveWeightArena, LiveWeightArena, _TensorIdentity
from ._util import check_cuda
from ..geometry import hierarchy_group_sizes
from .geometry import BundleLayout

# Bound on waiting for every peer to publish its mapped plan. Steady state
# surfaces a lost peer promptly; the first generation allows for compilation
# inside the rendezvous and the resulting arrival skew.
_DEFAULT_PLAN_TIMEOUT_NS = 2_000_000_000
_DEFAULT_FIRST_PLAN_TIMEOUT_NS = 120_000_000_000


def _read_plan_timeout_ns() -> int:
    raw = os.environ.get("MEGAMOE_SAMI_PLAN_TIMEOUT_NS", "").strip()
    if not raw:
        return _DEFAULT_PLAN_TIMEOUT_NS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(
            "MEGAMOE_SAMI_PLAN_TIMEOUT_NS must be a positive integer of "
            f"nanoseconds; got {raw!r}"
        ) from None
    if value <= 0:
        raise ValueError(
            "MEGAMOE_SAMI_PLAN_TIMEOUT_NS must be positive; got "
            f"{value}. There is deliberately no way to disable the bound."
        )
    return value


# Resolved once at import. submit() is on the per-iteration path, so the bound
# must not cost an environment lookup per generation; it is also not something
# that may legitimately change mid-run.
_PLAN_TIMEOUT_NS = _read_plan_timeout_ns()
# An explicit override is a deliberate widening, so never let the first
# generation end up tighter than what the operator asked for.
_FIRST_PLAN_TIMEOUT_NS = max(_DEFAULT_FIRST_PLAN_TIMEOUT_NS, _PLAN_TIMEOUT_NS)


@dataclass(frozen=True)
class BoundHierarchicalLiveWeightArena:
    """Validated base arena plus this rank's multicast alias at every level."""

    arena: "HierarchicalLiveWeightArena" = field(repr=False, compare=False)
    base: BoundLiveWeightArena = field(repr=False, compare=False)
    group_sizes: tuple[int, ...]
    group_destination_ptrs: tuple[int, ...]
    group_mc_views: tuple[tuple[object, ...], ...] = field(
        repr=False, compare=False
    )
    group_mc_owners: tuple[tuple[object, ...], ...] = field(
        repr=False, compare=False
    )


@dataclass(frozen=True)
class HierarchicalLiveWeightArena:
    """One external live bank with a multicast alias for each HALO level.

    ``group_mc_views`` and ``group_mc_owners`` are plane-major:
    ``[plane][level]``.  At a given level, each rank exposes only the alias and
    strong VMM owner token for its own aligned group.  Level zero is the
    full-world alias already present in ``base.planes[*].mc_view``.
    """

    base: LiveWeightArena
    group_sizes: tuple[int, ...]
    group_mc_views: tuple[tuple[object, ...], ...] = field(
        repr=False, compare=False
    )
    group_mc_owners: tuple[tuple[object, ...], ...] = field(
        repr=False, compare=False
    )
    aliases_same_backing: bool = False

    def bind(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: BundleLayout,
        device: int,
    ) -> BoundHierarchicalLiveWeightArena:
        if self.aliases_same_backing is not True:
            raise ValueError("hierarchical multicast aliases are not provider-attested")
        expected_sizes = hierarchy_group_sizes(world)
        if self.group_sizes != expected_sizes:
            raise ValueError(
                f"hierarchy group sizes {self.group_sizes} != {expected_sizes}"
            )
        base = self.base.bind(
            world=world,
            rank=rank,
            home_count=home_count,
            helper_count=helper_count,
            bundle=bundle,
            device=device,
        )
        if len(self.group_mc_views) != len(bundle.planes):
            raise ValueError("hierarchical arena must expose every weight plane")
        if len(self.group_mc_owners) != len(bundle.planes):
            raise ValueError("hierarchical arena must retain every VMM owner")
        if self.base.terminals.backing_owner is None:
            raise ValueError("hierarchical terminal backing owner is not retained")

        identities: list[tuple[_TensorIdentity, ...]] = []
        for views, owners, base_view in zip(
            self.group_mc_views, self.group_mc_owners, self.base.planes
        ):
            if base_view.backing_owner is None:
                raise ValueError(f"{base_view.name} backing owner is not retained")
            if len(views) != len(expected_sizes):
                raise ValueError(
                    f"{base_view.name} has the wrong hierarchy depth"
                )
            if len(owners) != len(expected_sizes) or any(
                owner is None for owner in owners
            ):
                raise ValueError(
                    f"{base_view.name} must retain every subgroup VMM owner"
                )
            captured = tuple(
                _TensorIdentity.capture(
                    view, name=f"{base_view.name}.group_mc[{level}]"
                )
                for level, view in enumerate(views)
            )
            reference = _TensorIdentity.capture(
                base_view.mc_view, name=f"{base_view.name}.mc"
            )
            for identity in captured:
                if identity.metadata != reference.metadata:
                    raise ValueError(
                        f"{base_view.name} group multicast metadata changed"
                    )
            if captured[0].pointer != reference.pointer:
                raise ValueError(
                    f"{base_view.name} level-zero alias is not the base MC view"
                )
            if len({identity.pointer for identity in captured}) != len(captured):
                raise ValueError(
                    f"{base_view.name} hierarchy levels need distinct virtual mappings"
                )
            identities.append(captured)

        destinations: list[int] = []
        for level in range(len(expected_sizes)):
            for helper in range(helper_count):
                physical_slot = home_count + helper
                for plane_index, layout in enumerate(bundle.planes):
                    destinations.append(
                        identities[plane_index][level].pointer
                        + physical_slot * layout.nbytes
                    )
        return BoundHierarchicalLiveWeightArena(
            arena=self,
            base=base,
            group_sizes=expected_sizes,
            group_destination_ptrs=tuple(destinations),
            group_mc_views=self.group_mc_views,
            group_mc_owners=self.group_mc_owners,
        )

# runtime_checkable because __init__ isinstance()-checks the provider below.
# Without it that check raises TypeError instead of validating anything, and
# only at runtime -- no import, compile or signature check sees it.
@runtime_checkable
class HierarchicalLiveWeightArenaProvider(Protocol):
    """Provider boundary implemented by TEKit or a Fabric test allocator."""

    def build_hierarchical_live_arena(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: BundleLayout,
        device: int,
        comm: object,
    ) -> HierarchicalLiveWeightArena:
        """Construct the external live bank and all aligned-group aliases."""


class HierarchicalPlanChannel:
    """Private copy ABI6, populated by the fused GPU copy-policy adapter."""

    ABI_VERSION = 6

    @staticmethod
    def stride_for(helper_count: int) -> int:
        if type(helper_count) is not int or helper_count <= 0:
            raise ValueError("helper_count must be a positive exact int")
        return (helper_count + 3) // 4 * 4

    def __init__(self, helper_count: int) -> None:
        self.S = helper_count
        self.stride = self.stride_for(helper_count)
        self.word_count = 4 + 7 * self.stride
        if self.word_count > (1 << 31) - 1:
            raise ValueError("PlanChannel word count exceeds native int32 indexing")
        self.host = torch.full((self.word_count,), -1, dtype=torch.int32, pin_memory=True)
        self.host_ptr = int(self.host.data_ptr())
        if self.host_ptr % 16:
            raise RuntimeError("hierarchical PlanChannel is not 16-byte aligned")
        self.dev_ptr = int(check_cuda(cuda.cuMemHostGetDevicePointer(self.host_ptr, 0),
                                     "cuMemHostGetDevicePointer(HierarchicalPlanChannel)"))

    def _field(self, index: int) -> torch.Tensor:
        begin = 4 + index * self.stride
        return self.host[begin:begin + self.S]

    @property
    def selected(self) -> torch.Tensor:
        return self._field(0)

    @property
    def levels(self) -> torch.Tensor:
        return self._field(1)

    @property
    def owners(self) -> torch.Tensor:
        return self._field(2)

    @property
    def copy_modes(self) -> torch.Tensor:
        return self._field(3)

    @property
    def global_max_send(self) -> torch.Tensor:
        return self.host[1]

def _u64_array(size: int) -> ctypes.Array:
    return (ctypes.c_uint64 * size)()


def _i32_array(values: tuple[int, ...]) -> ctypes.Array:
    result = (ctypes.c_int32 * len(values))()
    for index, value in enumerate(values):
        result[index] = value
    return result


def _address(values: ctypes.Array) -> int:
    return ctypes.addressof(values)


def _stream_handle(stream: object) -> int:
    if isinstance(stream, bool):
        raise TypeError("stream must be a CUDA stream handle")
    handle = getattr(stream, "cuda_stream", stream)
    value = int(handle)
    if value <= 0:
        raise ValueError("hierarchical copy requires a positive non-default stream")
    return value


def _event_handle(event: object | None) -> int:
    if event is None:
        return 0
    if isinstance(event, bool):
        raise TypeError("event must be a CUDA event handle")
    return int(getattr(event, "cuda_event", event))


@dataclass(frozen=True)
class HierarchicalCopyTicket:
    generation: int
    terminal_flags: torch.Tensor = field(repr=False)
    command_count: int


class HierarchicalSamiWeightBroadcast:
    """One-rank HALO-Q copy endpoint with a single C hot-path context."""

    partition = "bundle"
    remote_visibility_release = True
    live_bank_count = 1

    def __init__(
        self,
        *,
        comm: object,
        device: int,
        bundle: BundleLayout,
        helper_count: int,
        global_expert_count: int,
        arena_provider: HierarchicalLiveWeightArenaProvider,
        copy_backend: str = "batch",
        tma_sm_count: int | None = None,
        tma_warps: int | None = None,
        tma_route: str = "plan",
        tma_source_load_factor: float = 1.0,
        tma_plan_mode: str | None = None,
    ) -> None:
        if copy_backend not in ("batch", "tma"):
            raise ValueError("copy_backend must be 'batch' or 'tma'")
        for name, value in (("tma_sm_count", tma_sm_count), ("tma_warps", tma_warps)):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive exact int or None")
        if tma_sm_count is not None and tma_sm_count > 8:
            raise ValueError("tma_sm_count must be at most 8")
        if tma_warps is not None and tma_warps > 32:
            raise ValueError("tma_warps must be at most 32; device shared-memory limits also apply")
        if tma_route not in ("plan", "source", "scatter"):
            raise ValueError("tma_route must be 'plan', 'source', or 'scatter'")
        if (type(tma_source_load_factor) not in (int, float) or
                not 0.01 <= tma_source_load_factor < 2.0 or
                abs(100 * tma_source_load_factor - round(100 * tma_source_load_factor)) > 1e-9):
            raise ValueError("tma_source_load_factor must be 0.01..1.99 in steps of 0.01")
        if copy_backend != "tma" and (
            tma_sm_count is not None or tma_warps is not None or tma_route != "plan" or
            tma_source_load_factor != 1.0
        ):
            raise ValueError("TMA settings require copy_backend='tma'")
        if tma_plan_mode is not None and tma_plan_mode not in ("gpu_direct", "host"):
            raise ValueError("tma_plan_mode must be 'gpu_direct', 'host', or None")
        if copy_backend != "tma" and tma_plan_mode == "gpu_direct":
            raise ValueError("device plan requires copy_backend='tma'")
        self.tma_plan_mode = tma_plan_mode or ("gpu_direct" if copy_backend == "tma" else "host")
        self.copy_backend = copy_backend
        self.tma_route = tma_route
        self.tma_source_load_factor = round(100 * tma_source_load_factor) / 100
        world = getattr(comm, "world", None)
        rank = getattr(comm, "rank", None)
        if type(world) is not int:
            raise ValueError("comm.world must be an exact int")
        hierarchy_group_sizes(world)
        if type(rank) is not int or not 0 <= rank < world:
            raise ValueError("comm.rank must be in [0, world)")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a non-negative exact int")
        if (
            type(global_expert_count) is not int
            or global_expert_count <= 0
            or global_expert_count % world
        ):
            raise ValueError("global_expert_count must be positive and EP-divisible")
        if type(helper_count) is not int or helper_count <= 0:
            raise ValueError("helper_count must be a positive exact int")
        if global_expert_count + world * helper_count > (1 << 31) - 1:
            raise ValueError("physical slots exceed public int32 indexing")
        if 4 + 7 * ((helper_count + 3) // 4 * 4) > (1 << 31) - 1:
            raise ValueError("PlanChannel word count exceeds native int32 indexing")
        if not isinstance(arena_provider, HierarchicalLiveWeightArenaProvider):
            raise TypeError("arena_provider lacks the hierarchical arena protocol")
        self.comm = comm
        self.device = device
        self.bundle = bundle
        self.helper_count = helper_count
        self.global_expert_count = global_expert_count
        self.home_count = global_expert_count // world
        self.home_begin = rank * self.home_count
        self.provider = arena_provider
        self.cuda = cuda
        torch.cuda.set_device(device)
        if self.tma_plan_mode == "host":
            self._validate_mapped_capabilities()
        # Fail toolchain/launch-geometry checks before allocating collective
        # Fabric resources. The optional backend must not leave half an arena.
        tma_module = None
        if copy_backend == "tma":
            device_sms = torch.cuda.get_device_properties(device).multi_processor_count
            if tma_sm_count is not None and tma_sm_count > device_sms:
                raise ValueError(f"tma_sm_count must be at most {device_sms} on this GPU")
            self._validate_tma_warp_capacity(tma_warps)
            # Unlike CTA/warp counts, the ownership rule must agree across ranks.
            # This is one cold-path collective, outside every copy submission.
            factors = comm.allgather(self.tma_source_load_factor)
            if any(value != self.tma_source_load_factor for value in factors):
                raise ValueError("tma_source_load_factor must agree across all ranks")
            if self.tma_plan_mode == "host":
                # The host-plan backend remains a diagnostic fallback. The
                # production GPU-direct path uses the statically linked THOP.
                from ..native import load_tma_native

                tma_module = load_tma_native()

        live = arena_provider.build_hierarchical_live_arena(
            world=world,
            rank=rank,
            home_count=self.home_count,
            helper_count=helper_count,
            bundle=bundle,
            device=device,
            comm=comm,
        )
        if not isinstance(live, HierarchicalLiveWeightArena):
            raise TypeError("provider returned a non-hierarchical live arena")
        self.live_arena = live
        self.arena = live.bind(
            world=world,
            rank=rank,
            home_count=self.home_count,
            helper_count=helper_count,
            bundle=bundle,
            device=device,
        )
        self.live_plane_tensors = self.arena.base.local_plane_views
        self.terminal_flags = self.live_arena.base.terminals.local_view
        self.flags_local = self.arena.base.terminal_local_ptr
        self.flags_mc = self.arena.base.terminal_mc_ptr + 8 * rank
        check_cuda(
            cuda.cuMemsetD8(
                cuda.CUdeviceptr(self.flags_local), 0, 8 * world
            ),
            "cuMemsetD8(hierarchical terminals)",
        )
        torch.cuda.synchronize(device)
        comm.barrier()

        # GPU-direct consumes the scheduler's original device storage: no
        # mapped channel, duplicate device plan, or plan publication is needed.
        self.plan_channel = (None if self.tma_plan_mode == "gpu_direct"
                             else HierarchicalPlanChannel(helper_count))
        plan_stride = (helper_count + 3) // 4 * 4
        plan_words = 4 + 7 * plan_stride
        self._thop_tma = copy_backend == "tma" and self.tma_plan_mode == "gpu_direct"
        self.mod = (None if self._thop_tma else
                    tma_module if copy_backend == "tma" else load_hierarchical_native())
        tma_kwargs = {}
        if copy_backend == "tma":
            tma_kwargs = dict(tma_sm_count=tma_sm_count or 0, tma_warps=tma_warps or 0,
                              tma_route={"plan": 0, "source": 1, "scatter": 2}[tma_route],
                              tma_source_load_percent=round(100 * self.tma_source_load_factor),
                              plan_on_device=self.tma_plan_mode == "gpu_direct")
        planes = len(bundle.planes)
        self._plane_bytes = _u64_array(planes)
        for index, plane in enumerate(bundle.planes):
            self._plane_bytes[index] = plane.nbytes
        self._source_table = _u64_array(global_expert_count * planes)
        for index, pointer in enumerate(self.arena.base.scatter_source_ptrs):
            self._source_table[index] = pointer
        self._destination_table = _u64_array(
            len(self.arena.group_destination_ptrs)
        )
        for index, pointer in enumerate(self.arena.group_destination_ptrs):
            self._destination_table[index] = pointer
        self._group_sizes = _i32_array(self.arena.group_sizes)
        partition = {"bundle": 1, "per_plane": 0}.get(self.partition)
        if partition is None:
            raise ValueError(f"unknown hierarchy partition {self.partition!r}")
        if self._thop_tma:
            ctx = torch.ops.trtllm.moe_rebalance_tma_create(
                helper_count * planes, tma_sm_count or 0, tma_warps or 0
            )
            try:
                target_count = sum(world // size for size in self.arena.group_sizes[1:])
                torch.ops.trtllm.moe_rebalance_tma_configure_gpu_plan(
                    ctx, world, rank, helper_count, planes,
                    global_expert_count, self.home_count, plan_stride,
                    len(self.arena.group_sizes), target_count, 6, plan_words, 0,
                    {"plan": 0, "source": 1, "scatter": 2}[tma_route],
                    round(100 * self.tma_source_load_factor),
                    list(self.arena.group_sizes),
                    list(self.arena.base.scatter_source_ptrs),
                    list(self.arena.group_destination_ptrs), [],
                    [plane.nbytes for plane in bundle.planes],
                )
                config_names = (
                    "abi_version", "device", "device_sm_count", "sms", "warps",
                    "threads_per_cta", "slots_per_warp", "bank0_slots_per_warp",
                    "bank1_slots_per_warp", "slice_bytes", "dynamic_shared_bytes",
                    "device_optin_shared_bytes", "device_shared_bytes_per_sm",
                    "max_active_ctas_per_sm", "compute_major", "compute_minor",
                    "total_slots", "max_slots_per_warp", "extra_slot_warps",
                    "max_warps", "max_segments",
                )
                tma_config = dict(zip(
                    config_names, torch.ops.trtllm.moe_rebalance_tma_config(ctx)
                ))
                current_generation = torch.ops.trtllm.moe_rebalance_tma_current_gen(ctx)
                if int(current_generation) != 1:
                    raise RuntimeError("hierarchical SAMI context is not fresh")
            except BaseException:
                torch.ops.trtllm.moe_rebalance_tma_destroy(ctx)
                raise
            self.ctx = ctx
            self.tma_config = tma_config
        else:
            self.ctx = self.mod.create(
                world=world,
                rank=rank,
                helper_count=helper_count,
                planes=planes,
                global_experts=global_expert_count,
                src_table=_address(self._source_table),
                dst_table=_address(self._destination_table),
                plane_bytes=_address(self._plane_bytes),
                plan_ptr=(0 if self.plan_channel is None else self.plan_channel.host_ptr),
                level_count=len(self.arena.group_sizes),
                group_sizes=_address(self._group_sizes),
                owner_stride=plan_stride,
                flag_mc=self.flags_mc,
                partition=partition,
                plan_abi_version=6,
                plan_words=plan_words,
                **tma_kwargs,
            )
            self.tma_config = (
                self.mod.tma_config(self.ctx) if copy_backend == "tma" else None
            )
            self.mod.set_loc_hint(self.ctx, device, device)
            current_generation = self.mod.current_gen(self.ctx)
        if not self._thop_tma and int(current_generation) != 1:
            raise RuntimeError("hierarchical SAMI context is not fresh")

        self._lock = threading.Lock()
        self._scheduler: object | None = None
        self._outputs: object | None = None
        self._stream = 0
        self._stream_owner: object | None = None
        self._generation = 1
        self._reuse_authority: object | None = None
        self._active_generation = 0
        self._local_release_generation = 0
        self._collective_safe_generation = 0
        self._handoff_aborted = False

    def _validate_tma_warp_capacity(self, warps: int | None) -> None:
        # Reject impossible device geometry before building a collective arena.
        # Native creation additionally checks the compiled kernel's own static
        # shared memory, thread limit and occupancy.
        device = check_cuda(self.cuda.cuDeviceGet(self.device), "cuDeviceGet(TMA geometry)")
        shared_bytes = int(check_cuda(self.cuda.cuDeviceGetAttribute(
            self.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device,
        ), "cuDeviceGetAttribute(TMA opt-in shared memory)"))
        max_threads = int(check_cuda(self.cuda.cuDeviceGetAttribute(
            self.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            device,
        ), "cuDeviceGetAttribute(TMA block threads)"))
        slot_bytes = 8208 + 32  # Logical 8 KiB plus aligned-load padding and SlotControl.
        slots = max(0, shared_bytes // slot_bytes)
        while slots and ((slots * slot_bytes + 127) & ~127) > shared_bytes:
            slots -= 1
        max_warps = min(32, slots // 2, max_threads // 64)
        if (warps or 7) > max_warps:
            raise ValueError(
                f"tma_warps must be at most {max_warps} on this GPU "
                "(at least two 8 KiB slices per worker pair)"
            )

    def _validate_mapped_capabilities(self) -> None:
        device = check_cuda(
            self.cuda.cuDeviceGet(self.device),
            "cuDeviceGet(hierarchical plan device)",
        )
        attributes = (
            (
                self.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
                "CAN_MAP_HOST_MEMORY",
            ),
            (
                self.cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
                "HOST_NATIVE_ATOMIC_SUPPORTED",
            ),
        )
        for attribute, name in attributes:
            supported = int(
                check_cuda(
                    self.cuda.cuDeviceGetAttribute(attribute, device),
                    f"cuDeviceGetAttribute({name})",
                )
            )
            if supported != 1:
                raise RuntimeError(f"CUDA device lacks {name}")

    def _validate_stream_context(self, handle: int) -> None:
        get_current = getattr(self.cuda, "cuCtxGetCurrent", None)
        get_device = getattr(self.cuda, "cuCtxGetDevice", None)
        get_stream_context = getattr(self.cuda, "cuStreamGetCtx", None)
        if not callable(get_current) or not callable(get_device):
            raise RuntimeError("CUDA driver binding cannot validate the current context")
        current_context = check_cuda(get_current(), "cuCtxGetCurrent")
        if current_context is None:
            raise RuntimeError("no current CUDA context for hierarchical scheduler")
        current_device = int(check_cuda(get_device(), "cuCtxGetDevice"))
        if current_device != self.device:
            raise ValueError("current CUDA context and SAMI devices differ")
        if handle:
            if not callable(get_stream_context):
                raise RuntimeError("CUDA driver binding cannot validate stream context")
            stream_context = check_cuda(
                get_stream_context(handle), "cuStreamGetCtx(hierarchical scheduler)"
            )
            if stream_context != current_context:
                raise ValueError("scheduler stream is from a different CUDA context")

    def bind_scheduler(self, scheduler: object, stream: object) -> None:
        """Bind the four-field CUDA scheduler and one static stream for life."""

        with self._lock:
            if self._handoff_aborted:
                raise RuntimeError(
                    "hierarchical handoff aborted; quiesced group rebuild required"
                )
            if self._scheduler is not None:
                raise RuntimeError("hierarchical scheduler binding is single-assignment")
            config = getattr(scheduler, "cfg", None)
            outputs = getattr(scheduler, "outputs", None)
            if config is None or outputs is None:
                raise TypeError("scheduler lacks cfg/outputs")

            expected = (
                ("ep_size", int(self.comm.world)),
                ("logical_expert_count", self.global_expert_count),
                ("extra_slots_per_rank", self.helper_count),
                ("local_rank", int(self.comm.rank)),
            )
            for name, value in expected:
                observed = getattr(config, name, None)
                if type(observed) is not int or observed != value:
                    raise ValueError(f"scheduler {name} is incompatible with SAMI")
            for name in ("max_tokens_per_rank", "topk"):
                observed = getattr(config, name, None)
                if type(observed) is not int or observed <= 0:
                    raise ValueError(f"scheduler {name} must be a positive exact int")
            fields = (
                "physical_slot_ids",
                "hot_expert_ids",
                "hot_expert_group_level",
                "hot_expert_source_ranks",
            )
            tensors = tuple(getattr(outputs, name, None) for name in fields)
            if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
                raise TypeError("scheduler does not expose the four-tensor HALO ABI")
            expected_shapes = (
                (config.max_tokens_per_rank, config.topk),
                (self.helper_count,),
                (self.helper_count,),
                (self.helper_count,),
            )
            for name, tensor, expected_shape in zip(fields, tensors, expected_shapes):
                if (
                    tensor.dtype != torch.int32
                    or not tensor.is_cuda
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"{name} must be contiguous CUDA int32 storage"
                    )
                if tuple(tensor.shape) != expected_shape:
                    raise ValueError(f"{name} must have shape {expected_shape}")

            scheduler_device = getattr(scheduler, "device", None)
            try:
                scheduler_device = torch.device(scheduler_device)
            except (TypeError, RuntimeError) as exc:
                raise TypeError("scheduler must expose one explicit CUDA device") from exc
            if scheduler_device.type != "cuda" or scheduler_device.index is None:
                raise ValueError("scheduler must expose one explicit CUDA device")
            if scheduler_device.index != self.device:
                raise ValueError("scheduler and SAMI devices differ")
            if any(tensor.device != scheduler_device for tensor in tensors):
                raise ValueError("HALO outputs must share the scheduler CUDA device")

            handle = _stream_handle(stream)
            stream_device = getattr(stream, "device", None)
            if stream_device is not None:
                try:
                    stream_device = torch.device(stream_device)
                except (TypeError, RuntimeError) as exc:
                    raise TypeError("scheduler stream exposes an invalid device") from exc
                if stream_device != scheduler_device:
                    raise ValueError("scheduler stream and output devices differ")
            gpu_direct = self.plan_channel is None
            bind = getattr(scheduler, "bind_gpu_direct" if gpu_direct else "bind_plan_channel", None)
            release = getattr(scheduler, "release_plan_channel", None)
            if not callable(bind) or not callable(release):
                raise TypeError("scheduler lacks the copy handoff lifecycle")
            if gpu_direct:
                workspace = getattr(scheduler, "plan_workspace", None)
                capacity = getattr(scheduler, "max_broadcasts", None)
                if type(capacity) is not int or capacity <= 0:
                    raise ValueError("GPU-direct requires the scheduler plan capacity")
                if (not isinstance(workspace, torch.Tensor) or
                        workspace.device != scheduler_device or workspace.dtype != torch.int32 or
                        not workspace.is_contiguous() or tuple(workspace.shape) != (2 + 6 * capacity,)):
                    raise ValueError("GPU-direct requires the original CUDA scheduler plan workspace")
                native_bind = (torch.ops.trtllm.moe_rebalance_tma_bind_gpu_direct
                               if self._thop_tma else
                               getattr(self.mod, "bind_gpu_direct", None))
                if not callable(native_bind):
                    raise TypeError("native copy module lacks GPU-direct binding")
            self._validate_stream_context(handle)
            try:
                if gpu_direct:
                    bind(handle)
                    if self._thop_tma:
                        native_bind(self.ctx, *tensors[1:], workspace, capacity)
                    else:
                        native_bind(
                            self.ctx,
                            *(int(tensor.data_ptr()) for tensor in tensors[1:]),
                            int(workspace.data_ptr()),
                            capacity,
                        )
                else:
                    bind(self.plan_channel.dev_ptr, handle,
                         abi_version=self.plan_channel.ABI_VERSION,
                         channel_words=self.plan_channel.word_count)
            except BaseException:
                self._handoff_aborted = True
                raise
            self._scheduler = scheduler
            self._outputs = outputs
            self._stream = handle
            self._stream_owner = stream

    def _validate_reuse_authority(self, authority: object) -> None:
        if authority is None:
            raise TypeError("generation reuse authority cannot be None")
        if getattr(authority, "collective_safe", False) is not True:
            raise ValueError("generation reuse authority lacks collective safety")
        if getattr(authority, "producer_reuse_guarded", False) is not True:
            raise ValueError("generation reuse authority does not guard producer reuse")
        if getattr(authority, "live_bank_count", None) != self.live_bank_count:
            raise ValueError("generation reuse authority must target one live bank")
        copy_module = getattr(authority, "bound_copy_module", None)
        if copy_module is not self and getattr(copy_module, "backend", None) is not self:
            raise ValueError(
                "generation reuse authority is not bound to this hierarchical endpoint"
            )
        if copy_module is not self:
            config = getattr(copy_module, "config", None)
            expected = {
                "ep_size": int(self.comm.world),
                "logical_expert_count": self.global_expert_count,
                "extra_slots_per_rank": self.helper_count,
                "local_rank": int(self.comm.rank),
            }
            if config is None or any(
                type(getattr(config, name, None)) is not int
                or getattr(config, name) != value
                for name, value in expected.items()
            ):
                raise ValueError("generation reuse authority geometry is inconsistent")

    def bind_generation_reuse_authority(self, authority: object) -> None:
        """Bind the exact collective single-bank lease provider once."""

        self._validate_reuse_authority(authority)
        with self._lock:
            if self._handoff_aborted:
                raise RuntimeError(
                    "hierarchical handoff aborted; quiesced group rebuild required"
                )
            if self._reuse_authority is not None:
                raise RuntimeError("generation reuse authority is single-assignment")
            if self._active_generation:
                raise RuntimeError("generation reuse authority must bind before submit")
            self._reuse_authority = authority

    def release_generation_after(
        self, generation: int, completion_event: object
    ) -> None:
        """Order the copy stream after this rank's live-bank consumer event.

        This establishes only local stream ordering.  The bound reuse authority
        must separately attest its bounded all-rank barrier before another
        generation may overwrite the single live bank.
        """

        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        event_handle = _event_handle(completion_event)
        if event_handle <= 0:
            raise ValueError("completion_event must expose a positive CUDA event")
        with self._lock:
            if self._handoff_aborted or self._scheduler is None:
                raise RuntimeError("hierarchical scheduler handoff is not operational")
            if generation != self._active_generation:
                raise ValueError("generation is not the active live bank")
            if self._local_release_generation >= generation:
                raise RuntimeError("generation already has a local release event")
            check_cuda(
                self.cuda.cuStreamWaitEvent(self._stream, event_handle, 0),
                "cuStreamWaitEvent(live-bank consumer)",
            )
            self._local_release_generation = generation

    def mark_collective_reuse_safe(
        self, authority: object, generation: int
    ) -> None:
        """Accept the bound provider's all-rank proof for one generation."""

        if type(generation) is not int or generation <= 0:
            raise ValueError("generation must be a positive exact int")
        with self._lock:
            if self._handoff_aborted or self._scheduler is None:
                raise RuntimeError("hierarchical scheduler handoff is not operational")
            if authority is not self._reuse_authority:
                raise PermissionError("generation reuse authority identity mismatch")
            self._validate_reuse_authority(authority)
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
        self._validate_reuse_authority(self._reuse_authority)
        if (
            self._active_generation
            and self._collective_safe_generation < self._active_generation
        ):
            raise RuntimeError(
                "single live bank is still leased by the previous generation"
            )

    def submit(
        self,
        outputs: object,
        *,
        start_event: object | None = None,
        end_event: object | None = None,
        payload_end_event: object | None = None,
        timeout_ns: int | None = None,
    ) -> HierarchicalCopyTicket:
        """Enqueue copy after the bound scheduler on its static stream.

        With ``tma_plan_mode="gpu_direct"``, the copy kernel consumes the
        original scheduler outputs and workspace directly. Submission never reads
        that plan, waits for its publication,
        or constructs/uploads host descriptors. ``timeout_ns`` is retained for
        the host-plan mode; it does not add a CPU wait to the device mode.
        The ticket counts one fused copy/READY kernel in device mode. Detailed
        descriptor counts are available only from ``last_mode_counts()``.

        For TMA, READY is published by the last completed GPU CTA. Both
        ``payload_end_event`` and ``end_event`` therefore follow fused payload
        and notification; the former is not a payload-only timing boundary.
        Batch-copy retains its separate payload/release/terminal boundaries.

        For host plans, the bound turns a lost peer into an error instead of a
        hang. It is not a
        correctness barrier, so widening it costs only how long a genuine hang
        takes to surface.

        ``timeout_ns=None`` selects a wider first-generation bound because
        ranks can enter the rendezvous at different times while required kernels are
        compiled. Later generations use the steady-state bound so a lost peer surfaces
        promptly. ``MEGAMOE_SAMI_PLAN_TIMEOUT_NS`` can raise both defaults; callers may
        pass ``timeout_ns`` explicitly when a different diagnostic bound is required.
        The timeout is an error bound, not a correctness barrier.
        """

        if timeout_ns is None:
            # ``_active_generation`` is still the PREVIOUS generation here; it is
            # advanced a few lines below, after the native submit returns. So 0
            # means "nothing has been submitted yet" == this is frame 1.
            timeout_ns = (_PLAN_TIMEOUT_NS if self._active_generation
                          else _FIRST_PLAN_TIMEOUT_NS)
        elif type(timeout_ns) is not int or timeout_ns <= 0:
            raise ValueError("timeout_ns must be a positive exact int")

        with self._lock:
            if self._handoff_aborted:
                raise RuntimeError(
                    "hierarchical handoff aborted; quiesced group rebuild required"
                )
            if outputs is not self._outputs or self._scheduler is None:
                raise RuntimeError("outputs are not from the bound CUDA scheduler")
            self._require_live_bank_reusable()
            try:
                if self._thop_tma:
                    with torch.cuda.device(self.device), torch.cuda.stream(
                        self._stream_owner
                    ):
                        if start_event is not None:
                            start_event.record()
                        commands = int(
                            torch.ops.trtllm.moe_rebalance_tma_submit_gpu_direct(
                                self.ctx, self.flags_mc
                            )
                        )
                        if payload_end_event is not None:
                            payload_end_event.record()
                        if end_event is not None:
                            end_event.record()
                elif payload_end_event is None:
                    commands = int(
                        self.mod.submit(
                            self.ctx,
                            self._stream,
                            _event_handle(start_event),
                            _event_handle(end_event),
                            timeout_ns,
                        )
                    )
                else:
                    commands = int(
                        self.mod.submit(
                            self.ctx,
                            self._stream,
                            _event_handle(start_event),
                            _event_handle(end_event),
                            timeout_ns,
                            _event_handle(payload_end_event),
                        )
                    )
                self._active_generation = self._generation
                release = getattr(self._scheduler, "release_plan_channel")
                release()
            except BaseException:
                self._handoff_aborted = True
                raise
            ticket = HierarchicalCopyTicket(
                generation=self._generation,
                terminal_flags=self.terminal_flags,
                command_count=commands + 1,
            )
            self._generation += 1
            return ticket

    def close(self) -> None:
        """Release the native copy context after all device work is drained."""
        with self._lock, torch.cuda.device(self.device):
            if self.ctx is None:
                return
            if self._thop_tma:
                torch.ops.trtllm.moe_rebalance_tma_destroy(self.ctx)
            # Non-THOP contexts are PyCapsules whose destructor owns cleanup.
            self.ctx = None
            self._scheduler = None
            self._outputs = None
            self._stream_owner = None

    def last_mode_counts(self) -> tuple[int, int, int]:
        """Return diagnostic ``(direct_slots, scatter_slots, commands)``.

        GPU-direct mode explicitly waits for the copy and reads GPU counters.
        This opt-in diagnostic must stay outside the submission/timing path.
        """

        if self._thop_tma:
            result = torch.ops.trtllm.moe_rebalance_tma_result(self.ctx)
            direct, scatter, commands = result[1], result[2], result[7]
        else:
            direct, scatter, commands = self.mod.last_mode_counts(self.ctx)
        return int(direct), int(scatter), int(commands)


__all__ = [
    "BoundHierarchicalLiveWeightArena",
    "HierarchicalCopyTicket",
    "HierarchicalLiveWeightArena",
    "HierarchicalLiveWeightArenaProvider",
    "HierarchicalPlanChannel",
    "HierarchicalSamiWeightBroadcast",
    "hierarchy_group_sizes",
]
