# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EP-scoped live-weight storage for HALO-Q and hierarchical SAMI copies.

Every layer owns its home-expert storage and maps one of two shared helper-slot
banks directly after it in virtual address space. Layers alternate banks, while
per-layer READY terminals and small planes remain independent. Allocation and
binding are collective across the EP group; vendor imports stay lazy until the
rebalance path is enabled.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from weakref import WeakKeyDictionary

import torch

from tensorrt_llm.logger import logger

__all__ = ["allocate_rebalance_arena_v2"]

BUFFER_ALIGNMENT = 2 * 1024 * 1024

#: The ``uint64[EP]`` terminal array is addressed by the SAMI copy kernel and
#: wants its own page.
TERMINAL_ALIGNMENT = 4096


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _plane_dtypes() -> Dict[str, torch.dtype]:
    """Return the canonical kernel-plane dtypes.

    Resolve optional Torch dtypes lazily so unsupported wheels fail only when
    constructing an enabled rebalance arena.
    """
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    fp8 = getattr(torch, "float8_e4m3fn", None)
    if fp4 is None or fp8 is None:
        raise RuntimeError(
            "MoE rebalance live arena needs torch.float4_e2m1fn_x2 and "
            "torch.float8_e4m3fn to type the NVFP4 weight planes; this torch "
            f"({torch.__version__}) exposes float4_e2m1fn_x2="
            f"{fp4 is not None}, float8_e4m3fn={fp8 is not None}."
        )
    return {
        "mega_fc1_weight": fp4,
        "mega_fc1_weight_sf": fp8,
        "mega_fc2_weight": fp4,
        "mega_fc2_weight_sf": fp8,
        "fc31_alpha": torch.float32,
        "fc2_alpha": torch.float32,
        "fc1_norm_const": torch.float32,
    }


def _fork_plane_view(
    flat_u8: torch.Tensor, name: str, nbytes: int, slots: int, hidden: int, intermediate: int
) -> torch.Tensor:
    """Create a kernel-facing view matching the canonical plane ABI.

    Weight views are K-major with the middle dimension contiguous.
    """
    typed = flat_u8.view(_plane_dtypes()[name])
    gate_up = 2 * intermediate
    if name == "mega_fc1_weight":
        return typed.as_strided(
            (slots, hidden // 2, gate_up), (hidden * gate_up // 2, 1, hidden // 2)
        )
    if name == "mega_fc2_weight":
        return typed.as_strided(
            (slots, intermediate // 2, hidden), (intermediate * hidden // 2, 1, intermediate // 2)
        )
    if name.endswith("_sf"):
        return typed.as_strided((slots, nbytes), (nbytes, 1))
    return typed.as_strided((slots,), (1,))


def _tekit_plane_view(
    flat_u8: torch.Tensor, name: str, nbytes: int, slots: int, hidden: int, intermediate: int
) -> torch.Tensor:
    """Create the framework storage view over the same plane bytes.

    Weights use the layout consumed by the loader; the launch path transposes
    them to the kernel layout. Do not substitute a kernel-facing view here.
    """
    gate_up = 2 * intermediate
    if name == "mega_fc1_weight":
        # (M, 2*I, H/2) uint8, contiguous -- matches quantization.py's
        # torch.empty(num_local_slots, expand_intermediate, hidden // 2).
        return flat_u8.view(slots, gate_up, hidden // 2)
    if name == "mega_fc2_weight":
        # (M, H, I/2) uint8, contiguous.
        return flat_u8.view(slots, hidden, intermediate // 2)
    if name.endswith("_sf"):
        # (M, sf_flat_size) uint8, contiguous.
        return flat_u8.view(slots, nbytes)
    # fc31_alpha / fc2_alpha / fc1_norm_const: (M,) float32, identical on both
    # sides, so no bridging is needed -- only the dtype reinterpretation.
    return flat_u8.view(torch.float32)


@dataclass
class RebalanceLiveArena:
    """Own a bound live arena and both views of its seven weight planes.

    Kernel views preserve the allocator's bound identities. Framework aliases
    back nn.Parameters without changing the allocation or plane offsets.
    """

    #: The allocator owns these bytes; the arena only retains and exposes it.
    provider: Any
    arena: Any
    bound: Any
    plane_names: Tuple[str, ...]
    local_plane_views: Tuple[torch.Tensor, ...]
    tekit_alias_views: Tuple[torch.Tensor, ...]
    home_experts: int
    helper_slots: int
    #: READY generation ``g`` currently published into this arena's helper
    #: slots. 0 means "nothing published yet" -- the kernel ABI rejects it
    #: (the gate requires g in [1, 2**63)), which is the intended fail-closed
    #: state before the weight transport has run for this generation.
    ready_generation_value: int = 0

    @property
    def slot_count(self) -> int:
        return self.home_experts + self.helper_slots

    def terminal_flags_tensor(self) -> torch.Tensor:
        """This rank's ``uint64[EP]`` terminal view -- the contract's
        ``hot_expert_weight_ready_flags``. Sources publish into the multicast
        alias of the SAME bytes, so a publish by any EP rank becomes visible
        here without any host-side copy."""
        # Hierarchical arenas expose terminals through their base arena.
        arena = getattr(self.arena, "base", self.arena)
        return arena.terminals.local_view

    def ready_generation(self) -> int:
        return int(self.ready_generation_value)

    def publish_ready_generation(self, generation: int) -> None:
        """Record the generation the kernel must wait for. Called by the
        weight transport AFTER it has issued the payload copies for ``g``;
        the terminal cells themselves are written by the transport."""
        if type(generation) is not int or not (1 <= generation < (1 << 63)):
            raise ValueError(
                f"READY generation must be an exact int in [1, 2**63); got {generation!r}"
            )
        self.ready_generation_value = generation

    def assert_identity(self) -> None:
        """Run the optional allocator identity guard.

        The caller must also check framework parameter pointers: allocator checks
        cannot detect replacement of a Parameter's storage.
        """
        # The allocator guard is optional; framework pointer checks remain mandatory.
        guard = getattr(self.bound, "assert_identity", None)
        if guard is not None:
            guard()


# Reuse one EP communicator per pipeline stage; Split is itself collective.
# Ranks within each stage are ordered by their EP rank.
_EP_MPI_COMMS: Dict[Tuple[int, int], Any] = {}


def _ep_mpi_comm(mapping: Any) -> Any:
    """Cache an EP-scoped MPI communicator per (pipeline rank, EP size)."""
    from tensorrt_llm._utils import mpi_comm

    key = (int(mapping.pp_rank), int(mapping.moe_ep_size))
    comm = _EP_MPI_COMMS.get(key)
    if comm is None:
        comm = mpi_comm().Split(int(mapping.pp_rank), int(mapping.moe_ep_rank))
        _EP_MPI_COMMS[key] = comm
    return comm


class _EpComm:
    """Adapt the EP-scoped MPI communicator to SAMI's collective interface.

    Using the default distributed group could include ranks outside this arena
    and deadlock its collective allocation or binding.
    """

    def __init__(self, ep_comm: Any) -> None:
        self._comm = ep_comm
        self.rank = int(ep_comm.Get_rank())
        self.world = int(ep_comm.Get_size())

    def barrier(self) -> None:
        self._comm.Barrier()

    def bcast(self, payload: object, root: int) -> object:
        return self._comm.bcast(payload, root=int(root))

    def allgather(self, value: object) -> list:
        return list(self._comm.allgather(value))


class _RawPointer:
    """Expose a CUDA address through __cuda_array_interface__.

    Tensor views do not own this allocation; the provider retains its owners.
    """

    def __init__(self, pointer: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "data": (int(pointer), False),
            "shape": (int(nbytes),),
            "typestr": "|u1",
            "strides": None,
            "version": 3,
        }


#: Planes whose whole [H + S] span is at most this many bytes (the fp32 scale
#: planes) live in a per-layer record of a small shared page instead of in
#: their own granularity-sized allocations.
_RECORD_PLANE_BYTES = 64 * 1024
_RECORD_PLANE_ALIGNMENT = 256


_HELPER_BANK_COUNT = 2
_POOL_SCOPE_ATTRIBUTE = "_trtllm_rebalance_shared_slot_scope"


class _PoolScope:
    """Identity key whose copy does not retain CUDA allocation owners."""


class _ViewSpec:
    """Address and layout of a plane view SAMI only binds, never dereferences.

    Peer and multicast views feed SAMI's bind-time pointer tables. A multicast
    view's row 0 lies in a reserved but unmapped prefix, which a torch tensor
    cannot wrap; only its helper rows are mapped and written by the copy.
    """

    __slots__ = ("_pointer", "shape", "_stride", "dtype", "_element_size", "device", "is_cuda")

    def __init__(self, pointer: int, like: torch.Tensor) -> None:
        self._pointer = int(pointer)
        self.shape = tuple(like.shape)
        self._stride = tuple(like.stride())
        self.dtype = like.dtype
        self._element_size = int(like.element_size())
        self.device = like.device
        self.is_cuda = True

    def data_ptr(self) -> int:
        return self._pointer

    def stride(self) -> Tuple[int, ...]:
        return self._stride

    def element_size(self) -> int:
        return self._element_size


class _Vmm:
    """CUDA VMM calls on FABRIC-exportable memory; retains every handle.

    Mappings live for the process and are retained with their owning pool.
    """

    def __init__(self, device: int, group_sizes: Tuple[int, ...]) -> None:
        from cuda.bindings import driver as cuda

        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami._util import check_cuda
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami.fabric import (
            FABRIC,
            _allocation_properties,
            _read_write_access,
        )

        self.cuda = cuda
        self.check = check_cuda
        self.fabric = FABRIC
        self.device = int(device)
        self.properties = _allocation_properties(self.device)
        self._access = [_read_write_access(self.device)]
        # Every chunk is multicast-bindable at every level, so align to the
        # coarsest MINIMUM granularity rather than the 512 MiB recommendation.
        values = [
            BUFFER_ALIGNMENT,
            int(
                check_cuda(
                    cuda.cuMemGetAllocationGranularity(
                        self.properties,
                        cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
                    ),
                    "cuMemGetAllocationGranularity(shared slots)",
                )
            ),
        ]
        for members in sorted(set(group_sizes)):
            values.append(
                int(
                    check_cuda(
                        cuda.cuMulticastGetGranularity(
                            self.multicast_properties(members, BUFFER_ALIGNMENT),
                            cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM,
                        ),
                        "cuMulticastGetGranularity(shared slots)",
                    )
                )
            )
        self.granularity = math.lcm(*values)
        self.handles: List[Any] = []
        self.allocated_bytes = 0

    def multicast_properties(self, members: int, nbytes: int) -> Any:
        properties = self.cuda.CUmulticastObjectProp()
        properties.numDevices = int(members)
        properties.size = int(nbytes)
        properties.handleTypes = int(self.fabric)
        properties.flags = 0
        return properties

    def create(self, nbytes: int) -> Tuple[Any, bytes]:
        handle = self.check(
            self.cuda.cuMemCreate(nbytes, self.properties, 0), "cuMemCreate(shared slots)"
        )
        self.handles.append(handle)
        self.allocated_bytes += nbytes
        exported = self.check(
            self.cuda.cuMemExportToShareableHandle(handle, self.fabric, 0),
            "cuMemExportToShareableHandle(shared slots)",
        )
        shareable = bytes(exported.data)
        if len(shareable) != 64:
            raise RuntimeError(f"fabric handle has {len(shareable)} bytes; expected 64")
        return handle, shareable

    def import_(self, shareable: bytes) -> Any:
        if not isinstance(shareable, bytes) or len(shareable) != 64:
            raise RuntimeError("peer fabric handle must contain exactly 64 bytes")
        handle = self.check(
            self.cuda.cuMemImportFromShareableHandle(bytearray(shareable), self.fabric),
            "cuMemImportFromShareableHandle(shared slots)",
        )
        self.handles.append(handle)
        return handle

    def reserve(self, nbytes: int) -> int:
        return int(
            self.check(
                self.cuda.cuMemAddressReserve(nbytes, self.granularity, 0, 0),
                "cuMemAddressReserve(shared slots)",
            )
        )

    def map(self, address: int, nbytes: int, handle: Any) -> None:
        self.check(self.cuda.cuMemMap(address, nbytes, 0, handle, 0), "cuMemMap(shared slots)")
        self.check(
            self.cuda.cuMemSetAccess(address, nbytes, self._access, 1),
            "cuMemSetAccess(shared slots)",
        )

    def zero(self, address: int, nbytes: int) -> None:
        self.check(
            self.cuda.cuMemsetD8(self.cuda.CUdeviceptr(address), 0, nbytes),
            "cuMemsetD8(shared slots)",
        )


class _MulticastSpec:
    """One aligned group's multicast pair: member binds and local mappings."""

    def __init__(
        self,
        members: Tuple[int, ...],
        nbytes: int,
        binds: List[Tuple[int, Any, int]],
        mappings: List[int],
    ) -> None:
        self.members = members
        self.nbytes = nbytes
        #: (multicast offset, member allocation handle, bytes) bound on every member.
        self.binds = binds
        #: Reserved addresses where this rank maps its selected object.
        self.mappings = mappings
        self.handles: List[Any] = []


def _create_multicast(vmm: _Vmm, comm: Any, rank: int, specs: List[_MulticastSpec]) -> None:
    """Collectively create, bind and map dual-creator multicast objects.

    Follows MulticastTeamPair: the first two members each create one object,
    every member binds its chunks to both, and the first member maps the second
    member's object while every other member maps the first member's. Ranks
    outside a group join only the collectives; specs are rank-ordered alike.
    """
    cuda, check = vmm.cuda, vmm.check
    created: Dict[Tuple[int, int], Any] = {}
    exported: Dict[Tuple[int, int], bytes] = {}
    for index, spec in enumerate(specs):
        for alias, creator in enumerate(spec.members[:2]):
            if rank == creator:
                handle = check(
                    cuda.cuMulticastCreate(
                        vmm.multicast_properties(len(spec.members), spec.nbytes)
                    ),
                    "cuMulticastCreate(shared slots)",
                )
                created[(index, alias)] = handle
                exported[(index, alias)] = bytes(
                    check(
                        cuda.cuMemExportToShareableHandle(handle, vmm.fabric, 0),
                        "cuMemExportToShareableHandle(shared multicast)",
                    ).data
                )
    gathered = comm.allgather(exported)
    joined = [(index, spec) for index, spec in enumerate(specs) if rank in spec.members]
    for index, spec in joined:
        for alias, creator in enumerate(spec.members[:2]):
            handle = created.get((index, alias))
            if handle is None:
                handle = vmm.import_(gathered[creator][(index, alias)])
            else:
                vmm.handles.append(handle)
            spec.handles.append(handle)
            check(
                cuda.cuMulticastAddDevice(handle, vmm.device), "cuMulticastAddDevice(shared slots)"
            )
    comm.barrier()
    for _, spec in joined:
        for handle in spec.handles:
            for offset, memory, nbytes in spec.binds:
                check(
                    cuda.cuMulticastBindMem_v2(handle, vmm.device, offset, memory, 0, nbytes, 0),
                    "cuMulticastBindMem_v2(shared slots)",
                )
    comm.barrier()
    for _, spec in joined:
        selected = spec.handles[1 if len(spec.handles) > 1 and rank == spec.members[0] else 0]
        for address in spec.mappings:
            vmm.map(address, spec.nbytes, selected)
    comm.barrier()


@dataclass
class _HelperSet:
    """One physical helper-slot set: a chunk per large plane, one multicast
    object per hierarchy level, and that object's view base per plane."""

    chunks: Dict[str, Any]
    level_bases: Tuple[Dict[str, int], ...]
    level_specs: Tuple[_MulticastSpec, ...]
    zeroed: bool = False


@dataclass
class _RecordPage:
    """Per-layer records of READY terminals and small planes, mapped on every
    rank and bound to one multicast object per hierarchy level."""

    uc_ptrs: Tuple[int, ...]
    mc_ptrs: Tuple[int, ...]
    level_specs: Tuple[_MulticastSpec, ...]


class _SharedLayer:
    """Anti-GC owner of one layer's home chunks and raw pointer holders."""

    def __init__(self, pool: "_SharedSlotPool", helper_set: int) -> None:
        self.pool = pool
        self.helper_set = helper_set
        self.home_handles: List[Any] = []
        self.holders: List[_RawPointer] = []


class _SharedSlotPool:
    """Helper-slot sets shared by every MoE layer of one geometry.

    Layer ``i`` maps helper set ``i % sets`` directly after its home rows, so
    each plane is still one contiguous ``[H + S]`` kernel view. Only the home
    rows are per layer. No extra synchronization guards reuse: layer ``i``'s
    copy follows HALO-Q's all-rank exchange, which every rank enters only after
    MAIN finished the previous layers' MegaMoE.

    READY terminals stay per layer because one forward shares its generation
    across layers. They and the fp32 planes live in per-layer records.
    """

    def __init__(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: Any,
        device: int,
        comm: Any,
    ) -> None:
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami import (
            hierarchy_group_sizes,
        )

        torch.cuda.set_device(device)
        self.world = world
        self.rank = rank
        self.device = device
        self.home_count = home_count
        self.helper_count = helper_count
        self.slots = home_count + helper_count
        self.bundle = bundle
        self.group_sizes = hierarchy_group_sizes(world)
        self.groups = tuple(
            tuple(range(rank // size * size, rank // size * size + size))
            for size in self.group_sizes
        )
        self.vmm = _Vmm(device, self.group_sizes)
        granularity = self.vmm.granularity

        self.large = tuple(p for p in bundle.planes if p.nbytes * self.slots > _RECORD_PLANE_BYTES)
        self.home_bytes = {
            p.name: _align_up(home_count * p.nbytes, granularity) for p in self.large
        }
        self.helper_bytes = {
            p.name: _align_up(helper_count * p.nbytes, granularity) for p in self.large
        }
        # One multicast object per set and level holds every large plane's chunk.
        self.mc_offsets: Dict[str, int] = {}
        cursor = 0
        for plane in self.large:
            self.mc_offsets[plane.name] = cursor
            cursor += self.helper_bytes[plane.name]
        self.set_bytes = cursor

        # A record keeps its READY terminals on their own page.
        cursor = _align_up(world * 8, TERMINAL_ALIGNMENT)
        self.record_offsets: Dict[str, int] = {}
        for plane in bundle.planes:
            if plane.name not in self.home_bytes:
                cursor = _align_up(cursor, _RECORD_PLANE_ALIGNMENT)
                self.record_offsets[plane.name] = cursor
                cursor += self.slots * plane.nbytes
        self.record_stride = _align_up(cursor, TERMINAL_ALIGNMENT)
        self.page_bytes = _align_up(self.record_stride, granularity)
        self.records_per_page = self.page_bytes // self.record_stride
        self.pages: List[_RecordPage] = []
        self.layer_count = 0
        self.sets = self._create_sets(comm)

    def _create_sets(self, comm: Any) -> Tuple[_HelperSet, ...]:
        if not self.large:
            return tuple(
                _HelperSet({}, tuple({} for _ in self.groups), ())
                for _ in range(_HELPER_BANK_COUNT)
            )
        sets, specs = [], []
        for _ in range(_HELPER_BANK_COUNT):
            chunks = {p.name: self.vmm.create(self.helper_bytes[p.name])[0] for p in self.large}
            level_bases, level_specs = [], []
            for members in self.groups:
                bases: Dict[str, int] = {}
                mappings: List[int] = []
                for plane in self.large:
                    # Row 0 of the view sits H rows before the plane's chunk.
                    prefix = self.home_bytes[plane.name]
                    address = self.vmm.reserve(prefix + self.set_bytes)
                    mappings.append(address + prefix)
                    bases[plane.name] = (
                        address
                        + prefix
                        + self.mc_offsets[plane.name]
                        - self.home_count * plane.nbytes
                    )
                spec = _MulticastSpec(
                    members,
                    self.set_bytes,
                    [
                        (self.mc_offsets[p.name], chunks[p.name], self.helper_bytes[p.name])
                        for p in self.large
                    ],
                    mappings,
                )
                specs.append(spec)
                level_specs.append(spec)
                level_bases.append(bases)
            sets.append(_HelperSet(chunks, tuple(level_bases), tuple(level_specs)))
        _create_multicast(self.vmm, comm, self.rank, specs)
        return tuple(sets)

    def _create_page(self, comm: Any) -> _RecordPage:
        handle, shareable = self.vmm.create(self.page_bytes)
        local = self.vmm.reserve(self.page_bytes)
        self.vmm.map(local, self.page_bytes, handle)
        self.vmm.zero(local, self.page_bytes)
        torch.cuda.synchronize(self.device)
        shareables = comm.allgather(shareable)
        if len(shareables) != self.world:
            raise RuntimeError("shared-slot record exchange returned an incomplete EP")
        uc_ptrs = []
        for peer in range(self.world):
            if peer == self.rank:
                uc_ptrs.append(local)
                continue
            address = self.vmm.reserve(self.page_bytes)
            self.vmm.map(address, self.page_bytes, self.vmm.import_(shareables[peer]))
            uc_ptrs.append(address)
        mc_ptrs = [self.vmm.reserve(self.page_bytes) for _ in self.groups]
        specs = [
            _MulticastSpec(members, self.page_bytes, [(0, handle, self.page_bytes)], [address])
            for members, address in zip(self.groups, mc_ptrs)
        ]
        _create_multicast(self.vmm, comm, self.rank, specs)
        return _RecordPage(tuple(uc_ptrs), tuple(mc_ptrs), tuple(specs))

    def _flat(self, owner: _SharedLayer, pointer: int, nbytes: int) -> torch.Tensor:
        holder = _RawPointer(pointer, nbytes)
        owner.holders.append(holder)
        return torch.as_tensor(holder, device=f"cuda:{self.device}")

    def build_layer(self, comm: Any) -> Tuple[Any, Dict[str, torch.Tensor], _SharedLayer, int]:
        """Collectively map one layer; returns its arena, framework views,
        owner and own allocation bytes."""
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami import (
            HierarchicalLiveWeightArena,
            LivePlaneView,
            LiveTerminalView,
            LiveWeightArena,
        )

        world, rank, slots = self.world, self.rank, self.slots
        hidden, intermediate = self.bundle.hidden, self.bundle.intermediate
        index = self.layer_count
        if index % self.records_per_page == 0:
            self.pages.append(self._create_page(comm))
        page = self.pages[-1]
        record = (index % self.records_per_page) * self.record_stride
        helper_set = self.sets[index % len(self.sets)]
        layer = _SharedLayer(self, index % len(self.sets))
        self.layer_count += 1

        homes = {p.name: self.vmm.create(self.home_bytes[p.name]) for p in self.large}
        layer.home_handles = [homes[p.name][0] for p in self.large]
        peer_handles = comm.allgather(tuple(homes[p.name][1] for p in self.large))
        if len(peer_handles) != world:
            raise RuntimeError("shared-slot home exchange returned an incomplete EP")

        planes, group_views, group_owners = [], [], []
        tekit_views: Dict[str, torch.Tensor] = {}
        for plane in self.bundle.planes:
            span = slots * plane.nbytes
            if plane.name in self.home_bytes:
                large_index = [p.name for p in self.large].index(plane.name)
                home_bytes = self.home_bytes[plane.name]
                helper_bytes = self.helper_bytes[plane.name]
                # Home rows end where the helper chunk begins.
                pad = home_bytes - self.home_count * plane.nbytes
                address = self.vmm.reserve(home_bytes + helper_bytes)
                self.vmm.map(address, home_bytes, homes[plane.name][0])
                self.vmm.map(address + home_bytes, helper_bytes, helper_set.chunks[plane.name])
                self.vmm.zero(address, home_bytes)
                if not helper_set.zeroed:
                    self.vmm.zero(address + home_bytes, helper_bytes)
                uc_ptrs = []
                for peer in range(world):
                    if peer == rank:
                        uc_ptrs.append(address + pad)
                        continue
                    # Peers expose only home rows; the reservation keeps the
                    # view's helper tail clear of every other mapping.
                    peer_address = self.vmm.reserve(home_bytes + helper_bytes)
                    self.vmm.map(
                        peer_address,
                        home_bytes,
                        self.vmm.import_(peer_handles[peer][large_index]),
                    )
                    uc_ptrs.append(peer_address + pad)
                level_ptrs = tuple(bases[plane.name] for bases in helper_set.level_bases)
                owners = helper_set.level_specs
            else:
                offset = record + self.record_offsets[plane.name]
                uc_ptrs = [pointer + offset for pointer in page.uc_ptrs]
                level_ptrs = tuple(pointer + offset for pointer in page.mc_ptrs)
                owners = page.level_specs
            flat = self._flat(layer, uc_ptrs[rank], span)
            kernel_view = _fork_plane_view(
                flat, plane.name, plane.nbytes, slots, hidden, intermediate
            )
            uc_views = tuple(
                kernel_view if peer == rank else _ViewSpec(pointer, kernel_view)
                for peer, pointer in enumerate(uc_ptrs)
            )
            level_views = tuple(_ViewSpec(pointer, kernel_view) for pointer in level_ptrs)
            tekit_views[plane.name] = _tekit_plane_view(
                flat, plane.name, plane.nbytes, slots, hidden, intermediate
            )
            planes.append(
                LivePlaneView(
                    name=plane.name,
                    uc_views=uc_views,
                    mc_view=level_views[0],
                    aliases_same_backing=True,
                    backing_owner=layer,
                )
            )
            group_views.append(level_views)
            group_owners.append(tuple(owners))
        helper_set.zeroed = True

        terminals = LiveTerminalView(
            local_view=self._flat(layer, page.uc_ptrs[rank] + record, world * 8).view(torch.uint64),
            mc_view=self._flat(layer, page.mc_ptrs[0] + record, world * 8).view(torch.uint64),
            aliases_same_backing=True,
            backing_owner=layer,
        )
        arena = HierarchicalLiveWeightArena(
            base=LiveWeightArena(planes=tuple(planes), terminals=terminals),
            group_sizes=self.group_sizes,
            group_mc_views=tuple(group_views),
            group_mc_owners=tuple(group_owners),
            aliases_same_backing=True,
        )
        return arena, tekit_views, layer, sum(self.home_bytes.values())


_SHARED_SLOT_POOLS: WeakKeyDictionary[_PoolScope, Dict[tuple, _SharedSlotPool]] = (
    WeakKeyDictionary()
)


class SharedSlotArenaProvider:
    """Provide one layer's arena over the two shared helper-slot banks.

    The broadcaster requests the arena again after Parameters are bound, so a
    provider memoizes its first build and requires identical geometry.
    """

    def __init__(self, owner: Any) -> None:
        # Scope physical helper banks to the model Mapping instance. A
        # process-global geometry cache would let two engines overwrite each
        # the slots of another engine when their shapes happen to match.
        self._owner = owner
        scope = getattr(owner, _POOL_SCOPE_ATTRIBUTE, None)
        if scope is None:
            scope = _PoolScope()
            setattr(owner, _POOL_SCOPE_ATTRIBUTE, scope)
        elif not isinstance(scope, _PoolScope):
            raise RuntimeError(f"Mapping attribute {_POOL_SCOPE_ATTRIBUTE!r} is already in use")
        self._scope = scope
        self._pools = _SHARED_SLOT_POOLS.setdefault(scope, {})
        self.buffer_bytes = 0
        self.group_sizes: Tuple[int, ...] = ()
        self.pool: Optional[_SharedSlotPool] = None
        self.layer: Optional[_SharedLayer] = None
        self._built: Optional[Any] = None
        self._signature: Optional[tuple] = None
        self._tekit_views: Dict[str, torch.Tensor] = {}

    def build_hierarchical_live_arena(
        self,
        *,
        world: int,
        rank: int,
        home_count: int,
        helper_count: int,
        bundle: Any,
        device: int,
        comm: Any,
    ) -> Any:
        signature = (
            int(world),
            int(rank),
            int(home_count),
            int(helper_count),
            int(bundle.hidden),
            int(bundle.intermediate),
            int(device),
            tuple((pl.name, pl.nbytes) for pl in bundle.planes),
        )
        if self._built is not None:
            if self._signature != signature:
                raise RuntimeError(
                    "shared-slot arena provider is per-layer and single-geometry; "
                    f"rebuilt with {signature} after {self._signature}"
                )
            return self._built
        self._signature = signature
        key = signature
        pool = self._pools.get(key)
        if pool is None:
            pool = _SharedSlotPool(
                world=int(world),
                rank=int(rank),
                home_count=int(home_count),
                helper_count=int(helper_count),
                bundle=bundle,
                device=int(device),
                comm=comm,
            )
            self._pools[key] = pool
        self._built, self._tekit_views, self.layer, self.buffer_bytes = pool.build_layer(comm)
        self.pool = pool
        self.group_sizes = pool.group_sizes
        return self._built

    def tekit_alias(self, name: str) -> torch.Tensor:
        """Return this rank's framework storage view for a weight plane."""
        return self._tekit_views[name]


def allocate_rebalance_arena_v2(
    *,
    home_experts: int,
    helper_slots: int,
    hidden_size: int,
    intermediate_size: int,
    mapping: Any,
    device: int,
    layer_idx: Optional[int],
) -> RebalanceLiveArena:
    """Allocate and bind one layer's arena collectively across its EP group.

    Every rank must call this with identical geometry and collective order.
    """
    from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami.geometry import BundleLayout

    world = int(mapping.moe_ep_size)
    rank = int(mapping.moe_ep_rank)
    if world < 2:
        raise RuntimeError(f"MoE rebalance needs at least two EP ranks; got moe_ep_size={world}.")

    bundle = BundleLayout.create(hidden=int(hidden_size), intermediate=int(intermediate_size))
    mpi = _ep_mpi_comm(mapping)
    provider = SharedSlotArenaProvider(mapping)
    arena = provider.build_hierarchical_live_arena(
        world=world,
        rank=rank,
        home_count=int(home_experts),
        helper_count=int(helper_slots),
        bundle=bundle,
        device=int(device),
        comm=_EpComm(mpi),
    )
    # Initialize READY terminals explicitly before peers can observe them.
    # This startup-only reset does not depend on allocator initialization details.
    arena.base.terminals.local_view.zero_()
    torch.cuda.synchronize()
    mpi.Barrier()
    bound = arena.bind(
        world=world,
        rank=rank,
        home_count=int(home_experts),
        helper_count=int(helper_slots),
        bundle=bundle,
        device=int(device),
    )
    plane_names = tuple(plane.name for plane in bundle.planes)
    # Retain the hierarchical arena and its bound base views.
    live = RebalanceLiveArena(
        provider=provider,
        arena=arena,
        bound=bound,
        plane_names=plane_names,
        # The bound hierarchy exposes local plane views through its base arena.
        local_plane_views=tuple(bound.base.local_plane_views),
        tekit_alias_views=tuple(provider.tekit_alias(name) for name in plane_names),
        home_experts=int(home_experts),
        helper_slots=int(helper_slots),
    )
    pool = provider.pool
    layer = provider.layer
    assert pool is not None and layer is not None
    verbose = pool.layer_count <= len(pool.sets) or pool.layer_count % 20 == 0
    log_fn = logger.info if verbose else logger.debug
    log_fn(
        f"[MegaMoECuteDsl] layer={layer_idx} MoE rebalance SHARED-SLOT live arena: "
        f"{provider.buffer_bytes} B of home rows for M={live.slot_count} slots "
        f"(H={home_experts} + S={helper_slots}) on helper set "
        f"{layer.helper_set}/{len(pool.sets)}; pool sets "
        f"{len(pool.sets)} x {pool.set_bytes} B, record pages {len(pool.pages)} x "
        f"{pool.page_bytes} B, granularity {pool.vmm.granularity} B, VMM total "
        f"{pool.vmm.allocated_bytes} B after {pool.layer_count} layers; ep_size={world} "
        f"ep_rank={rank} device={device}, group_sizes={provider.group_sizes}."
    )
    return live
