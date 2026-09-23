# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EP-scoped live-weight storage for HALO-Q and hierarchical SAMI copies.

Each layer packs seven weight planes and uint64[EP] READY terminals into one
allocation. Every multicast level aliases that backing through a distinct
window; the provider retains its allocation and mapping owners. Kernel-facing
and framework-facing views share bytes but use their respective layouts.

Allocation and binding are collective across the EP group. Vendor imports stay
lazy so the disabled path does not initialize the scheduler or compile kernels.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "HierarchicalFabricArenaProvider",
    "allocate_rebalance_arena_v2",
]

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

    #: The allocator that owns these bytes. Annotated ``Any`` because the
    #: hierarchical provider below and any future one are structurally typed
    #: here -- this dataclass only ever holds it and hands it back.
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


class _HierarchicalRegion:
    """Own one backing allocation, peer mappings and multicast levels.

    Packing all seven planes together needs only one multicast handle per level.
    """

    def __init__(
        self,
        nbytes: int,
        group_sizes: Tuple[int, ...],
        world: int,
        rank: int,
        device: int,
        comm: Any,
        owners: List[Any],
    ) -> None:
        import math

        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami.fabric import (
            FabricAllocation,
            MulticastGroup,
        )

        granularities = [
            MulticastGroup.granularity_for(size, nbytes, device)[0] for size in group_sizes
        ]
        alignment = math.lcm(BUFFER_ALIGNMENT, *granularities)
        self.allocation = FabricAllocation.create(nbytes, device, extra_alignment=alignment)
        if any(self.allocation.size % value for value in granularities):
            raise RuntimeError(
                "rebalance arena backing is not granular at every HALO level: "
                f"size={self.allocation.size} granularities={granularities}"
            )

        handles = comm.allgather(self.allocation.shareable)
        if len(handles) != world or any(
            not isinstance(handle, bytes) or len(handle) != 64 for handle in handles
        ):
            raise RuntimeError(
                "fabric handle exchange returned an incomplete EP; every rank "
                "must publish a 64-byte shareable handle"
            )
        self.uc_ptrs: Tuple[int, ...] = tuple(
            self.allocation.local_ptr
            if peer == rank
            else self.allocation.import_peer(handles[peer])
            for peer in range(world)
        )

        level_groups: List[Any] = []
        for group_size in group_sizes:
            # Every rank must visit each subgroup creation in the same order. Retain
            # all owners, including non-member tokens, for the lifetime of mapped views.
            local_group = None
            for begin in range(0, world, group_size):
                members = tuple(range(begin, begin + group_size))
                group = MulticastGroup.create_for_members(
                    world=world,
                    members=members,
                    rank=rank,
                    device=device,
                    size=self.allocation.size,
                    backing=self.allocation,
                    broadcast=comm.bcast,
                    barrier=comm.barrier,
                    all_gather=comm.allgather,
                    owners=owners,
                )
                if group is not None and rank in members:
                    if local_group is not None:
                        raise RuntimeError(f"rank {rank} joined two groups at level {group_size}")
                    local_group = group
            if local_group is None:
                raise RuntimeError(
                    f"rank {rank} did not join its aligned group at level {group_size}"
                )
            level_groups.append(local_group)

        self.level_groups: Tuple[Any, ...] = tuple(level_groups)
        self.group_ptrs: Tuple[int, ...] = tuple(g.mc_ptr for g in level_groups)
        if len(set(self.group_ptrs)) != len(self.group_ptrs):
            # Check allocation size before constructing plane views.
            raise RuntimeError(
                f"hierarchy levels reused one multicast virtual address: {self.group_ptrs}"
            )
        self.nbytes = nbytes


class HierarchicalFabricArenaProvider:
    """Provide idempotent hierarchical arenas and framework storage aliases."""

    def __init__(self, mpi_comm: Any) -> None:
        self._comm = mpi_comm
        self.buffer_bytes = 0
        self.plane_offsets: Dict[str, int] = {}
        self.terminal_offset = 0
        self.region: Optional[_HierarchicalRegion] = None
        self.group_sizes: Tuple[int, ...] = ()
        # Keep mappings and backing memory alive while any tensor view can use them.
        self._pointer_owners: List[_RawPointer] = []
        self._mc_owners: List[Any] = []
        # Memoize the arena and its allocation geometry.
        self._built: Optional[Any] = None
        self._signature: Optional[tuple] = None
        self._tekit_views: Dict[str, torch.Tensor] = {}

    # Expose raw CUDA memory as a flat uint8 tensor.
    def _flat(self, pointer: int, nbytes: int, device: int) -> torch.Tensor:
        holder = _RawPointer(pointer, nbytes)
        self._pointer_owners.append(holder)
        return torch.as_tensor(holder, device=f"cuda:{device}")

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
        from tensorrt_llm._torch.cute_dsl_kernels.megamoe_scheduler_v2.sami import (
            HierarchicalLiveWeightArena,
            LivePlaneView,
            LiveTerminalView,
            LiveWeightArena,
            hierarchy_group_sizes,
        )

        # The broadcaster requests this arena again after Parameters are bound.
        # Return the same object only when all geometry matches; allocating again
        # would send weight copies to storage the model does not read.
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
                    "hierarchical arena provider is per-layer and single-geometry; "
                    f"rebuilt with {signature} after {self._signature}"
                )
            return self._built
        self._signature = signature

        torch.cuda.set_device(device)
        self.group_sizes = hierarchy_group_sizes(world)
        slots = int(home_count) + int(helper_count)

        # Pack all planes into one aligned backing allocation.
        cursor = 0
        for plane in bundle.planes:
            self.plane_offsets[plane.name] = cursor
            cursor += slots * plane.nbytes
        cursor = _align_up(cursor, TERMINAL_ALIGNMENT)
        self.terminal_offset = cursor
        cursor += world * 8
        self.buffer_bytes = _align_up(cursor, BUFFER_ALIGNMENT)

        # Create one multicast mapping per hierarchy level.
        region = _HierarchicalRegion(
            self.buffer_bytes,
            self.group_sizes,
            world,
            rank,
            device,
            comm,
            self._mc_owners,
        )
        self.region = region

        # Build plane views; level zero must be the exact base multicast alias.
        planes = []
        group_views: List[Tuple[torch.Tensor, ...]] = []
        group_owners: List[Tuple[Any, ...]] = []
        for plane in bundle.planes:
            offset = self.plane_offsets[plane.name]
            span = slots * plane.nbytes
            uc_flats = tuple(self._flat(ptr + offset, span, device) for ptr in region.uc_ptrs)
            uc_views = tuple(
                _fork_plane_view(
                    flat, plane.name, plane.nbytes, slots, bundle.hidden, bundle.intermediate
                )
                for flat in uc_flats
            )
            level_views = tuple(
                _fork_plane_view(
                    self._flat(ptr + offset, span, device),
                    plane.name,
                    plane.nbytes,
                    slots,
                    bundle.hidden,
                    bundle.intermediate,
                )
                for ptr in region.group_ptrs
            )
            # Framework aliases use the same local unicast storage as the kernel views.
            self._tekit_views[plane.name] = _tekit_plane_view(
                uc_flats[rank], plane.name, plane.nbytes, slots, bundle.hidden, bundle.intermediate
            )
            planes.append(
                LivePlaneView(
                    name=plane.name,
                    uc_views=uc_views,
                    mc_view=level_views[0],  # Level zero is the base multicast view.
                    aliases_same_backing=True,
                    backing_owner=region,  # anti-GC
                )
            )
            group_views.append(level_views)
            group_owners.append(region.level_groups)

        # Consumers read local terminals; producers publish through global multicast.
        term_span = world * 8
        terminals = LiveTerminalView(
            local_view=self._flat(
                region.uc_ptrs[rank] + self.terminal_offset, term_span, device
            ).view(torch.uint64),
            mc_view=self._flat(region.group_ptrs[0] + self.terminal_offset, term_span, device).view(
                torch.uint64
            ),
            aliases_same_backing=True,
            backing_owner=region,
        )

        base = LiveWeightArena(planes=tuple(planes), terminals=terminals)
        self._built = HierarchicalLiveWeightArena(
            base=base,
            group_sizes=self.group_sizes,
            group_mc_views=tuple(group_views),
            group_mc_owners=tuple(group_owners),
            aliases_same_backing=True,
        )
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
    provider = HierarchicalFabricArenaProvider(mpi)
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
    log_fn = logger.info if layer_idx in (0, None) else logger.debug
    log_fn(
        f"[MegaMoECuteDsl] layer={layer_idx} MoE rebalance HIERARCHICAL live arena: "
        f"{provider.buffer_bytes} B for M={live.slot_count} slots "
        f"(H={home_experts} + S={helper_slots}), ep_size={world} ep_rank={rank} "
        f"device={device}, group_sizes={provider.group_sizes} "
        f"({len(provider.group_sizes)} multicast apertures over ONE backing)."
    )
    return live
