"""CUDA Fabric and NVSwitch multicast memory.

ON A RUNTIME PATH -- do not move this module out of the shipped package.

It was moved to bench/ once, on the reasoning that "the shipped scheduler
allocates its live banks through the framework, so none of this is on a runtime
path". That is wrong. The framework allocates them *with this module*: the
consumer implements HierarchicalLiveWeightArenaProvider and, inside
build_hierarchical_live_arena, imports FabricAllocation and MulticastGroup back
out of here. In TensorRT-LLM that is rebalance_live_arena_v2.py, which reaches
it through _HierarchicalRegion.__init__.

That import sits inside a function body on purpose because importing this package
at module scope triggers an eager native build. A module-level grep or AST
closure therefore does not see it and may conclude that nothing uses this. Walk
consumers with ast.walk before touching it.

megamoe_scheduler/sami/_driver.py carries the one helper shared with production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from cuda.bindings import driver as cuda

from ._util import check_cuda


FABRIC = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _read_write_access(device: int) -> cuda.CUmemAccessDesc:
    descriptor = cuda.CUmemAccessDesc()
    descriptor.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    descriptor.location.id = int(device)
    descriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return descriptor


def _allocation_properties(device: int) -> cuda.CUmemAllocationProp:
    properties = cuda.CUmemAllocationProp()
    properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = int(device)
    properties.requestedHandleTypes = FABRIC
    return properties


@dataclass
class FabricAllocation:
    """One exported device allocation plus the peer mappings imported from it."""

    device: int
    size: int
    granularity: int
    handle: object
    local_ptr: int
    shareable: bytes
    _imported_handles: list[object] = field(default_factory=list, repr=False)

    @classmethod
    def create(
        cls, nbytes: int, device: int, *, extra_alignment: int = 1
    ) -> "FabricAllocation":
        if nbytes <= 0:
            raise ValueError("allocation size must be positive")
        properties = _allocation_properties(device)
        granularity = int(
            check_cuda(
                cuda.cuMemGetAllocationGranularity(
                    properties,
                    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        alignment = math.lcm(granularity, max(1, int(extra_alignment)))
        size = _align_up(nbytes, alignment)
        handle = check_cuda(cuda.cuMemCreate(size, properties, 0), "cuMemCreate")
        allocation = cls(device, size, alignment, handle, 0, b"")
        allocation.local_ptr = allocation._map(handle)
        check_cuda(
            cuda.cuMemsetD8(cuda.CUdeviceptr(allocation.local_ptr), 0, size),
            "cuMemsetD8",
        )
        exported = check_cuda(
            cuda.cuMemExportToShareableHandle(handle, FABRIC, 0),
            "cuMemExportToShareableHandle",
        )
        allocation.shareable = bytes(exported.data)
        if len(allocation.shareable) != 64:
            raise RuntimeError(
                f"fabric handle has {len(allocation.shareable)} bytes; expected 64"
            )
        return allocation

    def _map(self, handle: object) -> int:
        address = check_cuda(
            cuda.cuMemAddressReserve(self.size, self.granularity, 0, 0),
            "cuMemAddressReserve",
        )
        check_cuda(cuda.cuMemMap(address, self.size, 0, handle, 0), "cuMemMap")
        check_cuda(
            cuda.cuMemSetAccess(
                address,
                self.size,
                [_read_write_access(self.device)],
                1,
            ),
            "cuMemSetAccess",
        )
        return int(address)

    def import_peer(self, shareable: bytes) -> int:
        if len(shareable) != 64:
            raise ValueError("peer fabric handle must contain exactly 64 bytes")
        handle = check_cuda(
            cuda.cuMemImportFromShareableHandle(bytearray(shareable), FABRIC),
            "cuMemImportFromShareableHandle",
        )
        self._imported_handles.append(handle)
        return self._map(handle)


@dataclass(frozen=True)
class MulticastGroup:
    """All-member NVSwitch multicast aperture bound to each rank's backing."""

    device: int
    size: int
    granularity: int
    handle: object
    mc_ptr: int
    creator_rank: int
    backing_owner: FabricAllocation = field(repr=False, compare=False)

    alias_set: "MulticastTeamPair | None" = field(default=None, repr=False, compare=False)

    @property
    def aliases(self) -> tuple:
        return self.alias_set.aliases if self.alias_set is not None else ()

    @property
    def closed(self) -> bool:
        return self.alias_set.closed if self.alias_set is not None else False

    def metadata(self) -> dict:
        if self.alias_set is not None:
            return self.alias_set.metadata()
        return {"device": self.device, "size": self.size,
                "creator_rank": self.creator_rank, "mc_ptr": self.mc_ptr,
                "alias_count": 1, "dual_creator": False, "closed": False}

    def close_collectively(self) -> bool:
        """Close a managed default group on every EP rank via its owners list.

        Explicit-creator legacy groups retain their original lifetime contract.
        """
        if self.alias_set is None:
            raise RuntimeError("explicit-creator groups have no managed pair lifecycle")
        success = self.alias_set.close_collectively()
        if success:
            object.__setattr__(self, "mc_ptr", 0)
            object.__setattr__(self, "handle", None)
        return success

    @staticmethod
    def granularity_for(world: int, nbytes: int, device: int) -> tuple[int, int]:
        properties = cuda.CUmulticastObjectProp()
        properties.numDevices = world
        properties.size = nbytes
        properties.handleTypes = int(FABRIC)
        properties.flags = 0
        multicast = int(
            check_cuda(
                cuda.cuMulticastGetGranularity(
                    properties,
                    cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
                ),
                "cuMulticastGetGranularity",
            )
        )
        allocation = int(
            check_cuda(
                cuda.cuMemGetAllocationGranularity(
                    _allocation_properties(device),
                    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                ),
                "cuMemGetAllocationGranularity",
            )
        )
        return multicast, allocation

    @classmethod
    def create(
        cls,
        *,
        world: int,
        rank: int,
        creator_rank: int,
        device: int,
        size: int,
        backing: FabricAllocation,
        broadcast: Callable[[object, int], object],
        barrier: Callable[[], None],
    ) -> "MulticastGroup":
        properties = cuda.CUmulticastObjectProp()
        properties.numDevices = world
        properties.size = size
        properties.handleTypes = int(FABRIC)
        properties.flags = 0
        granularity = int(
            check_cuda(
                cuda.cuMulticastGetGranularity(
                    properties,
                    cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
                ),
                "cuMulticastGetGranularity",
            )
        )
        if size % granularity:
            raise ValueError("multicast size is not aligned to its granularity")
        if backing.size < size:
            raise ValueError("backing allocation is smaller than multicast object")

        serialized = None
        if rank == creator_rank:
            handle = check_cuda(
                cuda.cuMulticastCreate(properties), "cuMulticastCreate"
            )
            exported = check_cuda(
                cuda.cuMemExportToShareableHandle(handle, FABRIC, 0),
                "cuMemExportToShareableHandle(multicast)",
            )
            serialized = bytes(exported.data)
        serialized = broadcast(serialized, creator_rank)
        if rank != creator_rank:
            handle = check_cuda(
                cuda.cuMemImportFromShareableHandle(bytearray(serialized), FABRIC),
                "cuMemImportFromShareableHandle(multicast)",
            )

        check_cuda(cuda.cuMulticastAddDevice(handle, device), "cuMulticastAddDevice")
        barrier()
        check_cuda(
            cuda.cuMulticastBindMem_v2(
                handle, device, 0, backing.handle, 0, size, 0
            ),
            "cuMulticastBindMem_v2",
        )
        address = check_cuda(
            cuda.cuMemAddressReserve(size, granularity, 0, 0),
            "cuMemAddressReserve(multicast)",
        )
        check_cuda(cuda.cuMemMap(address, size, 0, handle, 0), "cuMemMap(multicast)")
        check_cuda(
            cuda.cuMemSetAccess(
                address,
                size,
                [_read_write_access(device)],
                1,
            ),
            "cuMemSetAccess(multicast)",
        )
        barrier()
        return cls(
            device,
            size,
            granularity,
            handle,
            int(address),
            creator_rank,
            backing,
        )

    @classmethod
    def create_for_members(
        cls, *, world: int, members: tuple[int, ...], rank: int,
        creator_rank: int | None = None, device: int, size: int,
        backing: FabricAllocation, broadcast: Callable[[object, int], object],
        barrier: Callable[[], None], all_gather: Callable | None = None,
        owners: list | None = None,
    ) -> "MulticastGroup | None":
        """Default to two creators sharing backing; preserve explicit placement.

        The first member uses the second member's alias. Every other member
        uses the first member's alias. A singleton creates one object. Pass an
        ``owners`` list on every EP rank to retain even nonmember lifecycle
        tokens; close these pairs in reverse global creation order.
        """
        if creator_rank is not None:
            return cls._create_single_for_members(world=world, members=members,
                rank=rank, creator_rank=creator_rank, device=device, size=size,
                backing=backing, broadcast=broadcast, barrier=barrier)
        # These two values define the fallback control communicator itself.
        if type(world) is not int or world <= 0:
            raise ValueError("world must be a positive exact int")
        if type(rank) is not int or not 0 <= rank < world:
            raise ValueError("rank must be in [0, world)")
        if (type(members) is not tuple or not members or
                any(type(member) is not int for member in members)):
            raise ValueError("multicast members must be a non-empty tuple of ranks")
        count, begin = len(members), members[0]
        if members != tuple(range(begin, begin + count)):
            raise ValueError("multicast members must be contiguous sorted ranks")
        if begin < 0 or members[-1] >= world:
            raise ValueError("multicast members must be in [0, world)")
        if world % count or begin % count:
            raise ValueError("multicast members must form an aligned EP subgroup")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a non-negative exact int")
        if type(size) is not int or size <= 0:
            raise ValueError("multicast size must be a positive exact int")
        if owners is not None and type(owners) is not list:
            raise TypeError("owners must be a list")
        if not callable(broadcast) or not callable(barrier):
            raise TypeError("broadcast and barrier must be callable")
        if all_gather is None:
            def all_gather(value):
                return [broadcast(value if rank == root else None, root)
                        for root in range(world)]
        if not callable(all_gather):
            raise TypeError("all_gather must be callable")

        def validate_aligned_group():
            if not isinstance(backing, FabricAllocation):
                raise TypeError("backing must be a FabricAllocation")
            if backing.device != device or backing.size < size:
                raise ValueError("backing device or size differs from multicast")
            if rank in members:
                granularity = cls.granularity_for(count, size, device)[0]
                if granularity <= 0 or size % granularity:
                    raise ValueError("subgroup multicast size is not granular")

        try:
            pair = MulticastTeam.create_dual(world=world, rank=rank, device=device,
                members=members, size=size, all_gather=all_gather,
                backing=backing if rank in members else None,
                _validate=validate_aligned_group)
        except MulticastTeamError as exc:
            if owners is not None:
                owners.append(exc.team)
            raise
        if owners is not None:
            owners.append(pair)
        if rank not in members:
            return None
        return cls(device, size, pair.granularity, pair.handle, pair.mc_ptr,
                   pair.creator_rank, backing, pair)

    @classmethod
    def _create_single_for_members(
        cls,
        *,
        world: int,
        members: tuple[int, ...],
        rank: int,
        creator_rank: int | None = None,
        device: int,
        size: int,
        backing: FabricAllocation,
        broadcast: Callable[[object, int], object],
        barrier: Callable[[], None],
    ) -> "MulticastGroup | None":
        """Collectively create one aligned subgroup multicast alias.

        Every EP rank calls this method in the same order so setup may keep its
        existing world control communicator.  Only ``members`` add a device,
        bind their local backing, and map the multicast virtual address.
        An omitted creator selects the second member (the sole member for a
        singleton); an explicit creator preserves caller placement. This is
        a deterministic placement policy, not a topology-independent guarantee.
        """

        if type(world) is not int or world <= 0:
            raise ValueError("world must be a positive exact int")
        if type(rank) is not int or not 0 <= rank < world:
            raise ValueError("rank must be in [0, world)")
        if (
            not isinstance(members, tuple)
            or not members
            or any(type(member) is not int for member in members)
        ):
            raise ValueError("multicast members must be a non-empty tuple of ranks")
        group_size = len(members)
        group_begin = members[0]
        if members != tuple(range(group_begin, group_begin + group_size)):
            raise ValueError("multicast members must be contiguous sorted ranks")
        if group_begin < 0 or members[-1] >= world:
            raise ValueError("multicast members must be in [0, world)")
        if world % group_size or group_begin % group_size:
            raise ValueError("multicast members must form an aligned EP subgroup")
        if creator_rank is None:
            creator_rank = members[1] if group_size > 1 else members[0]
        if type(creator_rank) is not int:
            raise ValueError("multicast creator must be an exact rank")
        if creator_rank not in members:
            raise ValueError("multicast creator must belong to the subgroup")
        if type(device) is not int or device < 0:
            raise ValueError("device must be a non-negative exact int")
        if type(size) is not int or size <= 0:
            raise ValueError("multicast size must be a positive exact int")
        if not isinstance(backing, FabricAllocation):
            raise TypeError("backing must be a FabricAllocation")
        if backing.device != device:
            raise ValueError("backing and multicast devices differ")
        if not callable(broadcast) or not callable(barrier):
            raise TypeError("broadcast and barrier must be callable")
        properties = cuda.CUmulticastObjectProp()
        properties.numDevices = len(members)
        properties.size = size
        properties.handleTypes = int(FABRIC)
        properties.flags = 0
        granularity = int(
            check_cuda(
                cuda.cuMulticastGetGranularity(
                    properties,
                    cuda.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_RECOMMENDED,
                ),
                "cuMulticastGetGranularity(subgroup)",
            )
        )
        if size % granularity:
            raise ValueError("subgroup multicast size is not granular")
        if backing.size < size:
            raise ValueError("backing allocation is smaller than subgroup multicast")

        serialized = None
        handle = None
        if rank == creator_rank:
            handle = check_cuda(
                cuda.cuMulticastCreate(properties),
                "cuMulticastCreate(subgroup)",
            )
            exported = check_cuda(
                cuda.cuMemExportToShareableHandle(handle, FABRIC, 0),
                "cuMemExportToShareableHandle(subgroup)",
            )
            serialized = bytes(exported.data)
        serialized = broadcast(serialized, creator_rank)
        try:
            serialized = bytes(serialized)
        except (TypeError, ValueError) as exc:
            raise TypeError("broadcast returned an invalid multicast handle") from exc
        if len(serialized) != 64:
            raise RuntimeError(
                f"subgroup fabric handle has {len(serialized)} bytes; expected 64"
            )
        is_member = rank in members
        if is_member and rank != creator_rank:
            handle = check_cuda(
                cuda.cuMemImportFromShareableHandle(bytearray(serialized), FABRIC),
                "cuMemImportFromShareableHandle(subgroup)",
            )
        if is_member:
            check_cuda(
                cuda.cuMulticastAddDevice(handle, device),
                "cuMulticastAddDevice(subgroup)",
            )
        barrier()
        address = 0
        if is_member:
            check_cuda(
                cuda.cuMulticastBindMem_v2(
                    handle, device, 0, backing.handle, 0, size, 0
                ),
                "cuMulticastBindMem_v2(subgroup)",
            )
            address = check_cuda(
                cuda.cuMemAddressReserve(size, granularity, 0, 0),
                "cuMemAddressReserve(subgroup)",
            )
            check_cuda(
                cuda.cuMemMap(address, size, 0, handle, 0),
                "cuMemMap(subgroup)",
            )
            check_cuda(
                cuda.cuMemSetAccess(address, size, [_read_write_access(device)], 1),
                "cuMemSetAccess(subgroup)",
            )
        barrier()
        if not is_member:
            return None
        return cls(
            device,
            size,
            granularity,
            handle,
            int(address),
            creator_rank,
            backing,
        )


def pick_multicast_creator(rank_to_node: list[int], source_rank: int = 0) -> int:
    """Select a creator off source rank's node when the group spans nodes."""
    source_node = rank_to_node[source_rank]
    for rank, node in enumerate(rank_to_node):
        if node != source_node:
            return rank
    return source_rank


__all__ = [
    "FabricAllocation",
    "MulticastGroup",
    "check_cuda",
    "pick_multicast_creator",
]


_RETAINED_MULTICAST_TEAMS: list[object] = []


class MulticastTeamError(RuntimeError):
    """A collective team operation failed; keep its partial resource graph."""

    def __init__(self, message: str, team: "MulticastTeam", failures: list) -> None:
        super().__init__(message)
        self.team, self.failures = team, failures


class MulticastTeam:
    """Explicit arbitrary-member multicast team with collective teardown.

    Every EP rank participates in identical create/close calls. Members with a
    backing borrow it; members without one own a fresh dummy. Nonmembers never
    allocate or map. Callers must release consumers before collective close.
    There is deliberately no destructor that starts a distributed collective.
    """

    def __init__(self, *, world, rank, device, members, creator_rank, size,
                 all_gather, backing) -> None:
        self.world, self.rank, self.device = world, rank, device
        self.members, self.creator_rank, self.size = members, creator_rank, size
        self.all_gather, self.backing_owner = all_gather, backing
        self._borrowed = backing is not None
        self.is_member = self.bound = self.mc_mapped = self._uc_mapped = False
        self.handle = self._owned_handle = None
        self.mc_ptr = self.local_ptr = self._owned_ptr = 0
        self.closed = self.retained_after_cleanup_failure = False
        self.granularities = None

    @property
    def owns_backing(self) -> bool:
        return self.is_member and not self._borrowed

    def _call(self, name, *args):
        return check_cuda(getattr(cuda, name)(*args), name)

    def _phase(self, label, operation, *, cleanup=False):
        value, error = None, None
        try:
            value = operation()
        except Exception as exc:
            error = {"rank": self.rank, "type": type(exc).__name__, "message": str(exc)}
        rows = self.all_gather({"rank": self.rank, "world": self.world,
                                "phase": label, "error": error})
        failures = [row["error"] for row in rows if row["error"] is not None]
        if ([row["rank"] for row in rows] != list(range(len(rows))) or
                any(row["world"] != len(rows) or row["phase"] != label for row in rows)):
            failures.append({"message": "multicast team collective phase/rank mismatch"})
        if failures and not cleanup:
            raise MulticastTeamError(label + ": " + repr(failures), self, failures)
        return not failures if cleanup else value

    @staticmethod
    def granularities_for(member_count: int, nbytes: int, device: int) -> dict:
        """Query before allocation; sizes are never silently enlarged by create."""
        if any(type(x) is not int for x in (member_count, nbytes, device)) or min(member_count, nbytes) <= 0 or device < 0:
            raise ValueError("team geometry requires positive exact counts and a nonnegative device")
        supported = check_cuda(cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device),
            "cuDeviceGetAttribute(multicast)")
        if int(supported) != 1:
            raise RuntimeError("team device does not support multicast")
        prop = cuda.CUmulticastObjectProp()
        prop.numDevices, prop.size, prop.handleTypes, prop.flags = member_count, nbytes, int(FABRIC), 0
        cm, ca = cuda.CUmulticastGranularity_flags, cuda.CUmemAllocationGranularity_flags
        values = {"mc_minimum": int(check_cuda(cuda.cuMulticastGetGranularity(prop, cm.CU_MULTICAST_GRANULARITY_MINIMUM))),
                  "mc_recommended": int(check_cuda(cuda.cuMulticastGetGranularity(prop, cm.CU_MULTICAST_GRANULARITY_RECOMMENDED))),
                  "allocation_minimum": int(check_cuda(cuda.cuMemGetAllocationGranularity(_allocation_properties(device), ca.CU_MEM_ALLOC_GRANULARITY_MINIMUM))),
                  "allocation_recommended": int(check_cuda(cuda.cuMemGetAllocationGranularity(_allocation_properties(device), ca.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)))}
        if any(x <= 0 or x & (x - 1) for x in values.values()):
            raise ValueError("invalid CUDA team granularity")
        return values

    def _validate(self):
        if type(self.world) is not int or self.world <= 0:
            raise ValueError("world must be a positive exact int")
        if type(self.rank) is not int or not 0 <= self.rank < self.world:
            raise ValueError("rank must be an exact EP rank")
        if type(self.device) is not int or self.device < 0:
            raise ValueError("device must be a nonnegative exact ordinal")
        if (type(self.members) is not tuple or not self.members or
                any(type(x) is not int for x in self.members) or
                self.members != tuple(sorted(set(self.members))) or
                any(not 0 <= x < self.world for x in self.members)):
            raise ValueError("members must be sorted unique exact ranks within EP")
        if type(self.creator_rank) is not int or self.creator_rank not in self.members:
            raise ValueError("creator must be an exact member rank")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("size must be a positive exact byte count")
        self.is_member = self.rank in self.members
        if not self.is_member and self.backing_owner is not None:
            raise ValueError("nonmembers cannot provide backing")
        if self.backing_owner is not None:
            if not isinstance(self.backing_owner, FabricAllocation):
                raise TypeError("borrowed backing must be a FabricAllocation")
            if self.backing_owner.device != self.device or self.backing_owner.size < self.size or not self.backing_owner.local_ptr:
                raise ValueError("borrowed backing device, size or UC mapping mismatch")
            self.local_ptr = int(self.backing_owner.local_ptr)

    def _initialize(self):
        self._phase("validate team arguments", self._validate)
        configs = self.all_gather((self.world, self.members, self.creator_rank, self.size))
        def uniform():
            if any(row != configs[0] for row in configs):
                raise ValueError("multicast team configuration differs across ranks")
        self._phase("uniform team configuration", uniform)
        def granularity():
            if self.is_member:
                self.granularities = self.granularities_for(len(self.members), self.size, self.device)
                if self.size % self.granularities["mc_minimum"] or self.size % self.granularities["allocation_minimum"]:
                    raise ValueError("explicit size is not granular; no automatic enlargement")
            return self.granularities
        values = self.all_gather(self._phase("team capability and granularity", granularity))
        self.granularity = max(max(row.values()) for row in values if row is not None)
        def own_backing():
            if self.owns_backing:
                self._owned_handle = self._call("cuMemCreate", self.size, _allocation_properties(self.device), 0)
                self.backing_owner = FabricAllocation(self.device, self.size, self.granularity, self._owned_handle, 0, b"")
        self._phase("fresh or borrowed backing", own_backing)
        def reserve_uc():
            if self.owns_backing:
                self._owned_ptr = int(self._call("cuMemAddressReserve", self.size, self.granularity, 0, 0))
        self._phase("owned UC reservation", reserve_uc)
        def map_uc():
            if self.owns_backing:
                self._call("cuMemMap", self._owned_ptr, self.size, 0, self._owned_handle, 0)
                self._uc_mapped = True
        self._phase("owned UC mapping", map_uc)
        def access_uc():
            if self.owns_backing:
                self._call("cuMemSetAccess", self._owned_ptr, self.size, [_read_write_access(self.device)], 1)
                self.local_ptr = self._owned_ptr
                self.backing_owner.local_ptr = self.local_ptr
        self._phase("owned UC access", access_uc)
        def create():
            if self.rank == self.creator_rank:
                prop = cuda.CUmulticastObjectProp()
                prop.numDevices, prop.size, prop.handleTypes, prop.flags = len(self.members), self.size, int(FABRIC), 0
                self.handle = self._call("cuMulticastCreate", prop)
        self._phase("creator MC creation", create)
        def export():
            if self.rank == self.creator_rank:
                data = bytes(self._call("cuMemExportToShareableHandle", self.handle, FABRIC, 0).data)
                if len(data) != 64:
                    raise ValueError("FABRIC handle must contain exactly 64 bytes")
                return data
        exports = self.all_gather(self._phase("creator MC export", export))
        def import_member():
            data = exports[self.creator_rank]
            if type(data) is not bytes or len(data) != 64:
                raise ValueError("invalid collective FABRIC handle")
            if self.is_member and self.rank != self.creator_rank:
                self.handle = self._call("cuMemImportFromShareableHandle", bytearray(data), FABRIC)
        self._phase("member MC import", import_member)
        self._phase("all AddDevice before bind/map", lambda:
            self._call("cuMulticastAddDevice", self.handle, self.device) if self.is_member else None)
        def bind():
            if self.is_member:
                self._call("cuMulticastBindMem_v2", self.handle, self.device, 0, self.backing_owner.handle, 0, self.size, 0)
                self.bound = True
        self._phase("member backing bind", bind)
        def reserve_mc():
            if self.is_member:
                self.mc_ptr = int(self._call("cuMemAddressReserve", self.size, self.granularity, 0, 0))
        self._phase("member MC reservation", reserve_mc)
        def map_mc():
            if self.is_member:
                self._call("cuMemMap", self.mc_ptr, self.size, 0, self.handle, 0)
                self.mc_mapped = True
        self._phase("member MC mapping", map_mc)
        self._phase("member MC access", lambda:
            self._call("cuMemSetAccess", self.mc_ptr, self.size, [_read_write_access(self.device)], 1) if self.is_member else None)
        return self

    @classmethod
    def create(cls, *, world, rank, device, members, creator_rank, size,
               all_gather, backing=None) -> "MulticastTeam":
        if not callable(all_gather):
            raise TypeError("all_gather must be callable")
        team = cls(world=world, rank=rank, device=device, members=members,
                   creator_rank=creator_rank, size=size, all_gather=all_gather, backing=backing)
        try:
            return team._initialize()
        except Exception as exc:
            try:
                team.close_collectively()
            except Exception:
                team._retain()
            if isinstance(exc, MulticastTeamError):
                raise
            raise MulticastTeamError(str(exc), team, [{"message": str(exc)}]) from exc

    @classmethod
    def create_dual(cls, *, world, rank, device, members, size, all_gather,
                    backing=None, _validate=None) -> "MulticastTeamPair":
        """Create first/second-member aliases, with at most one owned backing.

        Explicit ``create(..., creator_rank=...)`` semantics are unchanged.
        """
        return MulticastTeamPair.create(world=world, rank=rank, device=device,
            members=members, size=size, all_gather=all_gather, backing=backing,
            team_factory=cls, validate=_validate)

    def _retain(self):
        if not self.retained_after_cleanup_failure:
            _RETAINED_MULTICAST_TEAMS.append(self)
            self.retained_after_cleanup_failure = True

    def close_collectively(self) -> bool:
        """Drain and free in dependency order; stop collectively on any error."""
        if self.closed:
            return True
        def unmap_mc():
            if self.mc_mapped:
                self._call("cuMemUnmap", self.mc_ptr, self.size)
                self.mc_mapped = False
            if self.mc_ptr:
                self._call("cuMemAddressFree", self.mc_ptr, self.size)
                self.mc_ptr = 0
        def unbind():
            if self.bound:
                self._call("cuMulticastUnbind", self.handle, self.device, 0, self.size)
                self.bound = False
        def release_imports():
            if self.rank != self.creator_rank and self.handle is not None:
                self._call("cuMemRelease", self.handle)
                self.handle = None
        def release_creator():
            if self.rank == self.creator_rank and self.handle is not None:
                self._call("cuMemRelease", self.handle)
                self.handle = None
        def unmap_uc():
            if self._uc_mapped:
                self._call("cuMemUnmap", self._owned_ptr, self.size)
                self._uc_mapped = False
            if self._owned_ptr:
                self._call("cuMemAddressFree", self._owned_ptr, self.size)
                self._owned_ptr = self.local_ptr = 0
        def release_backing():
            if self._owned_handle is not None:
                self._call("cuMemRelease", self._owned_handle)
                self._owned_handle = None
            self.backing_owner = None
        try:
            for label, fn in (("drain context", lambda: self._call("cuCtxSynchronize")),
                    ("unmap MC", unmap_mc), ("unbind MC", unbind),
                    ("release imports", release_imports), ("release creator", release_creator),
                    ("unmap owned UC", unmap_uc), ("release owned backing", release_backing)):
                if not self._phase("cleanup " + label, fn, cleanup=True):
                    self._retain()
                    return False
        except Exception:
            self._retain()
            raise
        self.closed = True
        return True

    def metadata(self) -> dict:
        return {"rank": self.rank, "device": self.device, "members": list(self.members),
                "creator_rank": self.creator_rank, "size": self.size, "is_member": self.is_member,
                "owns_backing": self.owns_backing, "borrowed_backing": self._borrowed,
                "mc_ptr": self.mc_ptr, "local_ptr": self.local_ptr, "bound": self.bound,
                "granularities": self.granularities, "closed": self.closed,
                "retained_after_cleanup_failure": self.retained_after_cleanup_failure}


class MulticastTeamPair:
    """Two MC objects over one backing, exposing the rank-selected aperture.

    The first alias owns any dummy; the second borrows that exact allocation.
    Every EP rank retains this object, including nonmembers, and calls close in
    the same global order. No destructor starts a collective.
    """

    _SELECTED_FIELDS = frozenset(("handle", "mc_ptr", "creator_rank", "bound",
        "mc_mapped", "granularity", "granularities"))
    _PRIMARY_FIELDS = frozenset(("backing_owner", "owns_backing", "local_ptr",
        "_owned_handle", "_owned_ptr", "_uc_mapped", "_borrowed", "is_member"))

    def __init__(self, *, world, rank, device, members, size, all_gather):
        self.world, self.rank, self.device = world, rank, device
        self.members, self.size, self.all_gather = members, size, all_gather
        self._aliases = []
        self.closed = self.retained_after_cleanup_failure = False

    @property
    def aliases(self) -> tuple[MulticastTeam, ...]:
        return tuple(self._aliases)

    @property
    def selected_alias(self) -> MulticastTeam:
        if not self._aliases:
            raise RuntimeError("multicast pair has no initialized alias")
        index = 1 if len(self._aliases) > 1 and self.rank == self.members[0] else 0
        return self._aliases[index]

    def __getattr__(self, name):
        if name in self._SELECTED_FIELDS:
            return getattr(self.selected_alias, name)
        if name in self._PRIMARY_FIELDS and self._aliases:
            return getattr(self._aliases[0], name)
        raise AttributeError(name)

    def _phase(self, label, operation, *, cleanup=False):
        return MulticastTeam._phase(self, label, operation, cleanup=cleanup)

    def _retain(self):
        MulticastTeam._retain(self)

    @classmethod
    def create(cls, *, world, rank, device, members, size, all_gather,
               backing=None, team_factory=MulticastTeam, validate=None):
        if not callable(all_gather):
            raise TypeError("all_gather must be callable")
        pair = cls(world=world, rank=rank, device=device, members=members,
                   size=size, all_gather=all_gather)
        try:
            creator = members[0] if type(members) is tuple and members else None
            probe = team_factory(world=world, rank=rank, device=device,
                members=members, creator_rank=creator, size=size,
                all_gather=all_gather, backing=backing)
            def validate_arguments():
                probe._validate()
                if validate is not None:
                    validate()
            pair._phase("validate dual multicast arguments", validate_arguments)
            primary = team_factory.create(world=world, rank=rank, device=device,
                members=members, creator_rank=members[0], size=size,
                all_gather=all_gather, backing=backing)
            pair._aliases.append(primary)
            if len(members) > 1:
                secondary = team_factory.create(world=world, rank=rank, device=device,
                    members=members, creator_rank=members[1], size=size,
                    all_gather=all_gather,
                    backing=primary.backing_owner if primary.is_member else None)
                pair._aliases.append(secondary)
            return pair
        except Exception as exc:
            if isinstance(exc, MulticastTeamError) and exc.team is not pair:
                if all(alias is not exc.team for alias in pair._aliases):
                    pair._aliases.append(exc.team)
            # A failed secondary cleanup may still reference primary's dummy.
            # Never free the primary allocation while any such alias survives.
            if any(alias.retained_after_cleanup_failure for alias in pair._aliases):
                pair._retain()
            else:
                try:
                    if not pair.close_collectively():
                        pair._retain()
                except Exception:
                    pair._retain()
            failures = exc.failures if isinstance(exc, MulticastTeamError) else [{"message": str(exc)}]
            raise MulticastTeamError(str(exc), pair, failures) from exc

    def close_collectively(self) -> bool:
        """Free the borrowed secondary first; the primary owns the only dummy."""
        if self.closed:
            return True
        if self.retained_after_cleanup_failure:
            return False
        try:
            for alias in reversed(self._aliases):
                if alias.retained_after_cleanup_failure or not alias.close_collectively():
                    self._retain()
                    return False
        except Exception:
            self._retain()
            raise
        self.closed = True
        return True

    def metadata(self) -> dict:
        selected = self.selected_alias.metadata() if self._aliases else {}
        selected.update({"rank": self.rank, "device": self.device,
            "members": list(self.members) if type(self.members) is tuple else [],
            "size": self.size, "dual_creator": len(self.members) > 1 if type(self.members) is tuple else False,
            "selection_policy": "first member uses second creator; all others use first creator",
            "alias_count": len(self._aliases),
            "aliases": [alias.metadata() for alias in self._aliases],
            "closed": self.closed,
            "retained_after_cleanup_failure": self.retained_after_cleanup_failure,
            "owns_backing": self._aliases[0].owns_backing if self._aliases else False,
            "allocated_bytes": sum(alias.size for alias in self._aliases if alias._owned_handle is not None),
            "mapping_count": sum(bool(alias.mc_ptr) for alias in self._aliases),
            "reserved_va_bytes": sum((alias.size if alias.mc_ptr else 0) +
                                     (alias.size if alias._owned_ptr else 0) for alias in self._aliases)})
        return selected


__all__.extend(["MulticastTeam", "MulticastTeamPair", "MulticastTeamError"])
