"""Fabric-backed one-process-per-rank launcher for the CUDA schedulers."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from cuda.bindings import driver as cuda

from .runtime import (
    CudaPhysicalSlotScheduler,
    CudaSchedulerConfig,
    symmetric_buffer_ints,
)


def _check(result: object) -> object:
    if isinstance(result, tuple):
        error, *values = result
    else:
        error, values = result, []
    if int(error) != int(cuda.CUresult.CUDA_SUCCESS):
        _, name = cuda.cuGetErrorName(error)
        _, description = cuda.cuGetErrorString(error)
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(description, bytes):
            description = description.decode()
        raise RuntimeError(f"CUDA driver call failed: {name}: {description}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class FabricSymmetricBuffer:
    """One CUDA-fabric allocation mapped read/write into every EP process."""

    def __init__(self, nbytes: int, device: int):
        self.device = device
        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.location.id = device
        properties.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        self._properties = properties
        self.granularity = int(
            _check(
                cuda.cuMemGetAllocationGranularity(
                    properties,
                    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                )
            )
        )
        self.size = (nbytes + self.granularity - 1) // self.granularity
        self.size *= self.granularity
        self._mappings: list[int] = []
        self._reservations: list[int] = []
        self._imported: list[object] = []
        self._closed = False
        self._handle = None
        self.ptr = 0
        self.shareable = b""
        try:
            self._handle = _check(cuda.cuMemCreate(self.size, properties, 0))
            self.ptr = self._map(self._handle)
            _check(cuda.cuMemsetD8(cuda.CUdeviceptr(self.ptr), 0, self.size))
            exported = _check(
                cuda.cuMemExportToShareableHandle(
                    self._handle,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                )
            )
            self.shareable = bytes(exported.data)
        except BaseException:
            self.close()
            raise

    def _map(self, handle: object) -> int:
        address = int(
            _check(cuda.cuMemAddressReserve(self.size, self.granularity, 0, 0))
        )
        self._reservations.append(address)
        _check(cuda.cuMemMap(address, self.size, 0, handle, 0))
        self._mappings.append(address)
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cuda.cuMemSetAccess(address, self.size, [access], 1))
        return int(address)

    def import_peer(self, shareable: bytes) -> int:
        handle = _check(
            cuda.cuMemImportFromShareableHandle(
                shareable,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            )
        )
        self._imported.append(handle)
        return self._map(handle)

    def close(self) -> None:
        """Release every mapping and fabric handle after device work is drained."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            while self._mappings:
                address = self._mappings[-1]
                _check(cuda.cuMemUnmap(address, self.size))
                self._mappings.pop()
            while self._reservations:
                address = self._reservations[-1]
                _check(cuda.cuMemAddressFree(address, self.size))
                self._reservations.pop()
            while self._imported:
                handle = self._imported[-1]
                _check(cuda.cuMemRelease(handle))
                self._imported.pop()
            if self._handle is not None:
                _check(cuda.cuMemRelease(self._handle))
                self._handle = None
            self.ptr = 0
            self.shareable = b""
            self._closed = True


class DistributedCudaScheduler:
    """One rank of a simultaneous EP-wide pure-CUDA scheduler group."""

    def __init__(
        self,
        config: CudaSchedulerConfig,
        *,
        rank: int,
        world: int,
        local_rank: int,
        warmup_iterations: int = 5,
    ) -> None:
        if config.ep_size != world:
            raise ValueError("config.ep_size must equal distributed world size")
        self.rank = rank
        self.world = world
        self.device = local_rank
        torch.cuda.set_device(local_rank)
        rank_config = CudaSchedulerConfig(
            **{**config.__dict__, "local_rank": rank}
        )
        self.cfg = rank_config
        self.scheduler = CudaPhysicalSlotScheduler(
            rank_config, device=f"cuda:{local_rank}"
        )
        self.planner = self.scheduler

        nbytes = 4 * symmetric_buffer_ints(
            world, rank_config.logical_expert_count
        )
        self.symmetric = FabricSymmetricBuffer(nbytes, local_rank)
        handles: list[bytes | None] = [None] * world
        dist.all_gather_object(handles, self.symmetric.shareable)
        bases = []
        for peer, handle in enumerate(handles):
            if handle is None:
                raise RuntimeError("fabric handle exchange returned an incomplete EP")
            bases.append(
                self.symmetric.ptr
                if peer == rank
                else self.symmetric.import_peer(handle)
            )
        self.scheduler.connect_peer_bases(bases)
        torch.cuda.synchronize(local_rank)
        dist.barrier()
        self.warmup(warmup_iterations)

    def cpu_barrier(self) -> None:
        torch.cuda.synchronize(self.device)
        dist.barrier()

    def warmup(self, iterations: int = 5) -> None:
        if iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        for _ in range(iterations):
            self.cpu_barrier()
            self.scheduler.launch()
        self.cpu_barrier()
        self.scheduler.status.zero_()
        torch.cuda.synchronize(self.device)

    def stage(self, local_routes: object, *, validate_values: bool = False) -> None:
        routes = torch.as_tensor(
            local_routes,
            dtype=torch.int32,
            device=f"cuda:{self.device}",
        ).reshape(self.cfg.max_tokens_per_rank, self.cfg.topk)
        self.scheduler.stage(routes, validate_values=validate_values)

    def check_status(self) -> None:
        self.scheduler.check_status()


def init_from_env() -> tuple[int, int, int]:
    """Bootstrap the Gloo control plane from torchrun or Slurm variables."""

    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", 0)))
    world = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", 1)))
    local = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", 0)))
    if "MASTER_ADDR" not in os.environ:
        nodes = os.environ.get("SLURM_NODELIST", "localhost")
        first = nodes.split(",")[0]
        if "[" in first:
            head, tail = first.split("[", 1)
            first = head + tail.split("-")[0].split(",")[0].rstrip("]")
        os.environ["MASTER_ADDR"] = first
    os.environ.setdefault("MASTER_PORT", "29578")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    return rank, world, local


__all__ = [
    "DistributedCudaScheduler",
    "FabricSymmetricBuffer",
    "init_from_env",
]
