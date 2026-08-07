# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UBX (Caliper) all-to-all for Ulysses sequence parallelism.

``UBXAllToAll`` wraps Caliper's ``SymmAllocator`` to provide a zero-copy,
CUDA-graph-safe all-to-all that replaces the NCCL-backed ``all_to_all_4d`` /
``all_to_all_5d`` calls in ``UlyssesAttention``.

UBX Lamport wins by 1.3–1.5x over NCCL in CUDA graph mode (>=64KB payloads).
Falls back to NCCL before UBX starts when Caliper is unavailable or symmetric
pool setup fails on any rank. In-flight UBX collective failures propagate
rather than locally switching one rank to NCCL.
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Protocol

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_PoolCacheKey = tuple[torch.Size, torch.dtype]
_SharedPoolKey = tuple[int, int]


class _SymmAllocator(Protocol):
    """Minimal Caliper allocator protocol used by UBXAllToAll."""

    def create_tensor(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor | None:
        """Create a persistent symmetric tensor from the allocator pool."""
        ...

    def alltoall_auto(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run UBX all-to-all on a symmetric tensor."""
        ...


@dataclass
class _PoolState:
    """Shared UBX state for one CUDA device and process group."""

    allocator: _SymmAllocator | None = None
    init_err: BaseException | None = None
    pool_cache: dict[_PoolCacheKey, torch.Tensor] = field(default_factory=dict)
    ready_pool_keys: set[_PoolCacheKey] = field(default_factory=set)


def _sync_ready(local_ready: bool, device: torch.device, process_group: dist.ProcessGroup) -> bool:
    """Return True only when every rank reports ready."""
    ready = torch.tensor([int(local_ready)], device=device, dtype=torch.int32)
    dist.all_reduce(ready, op=dist.ReduceOp.MIN, group=process_group)
    return bool(ready.item())


def _ubx_available(
    process_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
) -> bool:
    """Return True if Caliper UBX is importable on every participating rank."""
    try:
        import ubx.allocator  # noqa: F401

        local_available = True
    except (ImportError, OSError, RuntimeError):
        local_available = False

    if process_group is None and dist.is_available() and dist.is_initialized():
        process_group = dist.group.WORLD

    if process_group is None or device is None or device.type != "cuda":
        return local_available
    if dist.get_world_size(group=process_group) == 1:
        return local_available
    return _sync_ready(local_available, device, process_group)


def _shared_pool_key(device: torch.device, process_group: dist.ProcessGroup) -> _SharedPoolKey:
    """Build the process-local shared-pool key."""
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return device_index, id(process_group)


def _create_allocator(
    pool_bytes: int,
    device: torch.device,
    process_group: dist.ProcessGroup,
) -> _SymmAllocator:
    """Create Caliper's allocator behind a typed protocol boundary."""
    from ubx.allocator import SymmAllocator

    return SymmAllocator(pool_bytes, device, process_group)


class UBXAllToAll:
    """UBX-backed all-to-all for Ulysses using Caliper's SymmAllocator.

    UBX Lamport wins by 1.3-1.5x over NCCL in CUDA graph mode (>=64KB
    payloads). It falls back to ``all_to_all_4d`` / ``all_to_all_5d`` only
    before entering UBX, after a process-group-wide readiness check.

    CUDA graph safe: ``pool_in`` tensors are cached persistently per
    ``(shape, dtype)`` key so no Python-level alloc/free occurs inside the
    captured region.  ``pool_out`` is a Lamport rolling buffer owned by
    ``alltoall_auto`` — never freed.

    Supports 4D tensors ``[B, S, H, D]`` and 5D tensors ``[B, S, Q, H, D]``
    (stacked QKV).  ``scatter_dim`` and ``gather_dim`` follow the same
    semantics as ``all_to_all_4d`` / ``all_to_all_5d``.
    """

    _POOL_MB = 2048  # symmetric pool per rank (MB)
    _shared_pools: ClassVar[dict[_SharedPoolKey, _PoolState]] = {}

    def __init__(self, process_group: dist.ProcessGroup) -> None:
        """Create a wrapper that lazily binds shared UBX state."""
        self._pg = process_group
        self._allocator: _SymmAllocator | None = None
        self._init_err: BaseException | None = None
        self._state: _PoolState | None = None
        # (flat_shape, dtype) -> persistent SymmTensor; never freed so CUDA
        # graphs can capture copy_ without the address being reallocated.
        self._pool_cache: dict[_PoolCacheKey, torch.Tensor] = {}

    def _bind_state(self, device: torch.device, process_group: dist.ProcessGroup) -> _PoolState:
        """Bind wrapper-local fields to the shared state for this device/group."""
        key = _shared_pool_key(device, process_group)
        state = self._shared_pools.get(key)
        if state is None:
            state = _PoolState()
            self._shared_pools[key] = state
        self._sync_from_state(state)
        return state

    def _sync_from_state(self, state: _PoolState) -> None:
        """Refresh wrapper-local aliases for the shared allocator and cache."""
        self._state = state
        self._allocator = state.allocator
        self._init_err = state.init_err
        self._pool_cache = state.pool_cache

    def _disable_state(self, state: _PoolState, init_err: BaseException) -> None:
        """Disable UBX and release references to symmetric pool objects."""
        state.init_err = init_err
        state.pool_cache.clear()
        state.ready_pool_keys.clear()
        state.allocator = None
        self._sync_from_state(state)

    def _try_init(self, device: torch.device, process_group: dist.ProcessGroup) -> bool:
        """Initialize the shared allocator only when every rank can do so."""
        if device.type != "cuda":
            return False

        state = self._bind_state(device, process_group)
        if state.init_err is not None:
            self._sync_from_state(state)
            return False
        if state.allocator is not None:
            return True

        local_err: BaseException | None = None
        try:
            pool_bytes = self._POOL_MB * 1024 * 1024
            state.allocator = _create_allocator(pool_bytes, device, process_group)
        except (ImportError, MemoryError, OSError, RuntimeError) as exc:
            local_err = exc

        if not _sync_ready(state.allocator is not None, device, process_group):
            self._disable_state(
                state,
                local_err
                or RuntimeError("UBXAllToAll disabled because a peer failed initialization"),
            )
            logger.warning(
                "UBXAllToAll: init failed on at least one rank, falling back to NCCL: %s",
                state.init_err,
            )
            return False

        self._sync_from_state(state)
        logger.info("UBXAllToAll: shared SymmAllocator ready")
        return True

    def _nccl_fallback(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """Run the existing NCCL-backed all-to-all implementation."""
        from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

        if tensor.ndim == 5:
            return all_to_all_5d(
                tensor,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
                process_group=process_group,
            )
        return all_to_all_4d(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            process_group=process_group,
        )

    def __call__(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        process_group: dist.ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Run UBX all-to-all when collectively ready, otherwise use NCCL."""
        pg = process_group or self._pg
        world_size = dist.get_world_size(pg)
        if world_size == 1:
            return tensor
        if not self._try_init(tensor.device, pg):
            return self._nccl_fallback(tensor, scatter_dim, gather_dim, pg)

        output = self._ubx_all_to_all(tensor, scatter_dim, gather_dim, pg, world_size)
        if output is None:
            return self._nccl_fallback(tensor, scatter_dim, gather_dim, pg)
        return output

    def _get_pool_in(
        self,
        flat: torch.Tensor,
        device: torch.device,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor | None:
        """Return a shared input pool tensor after collective readiness."""
        state = self._state
        alloc = self._allocator
        if state is None or alloc is None:
            raise RuntimeError("UBXAllToAll allocator is not initialized")

        key = (flat.shape, flat.dtype)
        pool_in = state.pool_cache.get(key)
        local_err: BaseException | None = None
        if pool_in is None:
            try:
                pool_in = alloc.create_tensor(flat.shape, flat.dtype)
            except (MemoryError, OSError, RuntimeError) as exc:
                local_err = exc
            if pool_in is not None:
                state.pool_cache[key] = pool_in

        if key not in state.ready_pool_keys:
            if not _sync_ready(pool_in is not None, device, process_group):
                self._disable_state(
                    state,
                    local_err
                    or RuntimeError("UBXAllToAll disabled because a peer failed pool allocation"),
                )
                logger.warning(
                    "UBXAllToAll: pool allocation failed on at least one rank, "
                    "falling back to NCCL: %s",
                    state.init_err,
                )
                return None
            state.ready_pool_keys.add(key)

        self._sync_from_state(state)
        if pool_in is None:
            raise RuntimeError("UBXAllToAll pool key is marked ready without a local tensor")
        return pool_in

    def _ubx_all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        process_group: dist.ProcessGroup,
        world_size: int,
    ) -> torch.Tensor | None:
        """Transform tensor layout, call UBX, and restore the requested layout."""
        alloc = self._allocator
        if alloc is None:
            raise RuntimeError("UBXAllToAll allocator is not initialized")
        t = tensor.contiguous()
        ndim = t.ndim
        if ndim == 4:
            batch, seq, heads, head_dim = t.shape
            if scatter_dim == 2:
                # [B, S/P, H, D] → scatter heads → [P, B, S/P, H/P, D]
                inp = t.view(batch, seq, world_size, heads // world_size, head_dim)
                inp = inp.permute(2, 0, 1, 3, 4).contiguous()
            else:
                # [B, S, H/P, D] → scatter seq → [P, B, S/P, H/P, D]
                inp = t.view(batch, world_size, seq // world_size, heads, head_dim)
                inp = inp.permute(1, 0, 2, 3, 4).contiguous()
        elif ndim == 5:
            batch, seq, qkv, heads, head_dim = t.shape
            if scatter_dim == 3:
                # [B, S/P, 3, H, D] → scatter heads → [P, B, S/P, 3, H/P, D]
                inp = t.view(batch, seq, qkv, world_size, heads // world_size, head_dim)
                inp = inp.permute(3, 0, 1, 2, 4, 5).contiguous()
            else:
                # [B, S, 3, H/P, D] → scatter seq → [P, B, S/P, 3, H/P, D]
                inp = t.view(batch, world_size, seq // world_size, qkv, heads, head_dim)
                inp = inp.permute(1, 0, 2, 3, 4, 5).contiguous()
        else:
            raise ValueError(f"UBXAllToAll: unsupported ndim={ndim}")

        flat = inp.flatten()

        # pool_in cached permanently; no alloc/free in hot path after warmup.
        pool_in = self._get_pool_in(flat, tensor.device, process_group)
        if pool_in is None:
            return None
        pool_in.copy_(flat)

        pool_out = alloc.alltoall_auto(pool_in)
        out_flat = pool_out.clone()  # copy to regular (non-symmetric) memory

        out_t = out_flat.view_as(inp)

        if ndim == 4:
            if gather_dim == 1:
                # [P, B, S/P, H/P, D] → [B, S, H/P, D]
                out = out_t.permute(1, 0, 2, 3, 4).contiguous()
                out = out.view(batch, seq * world_size, heads // world_size, head_dim)
            else:
                # [P, B, S/P, H/P, D] → [B, S/P, H, D]
                out = out_t.permute(1, 2, 0, 3, 4).contiguous()
                out = out.view(batch, seq // world_size, heads * world_size, head_dim)
        else:
            if gather_dim == 1:
                # [P, B, S/P, 3, H/P, D] → [B, S, 3, H/P, D]
                out = out_t.permute(1, 0, 2, 3, 4, 5).contiguous()
                out = out.view(batch, seq * world_size, qkv, heads // world_size, head_dim)
            else:
                # [P, B, S/P, 3, H/P, D] → [B, S/P, 3, H, D]
                out = out_t.permute(1, 2, 3, 0, 4, 5).contiguous()
                out = out.view(batch, seq // world_size, qkv, heads * world_size, head_dim)

        return out
