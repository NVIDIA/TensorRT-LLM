# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UBX (Caliper) all-to-all for Ulysses sequence parallelism.

``UBXAllToAll`` wraps Caliper's ``SymmAllocator`` to provide a zero-copy,
CUDA-graph-safe all-to-all that replaces the NCCL-backed ``all_to_all_4d`` /
``all_to_all_5d`` calls in ``UlyssesAttention``.

UBX Lamport wins by 1.3–1.5x over NCCL in CUDA graph mode (≥64KB payloads).
Falls back silently to NCCL when Caliper is not available or fails.
"""

import logging
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _ubx_available() -> bool:
    """Return True if Caliper UBX is importable on this node."""
    try:
        import ubx.allocator  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


class UBXAllToAll:
    """UBX-backed all-to-all for Ulysses using Caliper's SymmAllocator.

    UBX Lamport wins by 1.3–1.5x over NCCL in CUDA graph mode (≥64KB
    payloads).  Falls back silently to ``all_to_all_4d`` / ``all_to_all_5d``
    on any error.

    CUDA graph safe: ``pool_in`` tensors are cached persistently per
    ``(shape, dtype)`` key so no Python-level alloc/free occurs inside the
    captured region.  ``pool_out`` is a Lamport rolling buffer owned by
    ``alltoall_auto`` — never freed.

    Supports 4D tensors ``[B, S, H, D]`` and 5D tensors ``[B, S, Q, H, D]``
    (stacked QKV).  ``scatter_dim`` and ``gather_dim`` follow the same
    semantics as ``all_to_all_4d`` / ``all_to_all_5d``.
    """

    _POOL_MB = 2048  # symmetric pool per rank (MB)

    def __init__(self, process_group: dist.ProcessGroup):
        self._pg = process_group
        self._allocator = None
        self._init_err: Optional[Exception] = None
        # (flat_shape, dtype) → persistent SymmTensor; never freed so CUDA
        # graphs can capture copy_ without the address being reallocated.
        self._pool_cache: dict = {}

    def _try_init(self, device: torch.device) -> bool:
        if self._init_err is not None:
            return False
        if self._allocator is not None:
            return True
        try:
            from ubx.allocator import SymmAllocator

            pool_bytes = self._POOL_MB * 1024 * 1024
            self._allocator = SymmAllocator(pool_bytes, device, self._pg)
            logger.info("UBXAllToAll: SymmAllocator ready")
            return True
        except Exception as exc:
            self._init_err = exc
            logger.warning(f"UBXAllToAll: init failed, falling back to NCCL: {exc}")
            return False

    def _nccl_fallback(self, tensor, scatter_dim, gather_dim, pg):
        from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

        if tensor.ndim == 5:
            return all_to_all_5d(
                tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, process_group=pg
            )
        return all_to_all_4d(
            tensor, scatter_dim=scatter_dim, gather_dim=gather_dim, process_group=pg
        )

    def __call__(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        pg = process_group or self._pg
        world_size = dist.get_world_size(pg)
        if world_size == 1:
            return tensor
        if not self._try_init(tensor.device):
            return self._nccl_fallback(tensor, scatter_dim, gather_dim, pg)
        try:
            return self._ubx_all_to_all(tensor, scatter_dim, gather_dim, world_size)
        except Exception as exc:
            logger.warning(f"UBXAllToAll: kernel error, falling back to NCCL: {exc}")
            return self._nccl_fallback(tensor, scatter_dim, gather_dim, pg)

    def _ubx_all_to_all(
        self,
        tensor: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        world_size: int,
    ) -> torch.Tensor:
        alloc = self._allocator
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

        # pool_in cached permanently — no alloc/free in hot path (CUDA graph safe)
        key = (flat.shape, flat.dtype)
        pool_in = self._pool_cache.get(key)
        if pool_in is None:
            pool_in = alloc.create_tensor(flat.shape, flat.dtype)
            if pool_in is None:
                raise RuntimeError("UBX pool exhausted")
            self._pool_cache[key] = pool_in
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
