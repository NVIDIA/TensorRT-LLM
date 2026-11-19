# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from tensorrt_llm._torch.distributed import all_to_all_4d

from .._op import Arch, Cell, OpWrapper


def _fits_unscatter_kernel(heads: int, dim: int) -> bool:
    # One CUDA block copies a row using eight BF16 elements per thread.
    return dim % 8 == 0 and heads * (dim // 8) <= 1024


class UlyssesQKVAllToAll(OpWrapper):
    """Scatter heads and gather rows from NHD Q/K/V in group-rank order."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="packed BF16 head exchange",
            spec=dict(heads=32, dim=128, dtype=torch.bfloat16, fused=True),
        ),
        Cell(
            why="separate FP32 exchanges",
            spec=dict(heads=4, dim=16, dtype=torch.float32, fused=False),
        ),
        Cell(
            why="wide BF16 head shard",
            spec=dict(heads=130, dim=128, dtype=torch.bfloat16, fused=False),
        ),
    )

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, group: dist.ProcessGroup, hnd: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = dist.get_world_size(group)
        batch, rows, heads, dim = q.shape
        if (
            q.is_cuda
            and q.dtype == k.dtype == v.dtype == torch.bfloat16
            and q.shape == k.shape == v.shape
            and _fits_unscatter_kernel(heads // size, dim)
        ):
            qkv = torch.stack((q, k, v), dim=2)
            send = qkv.view(batch, rows, 3, size, heads // size, dim).permute(3, 0, 1, 2, 4, 5)
            send = send.contiguous()
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv.flatten(), send.flatten(), group=group)
            return torch.ops.trtllm.ulysses_packed_qkv_post_unscatter(recv, 0 if hnd else 1)

        def exchange(x: torch.Tensor) -> torch.Tensor:
            out = all_to_all_4d(x, scatter_dim=2, gather_dim=1, process_group=group)
            return out.transpose(1, 2) if hnd else out

        return exchange(q), exchange(k), exchange(v)

    def reference(
        self,
        per_rank_qkv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        rank: int,
        hnd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Take the destination rank's heads from each source rank, then concatenate rows."""

        def gather(index: int) -> torch.Tensor:
            shards = [qkv[index].chunk(len(per_rank_qkv), dim=2)[rank] for qkv in per_rank_qkv]
            out = torch.cat(shards, dim=1)
            return out.transpose(1, 2) if hnd else out

        return gather(0), gather(1), gather(2)


def _use_fused_unscatter(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return (
        q.is_cuda
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and all(t.is_contiguous() for t in (q, k, v))
        and _fits_unscatter_kernel(max(q.size(3), k.size(3), v.size(3)), q.size(4))
    )


class UlyssesUnscatterQKV(OpWrapper):
    """Convert [rank, batch, rows, heads, dim] receives to NHD or HND.

    Q/K/V share rank count, batch and head dimension. Set `aligned_receives`
    when the allocator guarantees 16-byte-aligned Q/K/V starting addresses.
    """

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="BF16 vectorized stores",
            spec=dict(heads=16, dim=128, dtype=torch.bfloat16, fused=True),
        ),
        Cell(
            why="FP32 receive buffers", spec=dict(heads=2, dim=16, dtype=torch.float32, fused=False)
        ),
        Cell(
            why="unaligned head dimension",
            spec=dict(heads=2, dim=12, dtype=torch.bfloat16, fused=False),
        ),
        Cell(
            why="strided receive views",
            spec=dict(heads=2, dim=16, dtype=torch.bfloat16, stride=2, fused=False),
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hnd: bool,
        *,
        aligned_receives: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if aligned_receives and _use_fused_unscatter(q, k, v):
            return torch.ops.trtllm.ulysses_post_unscatter_qkv(q, k, v, 0 if hnd else 1)

        def unscatter(x: torch.Tensor) -> torch.Tensor:
            out = x.permute(1, 0, 2, 3, 4).contiguous().flatten(1, 2)
            return out.transpose(1, 2).contiguous() if hnd else out

        return unscatter(q), unscatter(k), unscatter(v)

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hnd: bool,
        *,
        aligned_receives: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        def gather_rows(x: torch.Tensor) -> torch.Tensor:
            out = torch.cat(x.unbind(0), dim=1)
            return out.transpose(1, 2) if hnd else out

        return gather_rows(q), gather_rows(k), gather_rows(v)

    def is_valid(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        hnd: bool,
        *,
        aligned_receives: bool = False,
    ) -> None:
        assert q.device == k.device == v.device
        if aligned_receives and _use_fused_unscatter(q, k, v):
            assert all(t.data_ptr() % 16 == 0 for t in (q, k, v)), (
                "BF16 receives require 16-byte alignment"
            )


ulysses_qkv_all_to_all = UlyssesQKVAllToAll()
ulysses_unscatter_qkv = UlyssesUnscatterQKV()
