# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MNNVL all-reduce RMSNorm fusion's numerics, for the fused tail's collectives.

The unfused path reduces the routed latent through the engine ``AllReduce`` module, which on a
group that spans GB300 trays runs the MNNVL all-reduce with the RMSNorm fused in
(``cpp/tensorrt_llm/kernels/communicationKernels/mnnvlAllreduceKernels.cu``). Its result depends on
three orders, all fixed by the row count:

* the kernel: one-shot while ``rows * dim * ranks * 2`` bytes fit the one-shot threshold
  (``TLLM_MNNVL_ONESHOT_THRESHOLD_BYTES``, 1 MiB by default), two-shot above it;
* the rank sum, in FP32: the one-shot kernel adds the ranks in order up to 8 ranks and in chunks of
  8 above (each chunk summed from zero, the chunks then added in order); the two-shot kernel uses
  chunks of 16;
* the sum of squares: ``adjustGridConfig`` lays a row over a cluster of CTAs with 8 elements per
  thread. Each thread adds its 8 BF16 squares in order, each warp reduces by an XOR butterfly
  (lanes past a partial last warp count as zero), each CTA reduces its warps' sums by the same
  butterfly over zero-padded lanes, and the CTAs' sums are added in cluster order.

The normalized element is ``bf16((x * rstd) * w)`` in the one-shot kernel and ``bf16((w * x) *
rstd)`` in the two-shot kernel, with ``rstd = rsqrt(sum / dim + eps)``.

:class:`MnnvlNormPlan` resolves these for a group and a row range on the host;
:func:`square_sum_slot` and :func:`row_square_sum` reproduce the sum of squares on the device
from per-thread sums staged in shared memory at the MNNVL kernel's (CTA, warp, lane) positions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from .moe_tail_primitives import packed_u32x4_to_bf16x8

VEC = 8
_WARP = 32


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def _grid_config(
    num_tokens: int, dim: int, sm_count: int, use_cluster: bool
) -> tuple[int, int, int]:
    """``adjustGridConfig<UseCluster>`` for 8-element accesses: (block size, cluster size, loads)."""
    cluster = 8 if use_cluster else 1
    threads_needed = _div_up(dim, VEC)
    loads = 1
    block = _div_up(threads_needed, cluster)
    if use_cluster:
        while threads_needed % cluster != 0 and cluster > 1:
            cluster //= 2
        max_divisible = cluster
        block = _div_up(threads_needed, cluster)
        while block < 128 and cluster >= 2:
            block *= 2
            cluster //= 2
        while num_tokens * cluster > sm_count and cluster > 1 and block <= 512:
            block *= 2
            cluster //= 2
        while cluster < max_divisible:
            candidate_cluster = cluster * 2
            candidate_block = _div_up(threads_needed, candidate_cluster)
            if candidate_block < 64 or num_tokens * candidate_cluster > sm_count:
                break
            cluster, block = candidate_cluster, candidate_block
    while block > 1024:
        if use_cluster and cluster < 8:
            cluster <<= 1
        elif loads < 8:
            loads += 1
        else:
            break
        block = _div_up(threads_needed, cluster * loads)
    return block, cluster, loads


def norm_geometry(num_tokens: int, dim: int, sm_count: int) -> tuple[int, int]:
    """(threads per CTA, CTAs per row) of the MNNVL RMSNorm for ``num_tokens`` rows.

    The one-shot kernel launches ``adjustGridConfig<true>``; the two-shot RMSNorm kernel launches
    the same configuration when it is a multi-CTA cluster with one load per thread, else
    ``adjustGridConfig<false>``. For a row of at most 8192 elements both resolve to one load per
    thread, so the two kernels share the layout.
    """
    block, cluster, loads = _grid_config(num_tokens, dim, sm_count, True)
    if cluster == 1 or loads != 1:
        block, cluster, loads = _grid_config(num_tokens, dim, sm_count, False)
    if loads != 1 or block * cluster * VEC != dim:
        raise ValueError(f"MNNVL RMSNorm layout for dim {dim} is not one 8-element load per thread")
    return block, cluster


@dataclass(frozen=True)
class MnnvlNormPlan:
    """How the MNNVL all-reduce RMSNorm fusion reduces and normalizes ``rows`` of one group."""

    dim: int
    oneshot_max_rows: int
    oneshot_rank_chunk: int
    twoshot_rank_chunk: int
    # (largest row count, threads per CTA, CTAs per row), ascending, covering 1..max_rows.
    geometries: tuple[tuple[int, int, int], ...]

    @property
    def slots(self) -> int:
        """Shared-memory slots the staged thread sums take under the widest geometry."""
        return max(cluster * _div_up(block, _WARP) * _WARP for _, block, cluster in self.geometries)


def mnnvl_one_shot_threshold_bytes() -> int:
    """The MNNVL all-reduce's one-shot threshold, as the engine resolved it at import."""
    from tensorrt_llm._torch.distributed.ops import _MNNVL_ONE_SHOT_THRESHOLD_BYTES

    return _MNNVL_ONE_SHOT_THRESHOLD_BYTES


def mnnvl_norm_plan(
    *, tp_size: int, dim: int, max_rows: int, device: torch.device | None = None
) -> MnnvlNormPlan:
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    geometries: list[tuple[int, int, int]] = []
    for rows in range(1, max_rows + 1):
        block, cluster = norm_geometry(rows, dim, sm_count)
        if geometries and geometries[-1][1:] == (block, cluster):
            geometries[-1] = (rows, block, cluster)
        else:
            geometries.append((rows, block, cluster))
    row_bytes = dim * tp_size * 2
    return MnnvlNormPlan(
        dim=dim,
        oneshot_max_rows=mnnvl_one_shot_threshold_bytes() // row_bytes,
        oneshot_rank_chunk=tp_size if tp_size <= 8 else 8,
        twoshot_rank_chunk=tp_size if tp_size < 16 else 16,
        geometries=tuple(geometries),
    )


def square_sum_slot(packed_idx, block: int, cluster: int):
    """Shared-memory slot of the thread sum for 8-element vector ``packed_idx`` of a row: the
    MNNVL kernel's (CTA, warp, lane) of that vector, each warp padded to 32 lanes."""
    del cluster
    warps = _div_up(block, _WARP)
    cta = packed_idx // block
    in_cta = packed_idx - cta * block
    warp = in_cta // _WARP
    return (cta * warps + warp) * _WARP + (in_cta - warp * _WARP)


def rank_sum(accum: cute.Tensor, rank_words: cute.Tensor, tp_size: int, chunk: int) -> None:
    """``accum[e]`` = the FP32 sum of the ranks' BF16 vectors (``rank_words[rank, :]``, 8
    packed elements each) in the MNNVL kernel's order: chunks of ``chunk`` ranks, each summed
    from zero in rank order, then the chunk sums added in order.

    Called while tracing a kernel."""
    values = [
        packed_u32x4_to_bf16x8(rank_words[rank, None].load()).to(Float32) for rank in range(tp_size)
    ]
    for element in range(VEC):
        total = None
        for base in range(0, tp_size, chunk):
            part = Float32(0.0)
            for rank in range(base, base + chunk):
                part = part + values[rank][element]
            total = part if total is None else total + part
        accum[element] = total


def _butterfly_padded(values: list):
    """Lane 0 of a 32-lane XOR butterfly whose lanes past ``len(values)`` hold zero."""
    lanes: list = list(values) + [None] * (_WARP - len(values))
    half = _WARP // 2
    while half:
        folded = []
        for i in range(half):
            a, b = lanes[i], lanes[i + half]
            folded.append(a if b is None else (b if a is None else a + b))
        lanes = folded
        half //= 2
    return lanes[0]


def row_square_sum(sums: cute.Tensor, offset, lane: Int32, block: int, cluster: int) -> Float32:
    """The MNNVL kernel's sum of a row's squares, from the thread sums staged at
    :func:`square_sum_slot` (padding lanes zero) from ``offset`` in ``sums``.

    Called while tracing a kernel by every lane of a warp; every lane returns the full sum.
    """
    warps = _div_up(block, _WARP)
    total = None
    for cta in range(cluster):
        warp_sums = []
        for warp in range(warps):
            value = sums[offset + (cta * warps + warp) * _WARP + lane]
            for step in (16, 8, 4, 2, 1):
                value = value + cute.arch.shuffle_sync_bfly(
                    value, offset=step, mask=-1, mask_and_clamp=31
                )
            warp_sums.append(value)
        cta_sum = _butterfly_padded(warp_sums)
        # The cluster sum starts from zero; 0 + s == s for the non-negative first CTA sum.
        total = cta_sum if total is None else total + cta_sum
    return Float32(total)


__all__ = [
    "MnnvlNormPlan",
    "mnnvl_norm_plan",
    "mnnvl_one_shot_threshold_bytes",
    "norm_geometry",
    "rank_sum",
    "row_square_sum",
    "square_sum_slot",
]
