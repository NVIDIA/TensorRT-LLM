# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from FlashInfer (https://github.com/flashinfer-ai/flashinfer) under the Apache-2.0
# license.  Original file: flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
# Original copyright: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
# The high-throughput GEMM dispatchers use this module's target-specific
# launch policies and fused row-RMS epilogues.
"""SM10x (Blackwell) low-M BF16/FP16 GEMM with an in-kernel cluster split-K reduction.

Each cluster rank accumulates an exact K slice in FP32. Peers publish partials
to rank 0 through DSMEM; rank 0 reduces, casts, and stores once. The public
``A[M, K] @ B[K, N]`` problem is swapped internally, so tile dimensions below
use kernel coordinates: kernel-M carries public N and kernel-N carries public M.
"""

from __future__ import annotations

import dataclasses
import functools
import struct

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")


_CTA_K = 128


_SUPPORTED_MMA_M = (64, 128)


_SUPPORTED_MMA_N = (8, 16, 32)


_SUPPORTED_SPLIT_K = (1, 2, 3, 4)


MAX_M = 32


WIDE_MMA_N = 64


WIDE_MAX_M = 64


_FP32_BYTES = 4


_MAILBOX_ALIGN_BYTES = 128


_MBARRIER_BYTES = 8


_TMEM_POINTER_BYTES = 4


_TMEM_COLS = 256


_AB_ELEMENT_BYTES = 2


_AB_BUFFER_ALIGN_BYTES = 1024


_MIN_AB_STAGES = 1


_DEFAULT_MIN_AB_STAGES = 2


_MAX_AB_STAGES = 12


@dataclasses.dataclass(frozen=True)
class SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    tactic: SplitKTactic,
    ab_stages: int,
    mailbox_in_stages: bool = False,
    b_stages: int = 0,
    row_rms_k: int = 0,
) -> int:
    """Mirror the device allocator's shared-memory layout.

    ``mailbox_in_stages``: the split-K partials land in the A stages, beside two barriers, instead
    of a mailbox of their own. ``b_stages``: the B tiles have a ring of their own with that many
    stages and barriers (0: B shares the ``ab_stages`` stages and barriers of A). ``row_rms_k``:
    the K of a ``RowRmsB`` launch, which adds a barrier per B stage, the norm weight and two FP32
    values per row.
    """
    cursor = _align_up(
        tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * ab_stages,
        _AB_BUFFER_ALIGN_BYTES,
    ) + tactic.mma_n * _CTA_K * _AB_ELEMENT_BYTES * (b_stages or ab_stages)

    cursor = _align_up(cursor, _MBARRIER_BYTES)
    cursor += 2 * ab_stages * _MBARRIER_BYTES
    cursor += 2 * b_stages * _MBARRIER_BYTES
    cursor += 3 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES)
    cursor += _TMEM_POINTER_BYTES

    if row_rms_k:
        cursor = _align_up(cursor, _MBARRIER_BYTES) + b_stages * _MBARRIER_BYTES
        cursor = _align_up(cursor, _ROW_RMS_ALIGN_BYTES) + row_rms_k * _AB_ELEMENT_BYTES
        cursor = _align_up(cursor, _ROW_RMS_ALIGN_BYTES) + 2 * tactic.mma_n * _FP32_BYTES

    if tactic.split_k == 1:
        return cursor

    if mailbox_in_stages:
        # The reduction barrier and the mailbox-free barrier.
        return _align_up(cursor, _MBARRIER_BYTES) + 2 * _MBARRIER_BYTES

    return (
        _align_up(
            _align_up(cursor, _MAILBOX_ALIGN_BYTES)
            + (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES,
            _MBARRIER_BYTES,
        )
        + _MBARRIER_BYTES
    )


def mailbox_fits_stages(tactic: SplitKTactic, ab_stages: int | None = None) -> bool:
    """Whether the split-K partials of ``tactic`` fit in its A stages.

    The partials are the peers' FP32 tiles; ``ab_stages`` defaults to the tactic's. Where they fit,
    they can land in the A stages once the owner's MMAs are done with them (``mailbox_in_stages``).
    """
    stages = tactic.ab_stages if ab_stages is None else ab_stages
    return (
        tactic.split_k > 1
        and (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES
        <= tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * stages
    )


_LAUNCH_SMEM_HEADROOM_BYTES = 2048


_PUBLISH_ALIGN_BYTES = 16


_PUBLISH_BARRIER_ID = 1


_PUBLISH_VEC_ELEMENTS = 8


def _publish_smem_bytes(tactic: SplitKTactic, ab_stages: int) -> int:
    """Shared memory of the Lamport-publish kernel: the base layout, then the FP32 output tile."""
    return (
        _align_up(_smem_bytes(tactic, ab_stages), _PUBLISH_ALIGN_BYTES)
        + tactic.mma_m * tactic.mma_n * _FP32_BYTES
    )


@dataclasses.dataclass(frozen=True)
class LamportPublish:
    """Publish the GEMM output into the current buffer of an MNNVL one-shot all-reduce workspace
    instead of writing C.

    The owner CTA of each output tile stores its BF16 values, -0.0 written as +0.0, through the
    workspace's multicast address at [token][rank][column] of the buffer the workspace flags mark
    current: the shard store of the one-shot kernel, which then only polls. Every field is a
    compile-time constant of the kernel.

    ``trigger_after_wait``: the PDL dependents (the all-reduce) are released by the epilogue warps
    right after the grid dependency wait instead of by the weight-load warp once its loads are
    issued, so that the all-reduce's reads before its own wait (the flags, the residual) follow
    the completion of this GEMM's primary.
    """

    world_size: int
    rank: int
    trigger_after_wait: bool = True


HNORM_B_HIDDEN = 6144


_HNORM_B_VEC_ELEMENTS = 8


_WARP_LANES = 32


@dataclasses.dataclass(frozen=True)
class HnormB:
    """Produce B in the kernel instead of loading it: B is FlashInfer's RMSNorm
    (``flashinfer.norm.rmsnorm``, its CuTe DSL kernel at H = 6144) of the one activation row
    passed as B, with the weight passed as the bias operand, rounded to BF16. Warp 1 computes the
    norm with that kernel's arithmetic and summation order and writes this CTA's K slice into the
    B stages, so the MMA reads what the TMA would have loaded from the norm's output and the GEMM's
    result is bit-identical to the GEMM of that output. Every field is a compile-time constant.
    """

    hidden: int
    eps: float


def hnorm_b_supported(tactic: SplitKTactic, hidden: int) -> bool:
    """Whether ``HnormB`` serves ``tactic`` for a row of ``hidden``: the reproduced width, 8-row B
    tiles (one K_SW128 swizzle atom row group per 64 columns), every K tile of a CTA in its own
    stage, and a CTA slice of whole 32-vector groups (one warp, one vector per lane per group)."""
    if hidden != HNORM_B_HIDDEN or tactic.mma_n != 8:
        return False
    if hidden % (_CTA_K * tactic.split_k):
        return False
    k_tiles = hidden // _CTA_K // tactic.split_k
    return (
        k_tiles <= tactic.ab_stages
        and (k_tiles * _CTA_K // _HNORM_B_VEC_ELEMENTS) % _WARP_LANES == 0
    )


_ROW_RMS_ALIGN_BYTES = 16


_ROW_RMS_BARRIER_ID = 2


_ROW_RMS_CHUNKS = 8


@dataclasses.dataclass(frozen=True)
class RowRmsB:
    """B is the raw input x of the RMSNorm in front of the GEMM; its weight is the bias operand.

    The RMSNorm is ``x * rsqrt(mean(x^2) + eps) * gamma``. The epilogue warps, idle during the
    mainloop, take every B tile between its TMA load and its MMA: they add each row's squares of x
    to the row's sum, replace x by ``x * gamma`` (one product in FP32, rounded to BF16) in shared
    memory, and release the tile to the MMA. After the mainloop they scale every output row by
    ``rsqrt(sum / k + eps)``. The GEMM then computes ``(x * gamma) @ W^T * rstd``: the normalized
    activation's GEMM with the activation rounded to BF16 before the row scale instead of after it.
    Every field is a compile-time constant.
    """

    k: int
    eps: float


def row_rms_b_supported(tactic: SplitKTactic, b_stages: int, k: int) -> bool:
    """Whether ``RowRmsB`` serves ``tactic``.

    It needs no K split, a B ring of its own, one 64-token B tile (two epilogue threads per token
    row, one per 64-column swizzle atom of a K tile) and a K of whole K tiles.
    """
    return (
        tactic.split_k == 1
        and b_stages > 0
        and tactic.mma_n == WIDE_MMA_N
        and k > 0
        and k % _CTA_K == 0
    )


def tactic_launch_smem_bytes(
    tactic: SplitKTactic, mailbox_in_stages: bool = False, b_stages: int = 0
) -> int:
    """Dynamic shared memory for a launch that reserves only what ``tactic`` uses.

    The default launch reserves the full capacity, which keeps one CTA per SM. This size
    lets other CTAs share the SM: two CTAs of the c1 q_b tactic (128, 8, 8, 2) fit on one.
    """
    return min(
        _align_up(_smem_bytes(tactic, tactic.ab_stages, mailbox_in_stages, b_stages), 1024)
        + _LAUNCH_SMEM_HEADROOM_BYTES,
        _SMEM_CAPACITY_BYTES,
    )


def _max_ab_stages_for(
    tactic: SplitKTactic,
    smem_capacity: int,
    mailbox_in_stages: bool = False,
) -> int:
    return next(
        (
            stages
            for stages in range(_MAX_AB_STAGES, -1, -1)
            if _smem_bytes(tactic, stages, mailbox_in_stages) <= smem_capacity
        ),
        0,
    )


def validate_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
    mailbox_in_stages: bool = False,
    b_stages: int = 0,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``.

    ``mailbox_in_stages``: with its split-K partials in its A stages. ``b_stages``: with a B ring of
    its own of that many stages (0: B shares the A stages).
    """
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    # The WIDE_MMA_N tile (wide_tactic) is outside the autotune catalog.
    if tactic.mma_n not in _SUPPORTED_MMA_N and tactic.mma_n != WIDE_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    # Explicit split8 is supported without expanding the existing autotune catalog.
    if tactic.split_k not in _SUPPORTED_SPLIT_K and tactic.split_k != 8:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], got {tactic.ab_stages}"
        )
    if b_stages and not _MIN_AB_STAGES <= b_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"b_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], got {b_stages}"
        )
    max_m = WIDE_MAX_M if tactic.mma_n == WIDE_MMA_N else MAX_M
    if not 1 <= m <= max_m:
        raise ValueError(f"this low-M policy requires 1 <= M <= {max_m}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K or (k // _CTA_K) % tactic.split_k:
        raise ValueError(
            f"K={k} with CTA_K={_CTA_K} does not divide evenly across split_k={tactic.split_k}"
        )
    if mailbox_in_stages and not mailbox_fits_stages(tactic):
        raise ValueError(f"the split-K partials of {tactic} do not fit in its A stages")
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages, mailbox_in_stages, b_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available; max ab_stages is "
            f"{_max_ab_stages_for(tactic, smem_capacity, mailbox_in_stages)}"
        )


def default_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the default occupancy-oriented tactic."""
    if n <= 512:
        mma_m = 64
        mma_n = 16 if n == 512 and m > 24 else 8
        requested_split = 4
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        if n <= 3072:
            mma_m = 128 if m <= 16 else 64
            requested_split = 4 if m <= 16 else 2
        elif n < 8192:
            mma_m = 64
            requested_split = 2
        else:
            mma_m = 128 if k <= 1024 and m <= 24 else 64
            requested_split = 1

    if k <= 4 * _CTA_K:
        requested_split = 1
    split_k = next(
        split_k
        for split_k in reversed(_SUPPORTED_SPLIT_K)
        if split_k <= requested_split and (k // _CTA_K) % split_k == 0
    )
    tactic = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
    max_stages = _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=(
            _MIN_AB_STAGES
            if k == _CTA_K
            else _DEFAULT_MIN_AB_STAGES
            if k <= 2 * _CTA_K and m > 8
            else min(max_stages, 6)
            if k <= 4 * _CTA_K
            else max_stages
        ),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


def splitk_default_tactic(
    m: int, n: int, k: int, *, sm_count: int, smem_capacity: int
) -> SplitKTactic:
    """Refine the source default when split8 exposes otherwise idle SMs.

    This is an opt-in occupancy heuristic, not a measured winner for every shape.
    The launch reserves the full SM100 shared-memory capacity, even when the
    selected pipeline uses fewer bytes.
    """
    if sm_count <= 0 or smem_capacity < _SMEM_CAPACITY_BYTES:
        raise ValueError("device cannot serve the split-K launch resources")
    tactic = default_tactic(m, n, k)
    if tactic.split_k > sm_count:
        raise ValueError("device has too few SMs for the split-K cluster")
    output_tiles = ((n + tactic.mma_m - 1) // tactic.mma_m) * (
        (m + tactic.mma_n - 1) // tactic.mma_n
    )
    # Require exact K slices, at least two tiles per peer, and at most one
    # wave of eight-CTA clusters. Wide outputs retain their existing tactic.
    if (
        k % (8 * _CTA_K)
        or k // (8 * _CTA_K) < _DEFAULT_MIN_AB_STAGES
        or output_tiles * tactic.split_k >= sm_count
        or output_tiles * 8 > sm_count
    ):
        return tactic
    candidate = dataclasses.replace(tactic, split_k=8)
    stages = min(
        tactic.ab_stages,
        k // (8 * _CTA_K),
        _max_ab_stages_for(candidate, _SMEM_CAPACITY_BYTES),
    )
    if stages < _DEFAULT_MIN_AB_STAGES:
        return tactic
    candidate = dataclasses.replace(candidate, ab_stages=stages)
    validate_tactic(candidate, m, n, k)
    return candidate


_WIDE_SPLIT_K = (1, 2)


def wide_mailbox_in_stages(tactic: SplitKTactic) -> bool:
    """Whether a wide_tactic puts its split-K partials in its A stages.

    That is ``mailbox_in_stages`` of run_splitk_dense: wherever it has peers and they fit.
    """
    return mailbox_fits_stages(tactic)


def _wide_candidate(mma_m: int, split_k: int, k: int) -> SplitKTactic | None:
    """The deepest-pipeline tactic of 64-column tiles for these weight tiles and K split.

    The split-K partials go to the A stages where that buys a deeper pipeline. None where no
    two-stage pipeline fits.
    """
    k_tiles = k // _CTA_K
    if k_tiles % split_k or k_tiles // split_k < _DEFAULT_MIN_AB_STAGES:
        return None
    base = SplitKTactic(mma_m, WIDE_MMA_N, split_k, _MIN_AB_STAGES)
    own = min(_max_ab_stages_for(base, _SMEM_CAPACITY_BYTES), k_tiles // split_k)
    if split_k > 1:
        shared = min(
            _max_ab_stages_for(base, _SMEM_CAPACITY_BYTES, mailbox_in_stages=True),
            k_tiles // split_k,
        )
        if shared > own and mailbox_fits_stages(base, shared):
            return dataclasses.replace(base, ab_stages=shared)
    if own < _DEFAULT_MIN_AB_STAGES:
        return None
    return dataclasses.replace(base, ab_stages=own)


def wide_tactic(m: int, n: int, k: int, *, sm_count: int, smem_capacity: int) -> SplitKTactic:
    """The tactic for 33-64 rows.

    WIDE_MMA_N token columns (one kernel-N tile, so every CTA reads its weight tile once), 64- or
    128-row weight tiles and a K split of _WIDE_SPLIT_K (_wide_candidate): the most CTAs within one
    wave, ties to the 64-row tiles. Run it with ``mailbox_in_stages=wide_mailbox_in_stages(tactic)``.
    """
    if sm_count <= 0 or smem_capacity < _SMEM_CAPACITY_BYTES:
        raise ValueError("device cannot serve the split-K launch resources")
    if not MAX_M < m <= WIDE_MAX_M or n <= 0 or k <= 0 or k % _CTA_K:
        raise ValueError(f"wide_tactic serves {MAX_M} < M <= {WIDE_MAX_M}, got M={m} N={n} K={k}")
    best = None
    for mma_m in _SUPPORTED_MMA_M:
        for split_k in _WIDE_SPLIT_K:
            ctas = ((n + mma_m - 1) // mma_m) * split_k
            if split_k > sm_count or ctas > sm_count:
                continue
            tactic = _wide_candidate(mma_m, split_k, k)
            if tactic is not None and (best is None or ctas > best[0]):
                best = (ctas, tactic)
    if best is None:
        raise ValueError(f"no one-wave {WIDE_MMA_N}-column tactic for N={n} K={k}")
    validate_tactic(best[1], m, n, k, mailbox_in_stages=wide_mailbox_in_stages(best[1]))
    return best[1]


_WIDE_TUNED_B_STAGES = 6


@dataclasses.dataclass(frozen=True)
class WideLaunch:
    """A 33-64 row launch of run_splitk_dense: the tactic and the launch options it runs with."""

    tactic: SplitKTactic
    mailbox_in_stages: bool = False
    b_stages: int = 0
    mailbox_interleaved: bool = False
    launch_dependents_early: bool = True

    def run_kwargs(self) -> dict:
        """The run_splitk_dense keyword arguments of this launch, defaults left out."""
        kwargs = {}
        if self.mailbox_in_stages:
            kwargs["mailbox_in_stages"] = True
        if self.b_stages:
            kwargs["b_stages"] = self.b_stages
        if self.mailbox_interleaved:
            kwargs["mailbox_interleaved"] = True
        if not self.launch_dependents_early:
            kwargs["launch_dependents_early"] = False
        return kwargs


def wide_tuned_launch(m: int, n: int, k: int, *, sm_count: int, smem_capacity: int) -> WideLaunch:
    """The tuned launch for 33-64 rows.

    wide_tactic's tiles and K split. Where it splits K: the activation tiles in a ring of their own,
    _WIDE_TUNED_B_STAGES deep, beside the deepest weight ring that fits next to a mailbox of the
    owner's own (so the peers send their partials as soon as their MMAs are done), the mailbox laid
    out thread-interleaved, and the PDL dependents launched when the GEMM completes (they then find
    the SMs it held free). Same tiles, K slices and summation order as wide_tactic, so the same
    results. Without a K split: wide_tactic as it is.
    """
    tactic = wide_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
    if tactic.split_k == 1:
        return WideLaunch(tactic)
    k_tiles = k // _CTA_K // tactic.split_k
    b_stages = min(_WIDE_TUNED_B_STAGES, k_tiles)
    a_stages = next(
        (
            stages
            for stages in range(min(_MAX_AB_STAGES, k_tiles), b_stages - 1, -1)
            if _smem_bytes(dataclasses.replace(tactic, ab_stages=stages), stages, False, b_stages)
            <= smem_capacity
        ),
        0,
    )
    if not a_stages:
        return WideLaunch(tactic, mailbox_in_stages=wide_mailbox_in_stages(tactic))
    tuned = dataclasses.replace(tactic, ab_stages=a_stages)
    validate_tactic(tuned, m, n, k, smem_capacity=smem_capacity, b_stages=b_stages)
    return WideLaunch(
        tuned, b_stages=b_stages, mailbox_interleaved=True, launch_dependents_early=False
    )


_NARROW_MMA_M = 128


_NARROW_AB_STAGES = 3


def narrow_tactic(m: int, n: int, k: int, *, smem_capacity: int) -> SplitKTactic:
    """The tactic of a 33-64 row launch that holds few SMs.

    One 64-row kernel-N tile, 128-row weight tiles, no K split and _NARROW_AB_STAGES stages, so
    (n + 127) // 128 CTAs (48 at N = 6144). Run with ``reserve_full_smem=False``, a CTA takes only its own
    shared memory and finds SMs beside other kernels (the MoE shared expert's down projection beside the
    MegaMoE kernels).
    """
    if not MAX_M < m <= WIDE_MAX_M or n <= 0 or k <= 0 or k % _CTA_K:
        raise ValueError(f"narrow_tactic serves {MAX_M} < M <= {WIDE_MAX_M}, got M={m} N={n} K={k}")
    tactic = SplitKTactic(_NARROW_MMA_M, WIDE_MMA_N, 1, min(_NARROW_AB_STAGES, k // _CTA_K))
    validate_tactic(tactic, m, n, k, smem_capacity=smem_capacity)
    return tactic


_ROW_RMS_B_STAGES = 4


def row_rms_launch(
    m: int,
    n: int,
    k: int,
    *,
    sm_count: int,
    smem_capacity: int,
    launch_dependents_early: bool = True,
) -> WideLaunch:
    """The 33-64 row launch of run_splitk_row_rms.

    wide_tactic's tiles without a K split (q_b of the MLA: 128-row weight tiles, one CTA per tile),
    the activation tiles in a ring of their own (_ROW_RMS_B_STAGES deep) and the deepest weight
    ring that fits beside it and the RowRmsB shared memory. ``launch_dependents_early``: the PDL
    dependents are released once the weight loads are issued; without it when the GEMM completes.
    """
    tactic = wide_tactic(m, n, k, sm_count=sm_count, smem_capacity=smem_capacity)
    if tactic.split_k != 1:
        raise ValueError(f"row_rms_launch serves GEMMs without a K split, got {tactic}")
    k_tiles = k // _CTA_K
    b_stages = min(_ROW_RMS_B_STAGES, k_tiles)
    a_stages = next(
        (
            stages
            for stages in range(min(_MAX_AB_STAGES, k_tiles), 0, -1)
            if _smem_bytes(
                dataclasses.replace(tactic, ab_stages=stages), stages, False, b_stages, k
            )
            <= smem_capacity
        ),
        0,
    )
    if not a_stages:
        raise ValueError(f"no RowRmsB weight ring fits for N={n} K={k}")
    tuned = dataclasses.replace(tactic, ab_stages=a_stages)
    validate_tactic(tuned, m, n, k, smem_capacity=smem_capacity, b_stages=b_stages)
    return WideLaunch(tuned, b_stages=b_stages, launch_dependents_early=launch_dependents_early)


def one_wave_tactic(tactic: SplitKTactic, m: int, n: int, k: int, *, sm_count: int) -> SplitKTactic:
    """``tactic`` with 128-row instead of 64-row tiles when its CTAs (one per SM) need more than one wave and the
    128-row tiles fit in one, else ``tactic`` unchanged.

    Each output keeps its split-K slice and its K order, so the results are the same; the grid is half as large and
    has no partial second wave (the c1 LM head: 152 instead of 303 CTAs). Only long-K tactics (more than four K
    tiles, the full-depth pipelines) are refined.
    """

    def cta_count(t: SplitKTactic) -> int:
        return ((n + t.mma_m - 1) // t.mma_m) * ((m + t.mma_n - 1) // t.mma_n) * t.split_k

    if tactic.mma_m != 64 or k <= 4 * _CTA_K or cta_count(tactic) <= sm_count:
        return tactic
    candidate = dataclasses.replace(tactic, mma_m=128)
    if cta_count(candidate) > sm_count:
        return tactic
    stages = min(tactic.ab_stages, _max_ab_stages_for(candidate, _SMEM_CAPACITY_BYTES))
    if stages < _DEFAULT_MIN_AB_STAGES:
        return tactic
    candidate = dataclasses.replace(candidate, ab_stages=stages)
    try:
        validate_tactic(candidate, m, n, k)
    except ValueError:
        return tactic
    return candidate


@dsl_user_op
def _map_shared_rank(
    smem_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> "Int32":
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr: "cute.Pointer",
    mbar_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _arrive_shared_remote(
    mbar_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> None:
    """Arrive on the mbarrier at ``mbar_ptr``'s offset in a peer CTA (release, cluster scope)."""
    llvm.inline_asm(
        None,
        [_map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value()],
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _wait_parity_cluster(mbar_ptr: "cute.Pointer", phase: "Int32", *, loc=None, ip=None) -> None:
    """Wait for the phase of ``phase`` parity of this CTA's mbarrier at ``mbar_ptr``.

    Acquire at cluster scope: its arrival comes from a peer CTA.
    """
    llvm.inline_asm(
        None,
        [mbar_ptr.toint(loc=loc, ip=ip).ir_value(), phase.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred mbwp;\n\tMBW_RETRY:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 mbwp, [$0], $1;\n\t"
        "@!mbwp bra MBW_RETRY;\n\t}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _lamport_current_buffer(
    flags_address: "Int64",
    buffers_address: "Int64",
    *,
    loc=None,
    ip=None,
) -> "Int64":
    """The address of the workspace's current Lamport buffer: buffers + flags[0] (current index)
    * flags[2] (bytes per buffer), the flags read with one volatile 16-byte load (the previous
    all-reduce rotated them)."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [flags_address.ir_value(loc=loc, ip=ip), buffers_address.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lfcur, lfdirty, lfbytes, lfstages;\n\t.reg .b64 lfoffset;\n\t"
            "ld.volatile.global.v4.u32 {lfcur, lfdirty, lfbytes, lfstages}, [$1];\n\t"
            "mul.wide.u32 lfoffset, lfcur, lfbytes;\n\tadd.s64 $0, $2, lfoffset;\n\t}",
            "=l,l,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cvt_bf16x2_lamport(lo: "Float32", hi: "Float32", *, loc=None, ip=None) -> "Int32":
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half, and write a -0.0 half
    (0x8000, the idle value of a Lamport buffer) as +0.0, as the one-shot all-reduce does before it
    publishes its shard."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 lpword, lplo, lphi;\n\t.reg .pred lpzlo, lpzhi;\n\t"
            "cvt.rn.bf16x2.f32 lpword, $2, $1;\n\t"
            "and.b32 lplo, lpword, 0xFFFF;\n\tshr.u32 lphi, lpword, 16;\n\t"
            "setp.eq.u32 lpzlo, lplo, 0x8000;\n\tsetp.eq.u32 lpzhi, lphi, 0x8000;\n\t"
            "@lpzlo and.b32 lpword, lpword, 0xFFFF0000;\n\t"
            "@lpzhi and.b32 lpword, lpword, 0xFFFF;\n\t"
            "mov.b32 $0, lpword;\n\t}",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_global_v4(
    address: "Int64", v0: "Int32", v1: "Int32", v2: "Int32", v3: "Int32", *, loc=None, ip=None
) -> None:
    """One 16-byte global store (also through a multicast address)."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)] + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_global_cg_v4(address: "Int64", *, loc=None, ip=None):
    """One 16-byte load through L2 (``ld.global.cg``): four words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _st_shared_v4(
    address: "Int32", v0: "Int32", v1: "Int32", v2: "Int32", v3: "Int32", *, loc=None, ip=None
) -> None:
    """One 16-byte shared-memory store."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)] + [v.ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_v4_f32(smem_ptr: "cute.Pointer", *, loc=None, ip=None):
    """One 16-byte shared-memory load: four FP32 values."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None) -> None:
    """Order this thread's shared-memory stores before later async-proxy (tensor core) reads."""
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _add_bf16x2_squares(acc: "Float32", word: "Int32", *, loc=None, ip=None) -> "Float32":
    """``(acc + lo * lo) + hi * hi`` for the two BF16 halves of ``word`` (low half first), every
    product and sum rounded to nearest, no FMA: FlashInfer RMSNorm's per-thread accumulation."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [acc.ir_value(loc=loc, ip=ip), word.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 sqlo, sqhi;\n\t.reg .f32 sqa, sqb, sqp, sqs;\n\t"
            "shl.b32 sqlo, $2, 16;\n\tand.b32 sqhi, $2, 0xFFFF0000;\n\t"
            "mov.b32 sqa, sqlo;\n\tmov.b32 sqb, sqhi;\n\t"
            "mul.rn.f32 sqp, sqa, sqa;\n\tadd.rn.f32 sqs, $1, sqp;\n\t"
            "mul.rn.f32 sqp, sqb, sqb;\n\tadd.rn.f32 $0, sqs, sqp;\n\t}",
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _add_rn(a: "Float32", b: "Float32", *, loc=None, ip=None) -> "Float32":
    """``a + b`` rounded to nearest (never contracted)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _f32_hex(value: float) -> str:
    """PTX literal of the FP32 rounding of ``value``."""
    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


@dsl_user_op
def _rmsnorm_rstd(total: "Float32", hidden: int, eps: float, *, loc=None, ip=None) -> "Float32":
    """FlashInfer RMSNorm's ``rsqrt.approx.ftz(total / hidden + eps)``, the division and the sum
    rounded to nearest (IEEE)."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [total.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .f32 rsm, rst;\n\t"
            f"div.rn.f32 rsm, $1, {_f32_hex(float(hidden))};\n\t"
            f"add.rn.f32 rst, rsm, {_f32_hex(eps)};\n\t"
            "rsqrt.approx.ftz.f32 $0, rst;\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _rmsnorm_bf16x2(
    x_word: "Int32", w_word: "Int32", rstd: "Float32", *, loc=None, ip=None
) -> "Int32":
    """FlashInfer RMSNorm's output for two BF16 elements (low half first): ``(x * rstd) * (w + 0)``
    rounded to nearest at every step, then to BF16 (nearest even)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                x_word.ir_value(loc=loc, ip=ip),
                w_word.ir_value(loc=loc, ip=ip),
                rstd.ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .b32 rnxl, rnxh, rnwl, rnwh;\n\t"
            ".reg .f32 rfxl, rfxh, rfwl, rfwh, ryl, ryh;\n\t"
            "shl.b32 rnxl, $1, 16;\n\tand.b32 rnxh, $1, 0xFFFF0000;\n\t"
            "shl.b32 rnwl, $2, 16;\n\tand.b32 rnwh, $2, 0xFFFF0000;\n\t"
            "mov.b32 rfxl, rnxl;\n\tmov.b32 rfxh, rnxh;\n\t"
            "mov.b32 rfwl, rnwl;\n\tmov.b32 rfwh, rnwh;\n\t"
            "add.rn.f32 rfwl, rfwl, 0f00000000;\n\tadd.rn.f32 rfwh, rfwh, 0f00000000;\n\t"
            "mul.rn.f32 ryl, rfxl, $3;\n\tmul.rn.f32 ryl, ryl, rfwl;\n\t"
            "mul.rn.f32 ryh, rfxh, $3;\n\tmul.rn.f32 ryh, ryh, rfwh;\n\t"
            "cvt.rn.bf16x2.f32 $0, ryh, ryl;\n\t}",
            "=r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ld_shared_v4_b32(address: "Int32", *, loc=None, ip=None):
    """One 16-byte shared-memory load: four words."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _mul_bf16x2(x_word: "Int32", g_word: "Int32", *, loc=None, ip=None) -> "Int32":
    """Two BF16 products (low half first), each rounded once to BF16 (nearest even).

    The FP32 product of two BF16 values is exact, so this equals rounding the FP32 product.
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [x_word.ir_value(loc=loc, ip=ip), g_word.ir_value(loc=loc, ip=ip)],
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _fma_squares_bf16x2(acc: "Float32", word: "Int32", *, loc=None, ip=None) -> "Float32":
    """``acc + lo * lo + hi * hi`` for the BF16 halves of ``word``, as two FP32 fused multiply-adds."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [acc.ir_value(loc=loc, ip=ip), word.ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b32 fsl, fsh;\n\t.reg .f32 fsa, fsb, fst;\n\t"
            "shl.b32 fsl, $2, 16;\n\tand.b32 fsh, $2, 0xFFFF0000;\n\t"
            "mov.b32 fsa, fsl;\n\tmov.b32 fsb, fsh;\n\t"
            "fma.rn.f32 fst, fsa, fsa, $1;\n\tfma.rn.f32 $0, fsb, fsb, fst;\n\t}",
            "=f,f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


OWNER_RANK = 0


class SplitKDenseGemmKernel:
    """Standalone BF16/FP16 GEMM with a cluster-local split-K reduction."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        reserve_full_smem: bool = True,
        publish: LamportPublish | None = None,
        hnorm_b: HnormB | None = None,
        tmem_cols: int = _TMEM_COLS,
        mailbox_in_stages: bool = False,
        b_stages: int = 0,
        mailbox_interleaved: bool = False,
        launch_dependents_early: bool = True,
        row_rms: RowRmsB | None = None,
    ) -> None:
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        # Shared memory requested at launch: the full capacity (one CTA per SM) or only the
        # tactic's footprint. The kernel itself is the same either way.
        self.reserve_full_smem = reserve_full_smem
        # Tensor-memory columns the MMA warp allocates (a power of two of at least 32 that holds
        # the FP32 accumulator's mma_n columns).
        if (
            tmem_cols < 32
            or tmem_cols > 512
            or tmem_cols & (tmem_cols - 1)
            or tmem_cols < tactic.mma_n
        ):
            raise ValueError(f"tmem_cols {tmem_cols} does not hold the {tactic.mma_n}-column tile")
        self.tmem_cols = tmem_cols
        # mailbox_in_stages: the peers' FP32 partials land in the owner's A stages once its MMAs
        # are done with them (the owner then arrives on each peer's mailbox-free barrier), instead
        # of a mailbox of their own; that shared memory holds more A/B stages.
        if mailbox_in_stages and (
            not mailbox_fits_stages(tactic) or publish is not None or hnorm_b is not None
        ):
            raise ValueError(
                f"mailbox_in_stages needs split-K partials that fit in the A stages, "
                f"no publish and no HnormB: {tactic}"
            )
        self.mailbox_in_stages = mailbox_in_stages
        # b_stages > 0: the B tiles get a ring of their own, b_stages deep, with their own full and
        # empty barriers; the A ring keeps tactic.ab_stages. A deeper A ring prefetches more weights
        # (also before the grid dependency wait) where B's stages would otherwise cap it.
        if b_stages and (hnorm_b is not None or publish is not None):
            raise ValueError("a separate B ring excludes HnormB and the Lamport publish")
        self.split_b_ring = b_stages > 0
        self.num_b_stage = b_stages if b_stages > 0 else tactic.ab_stages
        # mailbox_interleaved: each peer's partial lies in the mailbox as [4-value vector][epilogue
        # thread][4], so that a warp's remote stores and the owner's 16-byte loads cover contiguous
        # shared memory (no bank conflicts). The same values are summed in the same order.
        self.mailbox_interleaved = mailbox_interleaved
        # launch_dependents_early: the weight-load warp releases the PDL dependents once its loads
        # are issued. Without it no CTA releases them, so they launch when this grid completes and
        # find the SMs it held free.
        if not launch_dependents_early and publish is not None:
            raise ValueError("the Lamport publish keeps its own dependent release")
        self.launch_dependents_early = launch_dependents_early
        self.launch_smem_bytes = tactic_launch_smem_bytes(tactic, mailbox_in_stages, b_stages)
        # Lamport publish (run_splitk_publish): the owner stages the reduced FP32 tile in shared
        # memory and stores it into the all-reduce workspace instead of writing C.
        self.publish = publish
        # HnormB: warp 1 computes B (FlashInfer's RMSNorm of the activation row) instead of
        # loading it; the norm weight takes the bias operand's place.
        self.hnorm_b = hnorm_b
        if hnorm_b is not None and not hnorm_b_supported(tactic, hnorm_b.hidden):
            raise ValueError(
                f"HnormB needs hidden {HNORM_B_HIDDEN} and a tactic of hnorm_b_supported: "
                f"{tactic}, hidden {hnorm_b.hidden}"
            )
        if hnorm_b is not None and (has_bias or publish is not None):
            raise ValueError("HnormB excludes a bias and the Lamport publish")
        # RowRmsB: the epilogue warps scale B by the norm weight (the bias operand) in shared memory
        # between its TMA load and its MMA, and the outputs by each row's rsqrt(mean(x^2) + eps).
        self.row_rms = row_rms
        if row_rms is not None and (
            not row_rms_b_supported(tactic, b_stages, row_rms.k)
            or has_bias
            or publish is not None
            or hnorm_b is not None
        ):
            raise ValueError(
                f"RowRmsB needs a tactic of row_rms_b_supported with its own B ring, no bias, "
                f"no publish and no HnormB: {tactic}, b_stages {b_stages}, k {row_rms.k}"
            )
        if row_rms is not None:
            self.launch_smem_bytes = min(
                _align_up(_smem_bytes(tactic, tactic.ab_stages, False, b_stages, row_rms.k), 1024)
                + _LAUNCH_SMEM_HEADROOM_BYTES,
                _SMEM_CAPACITY_BYTES,
            )
        if publish is not None:
            if has_bias or self.cta_m % _PUBLISH_VEC_ELEMENTS:
                raise ValueError(f"Lamport publish needs no bias and 8-column tiles: {tactic}")
            self.launch_smem_bytes = min(
                _align_up(_publish_smem_bytes(tactic, tactic.ab_stages), 1024)
                + _LAUNCH_SMEM_HEADROOM_BYTES,
                _SMEM_CAPACITY_BYTES,
            )

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)

        values_per_thread = (tactic.mma_m * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"CTA tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                f"{values_per_thread} "
                "values per epilogue thread; remote stores require a multiple of 4"
            )
        self.mailbox_elements = (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        bias: cute.Tensor,
        lamport: cute.Tensor,
        flags: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias, lamport, flags).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(
                utils.get_smem_capacity_in_bytes("sm_100")
                if self.reserve_full_smem
                else self.launch_smem_bytes
            ),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
        mLamport: cute.Tensor,  # BF16 all-reduce workspace, multicast address; dead without publish
        mFlags: cute.Tensor,  # The workspace's buffer flags; dead without publish
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, self.num_b_stage),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        if cutlass.const_expr(self.split_b_ring):
            bar_full_b = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(self.num_b_stage),
                alignment=_MBARRIER_BYTES,
            ).iterator
            bar_empty_b = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(self.num_b_stage),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            bar_full_b = bar_full
            bar_empty_b = bar_empty
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator

        if cutlass.const_expr(self.split_k > 1):
            if cutlass.const_expr(self.mailbox_in_stages):
                # The partials land in the A stages once the owner's MMAs are done with them.
                mailbox = cute.make_tensor(
                    cute.recast_ptr(sA.iterator, dtype=cutlass.Float32),
                    cute.make_layout(self.mailbox_elements),
                )
            else:
                mailbox = cute_ext.allocate(
                    cutlass.Float32,
                    cute.AddressSpace.smem,
                    cute.make_layout(self.mailbox_elements),
                    alignment=_MAILBOX_ALIGN_BYTES,
                )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
            if cutlass.const_expr(self.mailbox_in_stages):
                # A peer's signal that the owner's A stages may take its partials.
                bar_mailbox_free = cute_ext.allocate(
                    cutlass.Int64,
                    cute.AddressSpace.smem,
                    cute.make_layout(1),
                    alignment=_MBARRIER_BYTES,
                ).iterator
            else:
                bar_mailbox_free = bar_reduce
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog
            bar_mailbox_free = bar_mma_epilog

        if cutlass.const_expr(self.publish is not None):
            # Lamport publish: the reduced FP32 tile, M-major as the output tile.
            sPub = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout((self.cta_m, self.cta_n)),
                alignment=_PUBLISH_ALIGN_BYTES,
            )
        else:
            # Dummy operand for the compile-time-elided publish.
            sPub = sA

        if cutlass.const_expr(self.row_rms is not None):
            # RowRmsB: a B stage's release to the MMA once the epilogue warps scaled it, the norm
            # weight, and two FP32 values per token row (a partial square sum, then the row scale).
            bar_xf = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(self.num_b_stage),
                alignment=_MBARRIER_BYTES,
            ).iterator
            sGamma = cute_ext.allocate(
                ab_dtype,
                cute.AddressSpace.smem,
                cute.make_layout(self.row_rms.k),
                alignment=_ROW_RMS_ALIGN_BYTES,
            )
            sRowStat = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(2 * self.cta_n),
                alignment=_ROW_RMS_ALIGN_BYTES,
            )
        else:
            # Dummy operands for the compile-time-elided RowRmsB.
            bar_xf = bar_full_b
            sGamma = sA
            sRowStat = sA

        if warp_idx == 0:
            with cute.arch.elect_one():
                if cutlass.const_expr(self.row_rms is not None):
                    # One arrival per epilogue warp.
                    for i in range(self.num_b_stage):
                        cute.arch.mbarrier_init(bar_xf + i, 4)
                if cutlass.const_expr(self.split_b_ring):
                    # One producer per ring: A's arrive_and_expect_tx, or B's.
                    for i in range(stages):
                        cute.arch.mbarrier_init(bar_full + i, 1)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                    for i in range(self.num_b_stage):
                        cute.arch.mbarrier_init(bar_full_b + i, 1)
                        cute.arch.mbarrier_init(bar_empty_b + i, 1)
                else:
                    for i in range(stages):
                        cute.arch.mbarrier_init(bar_full + i, 2)
                        cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)
                    if cutlass.const_expr(self.mailbox_in_stages):
                        # The owner's arrival.
                        cute.arch.mbarrier_init(bar_mailbox_free, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            if cutlass.const_expr(self.hnorm_b is not None):
                self.hnorm_b_warp(bar_full, bar_empty, sB, mB, mBias, k_tile_start)
            else:
                self.dma_warp(
                    bar_full_b,
                    bar_empty_b,
                    bar_tma_epilog,
                    cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                    sB,
                    cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                    k_tile_start,
                    k_tile_count,
                    False,
                )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_xf,
                bar_empty_b,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                sPub,
                mC,
                mLamport,
                mFlags,
                bidx,
                n_idx,
                bar_mailbox_free,
                bar_full_b,
                bar_xf,
                sB,
                sGamma,
                sRowStat,
                mBias,
                k_tile_count,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_b_stage
        if cutlass.const_expr(is_a):
            stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(
            is_a
            and self.use_pdl
            and self.launch_dependents_early
            and not (self.publish is not None and self.publish.trigger_after_wait)
        ):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not is_a and self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)
        self._drain_producer(bar_empty, empty_phase, k_tile_count, stages)

    @cute.experimental.jit
    def hnorm_b_warp(
        self,
        bar_full,
        bar_empty,
        sB: cute.Tensor,
        mX: cute.Tensor,
        mW: cute.Tensor,
        k_tile_start: cutlass.Int32,
    ):
        """B of HnormB: FlashInfer's RMSNorm of the activation row x with weight w, written into
        this CTA's B stages where the TMA would have put the norm's output.

        FlashInfer's CuTe kernel (H = 6144) gives a row to 64 threads; thread t holds the 8-element
        vectors t + 64 b (b = 0..11), sums their squares in order, and the row sum is the butterfly
        sum of threads 0-31 plus that of threads 32-63. Lane l of this warp plays threads l and
        l + 32: it holds the vectors l + 32 m (m = 0..23; even m are thread l's, odd m thread
        l + 32's). The B tiles are 8 x 128 K-major with the K_SW128 layout: row 0 (the activation;
        rows 1-7 are zero, as the TMA fills rows past M) is unswizzled, K columns [0, 64) at bytes
        [0, 128) and [64, 128) at bytes [1024, 1152) of the stage."""
        hidden = self.hnorm_b.hidden
        lane = cute.arch.lane_idx()
        vec_per_tile = self.cta_k // _HNORM_B_VEC_ELEMENTS
        k_tiles = hidden // self.cta_k // self.split_k
        per_lane = k_tiles * vec_per_tile // _WARP_LANES
        row_rounds = hidden // _HNORM_B_VEC_ELEMENTS // _WARP_LANES
        stage_bytes = self.cta_n * self.cta_k * _AB_ELEMENT_BYTES
        atom_bytes = stage_bytes // 2
        row_bytes = atom_bytes // self.cta_n
        sb = sB.iterator.toint()
        zero = Int32(0)

        # Rows 1-7 of the stages this CTA uses.
        zero_vecs = (atom_bytes - row_bytes) // 16
        for s in cutlass.range_constexpr(k_tiles):
            for r in cutlass.range_constexpr((2 * zero_vecs + _WARP_LANES - 1) // _WARP_LANES):
                idx = lane + Int32(r * _WARP_LANES)
                if idx < 2 * zero_vecs:
                    atom = idx // zero_vecs
                    _st_shared_v4(
                        sb
                        + Int32(s * stage_bytes + row_bytes)
                        + atom * atom_bytes
                        + (idx - atom * zero_vecs) * 16,
                        zero,
                        zero,
                        zero,
                        zero,
                    )

        # This CTA's K slice is the vectors [v0, v0 + k_tiles * vec_per_tile); lane l produces the
        # slice vectors v0 + l + 32 j. Their norm weights are constants: loaded before the wait.
        v0 = k_tile_start * vec_per_tile
        w_words = []
        for j in cutlass.range_constexpr(per_lane):
            v = v0 + lane + Int32(j * _WARP_LANES)
            w_words.append(_ld_global_cg_v4((mW.iterator + v * _HNORM_B_VEC_ELEMENTS).toint()))

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        x_words = []
        for m in cutlass.range_constexpr(row_rounds):
            v = lane + Int32(m * _WARP_LANES)
            x_words.append(_ld_global_cg_v4((mX.iterator + v * _HNORM_B_VEC_ELEMENTS).toint()))
        xs_words = []
        for j in cutlass.range_constexpr(per_lane):
            v = v0 + lane + Int32(j * _WARP_LANES)
            xs_words.append(_ld_global_cg_v4((mX.iterator + v * _HNORM_B_VEC_ELEMENTS).toint()))

        acc_lo = Float32(0.0)
        acc_hi = Float32(0.0)
        for b in cutlass.range_constexpr(row_rounds // 2):
            for q in cutlass.range_constexpr(4):
                acc_lo = _add_bf16x2_squares(acc_lo, x_words[2 * b][q])
        for b in cutlass.range_constexpr(row_rounds // 2):
            for q in cutlass.range_constexpr(4):
                acc_hi = _add_bf16x2_squares(acc_hi, x_words[2 * b + 1][q])
        for offset in (1, 2, 4, 8, 16):
            acc_lo = _add_rn(acc_lo, cute.arch.shuffle_sync_bfly(acc_lo, offset))
            acc_hi = _add_rn(acc_hi, cute.arch.shuffle_sync_bfly(acc_hi, offset))
        rstd = _rmsnorm_rstd(_add_rn(acc_lo, acc_hi), hidden, self.hnorm_b.eps)

        for j in cutlass.range_constexpr(per_lane):
            local = lane + Int32(j * _WARP_LANES)
            tile = local // vec_per_tile
            col_vec = local - tile * vec_per_tile
            col_atom = col_vec // (vec_per_tile // 2)
            _st_shared_v4(
                sb
                + tile * stage_bytes
                + col_atom * atom_bytes
                + (col_vec - col_atom * (vec_per_tile // 2)) * 16,
                _rmsnorm_bf16x2(xs_words[j][0], w_words[j][0], rstd),
                _rmsnorm_bf16x2(xs_words[j][1], w_words[j][1], rstd),
                _rmsnorm_bf16x2(xs_words[j][2], w_words[j][2], rstd),
                _rmsnorm_bf16x2(xs_words[j][3], w_words[j][3], rstd),
            )
        # The tensor core reads the stages through the async proxy.
        _fence_proxy_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            for s in cutlass.range_constexpr(k_tiles):
                cute.arch.mbarrier_arrive(bar_full + s)
        # As the TMA producer: wait until the MMA has released every stage this CTA filled.
        self._drain_producer(
            bar_empty, Int32(1 ^ (k_tiles // self.num_ab_stage)), Int32(k_tiles), self.num_ab_stage
        )

    @cute.experimental.jit
    def _row_rms_b_tiles(
        self,
        bar_full_b,
        bar_xf,
        sB: cute.Tensor,
        sGamma: cute.Tensor,
        sRowStat: cute.Tensor,
        mGamma: cute.Tensor,
        k_tile_count: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """RowRmsB in the epilogue warps: every B tile between its TMA load and its MMA.

        A B stage is cta_n x 128 K-major with the K_SW128 layout: two 64-column swizzle atoms of
        cta_n 128-byte rows, a row's eight 16-byte chunks permuted by the row index. Thread t takes
        token row t % cta_n of atom t // cta_n: it adds the squares of the row's 64 values to its
        four partial sums (FP32 FMAs), writes back each value times the norm weight of its K column
        (rounded to BF16), and once the warp is done the stage is released to the MMA (bar_xf). A
        thread starts at chunk t % 8, so that the eight rows of a quarter warp hit distinct banks.
        After the last K tile, sRowStat[cta_n + r] holds row r's rsqrt(sum / k + eps), behind a
        barrier of the epilogue warps.
        """
        k = self.row_rms.k
        stage_bytes = self.cta_n * self.cta_k * _AB_ELEMENT_BYTES
        atom_bytes = stage_bytes // 2
        row_bytes = atom_bytes // self.cta_n
        atom_cols = self.cta_k // 2
        gamma_vecs = k * _AB_ELEMENT_BYTES // 16
        row = epi_tid % self.cta_n
        atom = epi_tid // self.cta_n

        # The norm weight is a constant: staged before any wait.
        gamma_smem = sGamma.iterator.toint()
        for i in cutlass.range_constexpr(
            (gamma_vecs + self.epilog_threads - 1) // self.epilog_threads
        ):
            v = epi_tid + Int32(i * self.epilog_threads)
            if v < gamma_vecs:
                words = _ld_global_cg_v4((mGamma.iterator + v * 8).toint())
                _st_shared_v4(gamma_smem + v * 16, words[0], words[1], words[2], words[3])
        cute.arch.barrier(barrier_id=_ROW_RMS_BARRIER_ID, number_of_threads=self.epilog_threads)

        line = sB.iterator.toint() + atom * atom_bytes + row * row_bytes
        swz = row % _ROW_RMS_CHUNKS
        # Four partial sums (one per word of a chunk) keep the FMA chains short.
        ssq0 = Float32(0.0)
        ssq1 = Float32(0.0)
        ssq2 = Float32(0.0)
        ssq3 = Float32(0.0)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_b_stage
            cute.arch.mbarrier_wait(bar_full_b + stage, full_phase)
            base = line + stage * stage_bytes
            gbase = gamma_smem + (k_tile * self.cta_k + atom * atom_cols) * _AB_ELEMENT_BYTES
            # Every load of the row's chunks and their norm weights first, then the arithmetic,
            # then the stores.
            xs = []
            gs = []
            for i in cutlass.range_constexpr(_ROW_RMS_CHUNKS):
                chunk = (Int32(i) + swz) % _ROW_RMS_CHUNKS
                xs.append(_ld_shared_v4_b32(base + chunk * 16))
                gs.append(_ld_shared_v4_b32(gbase + (chunk ^ swz) * 16))
            for i in cutlass.range_constexpr(_ROW_RMS_CHUNKS):
                ssq0 = _fma_squares_bf16x2(ssq0, xs[i][0])
                ssq1 = _fma_squares_bf16x2(ssq1, xs[i][1])
                ssq2 = _fma_squares_bf16x2(ssq2, xs[i][2])
                ssq3 = _fma_squares_bf16x2(ssq3, xs[i][3])
            for i in cutlass.range_constexpr(_ROW_RMS_CHUNKS):
                chunk = (Int32(i) + swz) % _ROW_RMS_CHUNKS
                _st_shared_v4(
                    base + chunk * 16,
                    _mul_bf16x2(xs[i][0], gs[i][0]),
                    _mul_bf16x2(xs[i][1], gs[i][1]),
                    _mul_bf16x2(xs[i][2], gs[i][2]),
                    _mul_bf16x2(xs[i][3], gs[i][3]),
                )
            # The tensor core reads the stage through the async proxy.
            _fence_proxy_async_shared()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(bar_xf + stage)
            if stage == self.num_b_stage - 1:
                full_phase = full_phase ^ 1

        ssq = (ssq0 + ssq1) + (ssq2 + ssq3)
        # A row's sum: atom 0's part plus atom 1's part.
        if atom == 1:
            sRowStat[row] = ssq
        cute.arch.barrier(barrier_id=_ROW_RMS_BARRIER_ID, number_of_threads=self.epilog_threads)
        if atom == 0:
            sRowStat[self.cta_n + row] = _rmsnorm_rstd(
                _add_rn(ssq, sRowStat[row]), k, self.row_rms.eps
            )
        cute.arch.barrier(barrier_id=_ROW_RMS_BARRIER_ID, number_of_threads=self.epilog_threads)

    @cute.experimental.jit
    def _drain_producer(
        self,
        bar_empty,
        empty_phase: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        stages: cutlass.Constexpr,
    ):
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_full_b,
        bar_empty_b,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = self.tmem_cols
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        full_phase_b = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            stage_b = stage
            if cutlass.const_expr(self.split_b_ring):
                stage_b = k_tile % self.num_b_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            if cutlass.const_expr(self.split_b_ring):
                cute.arch.mbarrier_wait(bar_full_b + stage_b, full_phase_b)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage_b], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
                if cutlass.const_expr(self.split_b_ring):
                    tcgen05.commit(bar_empty_b + stage_b, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1
            if cutlass.const_expr(self.split_b_ring):
                if stage_b == self.num_b_stage - 1:
                    full_phase_b = full_phase_b ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        sPub,
        mC: cute.Tensor,
        mLamport: cute.Tensor,
        mFlags: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        bar_mailbox_free,
        bar_full_b,
        bar_xf,
        sB: cute.Tensor,
        sGamma: cute.Tensor,
        sRowStat: cute.Tensor,
        mGamma: cute.Tensor,
        k_tile_count: cutlass.Int32,
    ):
        lamport_buffer = Int64(0)
        if cutlass.const_expr(self.publish is not None):
            # The flags are read after the grid dependency wait: the previous all-reduce, which
            # rotated them, precedes this grid's primary. Every CTA's epilogue warps wait; with
            # trigger_after_wait they also release the dependents here.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
                if cutlass.const_expr(self.publish.trigger_after_wait):
                    cute.arch.griddepcontrol_launch_dependents()
            lamport_buffer = _lamport_current_buffer(
                mFlags.iterator.toint(), mLamport.iterator.toint()
            )

        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

        if cutlass.const_expr(self.row_rms is not None):
            # The B tiles pass through here between their TMA load and their MMA.
            self._row_rms_b_tiles(
                bar_full_b, bar_xf, sB, sGamma, sRowStat, mGamma, k_tile_count, epi_tid
            )

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            if split_rank == OWNER_RANK:
                cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                cute_ext.partition_and_copy(
                    cute.make_tiled_copy_D(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                        tiled_copy_t2r,
                    ).get_slice(epi_tid),
                    cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                    rBias,
                )
                rBiasAcc.store(rBias.load().to(self.acc_dtype))

        if cutlass.const_expr(self.row_rms is not None):
            # Every output's row scale (the token's rsqrt(mean(x^2) + eps)), in this thread's
            # accumulator layout: the bias load's partition over a view that broadcasts the scales
            # along the output features.
            rScale = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            sScale = cute.make_tensor(
                sRowStat.iterator + self.cta_n,
                cute.make_layout((self.cta_m, self.cta_n), stride=(0, 1)),
            )
            cute_ext.partition_and_copy(
                cute.make_tiled_copy_D(
                    cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.acc_dtype),
                    tiled_copy_t2r,
                ).get_slice(epi_tid),
                cute.flat_divide(sScale, epi_tile)[None, None, 0, 0],
                rScale,
            )

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                if cutlass.const_expr(self.mailbox_in_stages):
                    # The mailbox is the owner's A stages: wait until its MMAs are done with them.
                    _wait_parity_cluster(bar_mailbox_free, Int32(0))
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    if cutlass.const_expr(self.mailbox_interleaved):
                        dst = (
                            mailbox.iterator
                            + (split_rank - Int32(1)) * values_per_peer
                            + (value_idx // 4) * (self.epilog_threads * 4)
                            + epi_tid * 4
                        )
                    else:
                        dst = (
                            mailbox.iterator
                            + (split_rank - Int32(1)) * values_per_peer
                            + epi_tid * values_per_thread
                            + value_idx
                        )
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        dst,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if cutlass.const_expr(self.mailbox_in_stages):
                    # This CTA's MMAs have completed (bar_mma_epilog), so its A stages may take
                    # the partials: thread p signals peer p + 1.
                    if epi_tid < self.split_k - 1:
                        _arrive_shared_remote(bar_mailbox_free, epi_tid + Int32(1))
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    if cutlass.const_expr(self.mailbox_interleaved):
                        for vec in cutlass.range_constexpr(values_per_thread // 4):
                            vals = _ld_shared_v4_f32(
                                mailbox.iterator
                                + peer * values_per_peer
                                + vec * (self.epilog_threads * 4)
                                + epi_tid * 4
                            )
                            for j in cutlass.range_constexpr(4):
                                rAcc[4 * vec + j] = rAcc[4 * vec + j] + vals[j]
                    else:
                        for value_idx in cutlass.range_constexpr(values_per_thread):
                            rAcc[value_idx] = (
                                rAcc[value_idx]
                                + mailbox[
                                    peer * values_per_peer + epi_tid * values_per_thread + value_idx
                                ]
                            )

        if split_rank == OWNER_RANK:
            if cutlass.const_expr(self.publish is not None):
                # Stage the tile by output coordinate, then store eight columns per thread.
                cute_ext.partition_and_copy(thr_t2r, rAcc, sPub)
                cute.arch.barrier(
                    barrier_id=_PUBLISH_BARRIER_ID, number_of_threads=self.epilog_threads
                )
                self._publish_tile(sPub, mC, lamport_buffer, bidx, n_idx, epi_tid)
            else:
                if cutlass.const_expr(self.has_bias):
                    rAcc.store(rAcc.load() + rBiasAcc.load())
                if cutlass.const_expr(self.row_rms is not None):
                    rAcc.store(rAcc.load() * rScale.load())

                rD.store(rAcc.load().to(c_dtype))
                # Preserve TMEM coordinates; the copy predicates output tails.
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        # The reduction mbarrier covers remote stores; no cluster barrier needed.

    @cute.experimental.jit
    def _publish_tile(
        self,
        sPub: cute.Tensor,
        mC: cute.Tensor,
        lamport_buffer: Int64,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        epi_tid: cutlass.Int32,
    ):
        """Store the staged tile into the Lamport buffer: eight consecutive output columns of one
        row per thread, rounded to BF16 as C would be, -0.0 as +0.0, one 16-byte store each at
        [token][rank][column] (the one-shot all-reduce's slot of this rank)."""
        vecs_m = cutlass.const_expr(self.cta_m // _PUBLISH_VEC_ELEMENTS)
        num_vecs = cutlass.const_expr(vecs_m * self.cta_n)
        n_feat = cute.size(mC, mode=[0])
        n_tok = cute.size(mC, mode=[1])
        for it in cutlass.range_constexpr(
            (num_vecs + self.epilog_threads - 1) // self.epilog_threads
        ):
            vec = epi_tid + it * self.epilog_threads
            row = vec // vecs_m
            m8 = (vec - row * vecs_m) * _PUBLISH_VEC_ELEMENTS
            token = n_idx * self.cta_n + row
            feature = bidx * self.cta_m + m8
            if vec < num_vecs:
                if token < n_tok:
                    if feature < n_feat:
                        words = []
                        for j in cutlass.range_constexpr(_PUBLISH_VEC_ELEMENTS // 2):
                            words.append(
                                _cvt_bf16x2_lamport(
                                    sPub[m8 + 2 * j, row], sPub[m8 + 2 * j + 1, row]
                                )
                            )
                        slot = Int64(token * self.publish.world_size + self.publish.rank)
                        offset = (slot * Int64(n_feat) + Int64(feature)) * 2
                        _store_global_v4(
                            lamport_buffer + offset, words[0], words[1], words[2], words[3]
                        )


_SUPPORTED_TORCH_DTYPES = (_torch.bfloat16, _torch.float16)


@cute.experimental.jit
def _bmm_no_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # C also stands in for the dead bias and Lamport operands.
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[0, 1, 2])),
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_row_rms(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    norm_weight: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # RowRmsB: b is the raw activation and the norm weight takes the dead bias operand's place
    # (with C's layout: only its address is read); C stands in for the dead Lamport operands.
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(norm_weight.iterator, c.layout),
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_bias(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    bias: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # C also stands in for the dead Lamport operands.
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        cute.make_tensor(bias.iterator, cute.select(bias.layout, mode=[1, 2, 0])),
        c,
        c,
        stream,
    )


@cute.experimental.jit
def _bmm_publish(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    lamport: cute.Tensor,
    flags: cute.Tensor,
    stream: _cuda.CUstream,
):
    c = cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0]))
    # C (not written) also stands in for the dead bias.
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        c,
        c,
        lamport,
        flags,
        stream,
    )


def _from_dlpack_dynamic(tensor, leading_dim: int, assumed_align: int = 32):
    return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(
        leading_dim=leading_dim
    )


def _detect_leading_dim(tensor: _torch.Tensor) -> int:
    # Ignore synthetic batch stride, including 1x1.
    for dim, stride in enumerate(tensor.stride()[1:], start=1):
        if stride == 1:
            return dim
    raise ValueError("tensor has no stride-1 dimension")


def _make_layout_tensor(
    shape: tuple[int, ...], dtype: _torch.dtype, leading_dim: int
) -> _torch.Tensor:
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [leading_dim]
    return _torch.empty(
        tuple(shape[dim] for dim in permutation), dtype=dtype, device="cuda"
    ).permute([permutation.index(dim) for dim in range(len(shape))])


def _make_compile_repr_tensors(
    dtype: _torch.dtype,
    has_bias: bool,
    a_leading: int,
    b_leading: int,
    c_leading: int,
):
    m, n, k, batch = 64, 8, _CTA_K, 1
    tensors = tuple(
        _from_dlpack_dynamic(_make_layout_tensor(shape, dtype, leading_dim), leading_dim)
        for shape, leading_dim in (
            ((batch, n, k), a_leading),
            ((batch, k, m), b_leading),
            ((batch, n, m), c_leading),
        )
    )
    if not has_bias:
        return (*tensors, None)
    return (
        *tensors,
        _from_dlpack_dynamic(
            _torch.empty((n,), dtype=dtype, device="cuda").as_strided(
                size=(batch, n, m), stride=(0, 1, 0)
            ),
            1,
            2,
        ),
    )


def _to_cute_swap(a, b, out, bias):
    a_swap = b.unsqueeze(0).transpose(-2, -1)
    b_swap = a.unsqueeze(0).transpose(-2, -1)
    c_swap = out.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_detect_leading_dim(tensor) for tensor in (a_swap, b_swap, c_swap))
    cute_tensors = tuple(
        _from_dlpack_dynamic(tensor, leading_dim)
        for tensor, leading_dim in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    if bias is None:
        return (*cute_tensors, None, leading_dims)
    return (
        *cute_tensors,
        _from_dlpack_dynamic(
            bias.as_strided(size=(1, c_swap.shape[1], c_swap.shape[2]), stride=(0, 1, 0)),
            1,
            2,
        ),
        leading_dims,
    )


@functools.cache
def _get_compiled_splitk_kernel(
    dtype,
    tactic: SplitKTactic,
    use_pdl: bool,
    has_bias: bool,
    leading_dims: tuple[int, int, int],
    reserve_full_smem: bool = True,
    publish: LamportPublish | None = None,
    tmem_cols: int = _TMEM_COLS,
    mailbox_in_stages: bool = False,
    b_stages: int = 0,
    mailbox_interleaved: bool = False,
    launch_dependents_early: bool = True,
    row_rms: RowRmsB | None = None,
):
    if dtype not in _SUPPORTED_TORCH_DTYPES:
        raise ValueError(f"split-K dense GEMM supports {_SUPPORTED_TORCH_DTYPES}; got {dtype}")

    row_rms_kwargs = {} if row_rms is None else {"row_rms": row_rms}
    kernel = SplitKDenseGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        has_bias=has_bias,
        reserve_full_smem=reserve_full_smem,
        publish=publish,
        tmem_cols=tmem_cols,
        mailbox_in_stages=mailbox_in_stages,
        b_stages=b_stages,
        mailbox_interleaved=mailbox_interleaved,
        launch_dependents_early=launch_dependents_early,
        **row_rms_kwargs,
    )
    compile_tensors = _make_compile_repr_tensors(dtype, has_bias, *leading_dims)
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    if row_rms is not None:
        weight = _from_dlpack_dynamic(_torch.empty((row_rms.k,), dtype=dtype, device="cuda"), 0)
        compiled = cute_ext.compile(_bmm_row_rms, kernel, *compile_tensors[:3], weight, stream)
    elif publish is not None:
        lamport = _from_dlpack_dynamic(_torch.empty((64,), dtype=dtype, device="cuda"), 0)
        flags = _from_dlpack_dynamic(_torch.empty((16,), dtype=_torch.int32, device="cuda"), 0)
        compiled = cute_ext.compile(
            _bmm_publish, kernel, *compile_tensors[:3], lamport, flags, stream
        )
    elif has_bias:
        compiled = cute_ext.compile(_bmm_bias, kernel, *compile_tensors, stream)
    else:
        compiled = cute_ext.compile(_bmm_no_bias, kernel, *compile_tensors[:3], stream)
    return compiled


def _validate_runtime_tensors(a, b, bias, out) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("split-K dense GEMM accepts only 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("all tensors must be on the same CUDA device")
    if a.dtype not in _SUPPORTED_TORCH_DTYPES or any(tensor.dtype != a.dtype for tensor in tensors):
        raise ValueError("a, b, out, and bias must share BF16 or FP16 dtype")
    if any(not (tensor.is_contiguous() or tensor.t().is_contiguous()) for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be dense row-major or column-major matrices")
    if any(tensor.data_ptr() % 32 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 32-byte aligned")
    # cute_ext.tma_load is used for a and b (not out, which uses partition_and_copy).
    # TMA requires the non-innermost byte stride to be a multiple of 16 bytes.
    # For a dense 2D tensor this is max(stride(0), stride(1)) * element_size.
    _TMA_STRIDE_ALIGN = 16
    for tensor, tname in ((a, "a"), (b, "b")):
        outer_stride_bytes = max(tensor.stride(0), tensor.stride(1)) * tensor.element_size()
        if outer_stride_bytes % _TMA_STRIDE_ALIGN:
            raise ValueError(
                f"tensor '{tname}' outer byte stride {outer_stride_bytes} is not a "
                f"multiple of {_TMA_STRIDE_ALIGN}; TMA load requires "
                f"{_TMA_STRIDE_ALIGN}-byte-aligned strides "
                f"(shape={tuple(tensor.shape)}, strides={tensor.stride()})."
            )

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != n or not bias.is_contiguous()):
        raise ValueError(
            f"bias must be contiguous with shape {(n,)}, "
            f"got shape {tuple(bias.shape)} and stride {bias.stride()}"
        )

    return m, n, k


def run_splitk_dense(
    a: _torch.Tensor,
    b: _torch.Tensor,
    bias: _torch.Tensor | None,
    out: _torch.Tensor,
    pdl: bool,
    tactic: SplitKTactic,
    reserve_full_smem: bool = True,
    mailbox_in_stages: bool = False,
    b_stages: int = 0,
    mailbox_interleaved: bool = False,
    launch_dependents_early: bool = True,
) -> _torch.Tensor:
    """Run ``A[M,K] @ B[K,N]`` with the split-K BF16/FP16 GEMM kernel.

    Args:
        a: Input tensor of shape ``[M, K]``, row-major BF16/FP16.
        b: Weight tensor of shape ``[K, N]``, column-major BF16/FP16
           (i.e. the transpose of a ``[N, K]`` contiguous weight matrix).
        bias: Optional bias of shape ``[N]``, or ``None``.
        out: Pre-allocated output tensor of shape ``[M, N]``, BF16/FP16.
        pdl: Whether to use Programmatic Dependent Launch.
        tactic: The ``SplitKTactic`` specifying tile and pipeline parameters.
        reserve_full_smem: Reserve the full shared-memory capacity per CTA (one CTA per
            SM). ``False`` reserves only ``tactic_launch_smem_bytes(tactic)``; same kernel
            and result.
        mailbox_in_stages: The split-K partials land in the owner's A stages once its MMAs are
            done with them, instead of a mailbox of their own (mailbox_fits_stages); same
            products and summation order.
        b_stages: The B tiles get a ring of their own with this many stages (0: they share the
            tactic's stages); same products and summation order.
        mailbox_interleaved: The split-K partials lie in the mailbox thread-interleaved (no
            shared-memory bank conflicts); same products and summation order.
        launch_dependents_early: The kernel releases its PDL dependents once its weight loads are
            issued; without it they launch when it completes.

    Returns:
        The ``out`` tensor, filled with the GEMM result.
    """
    validate_tactic(
        tactic,
        *_validate_runtime_tensors(a, b, bias, out),
        mailbox_in_stages=mailbox_in_stages,
        b_stages=b_stages,
    )
    has_bias = bias is not None
    cute_tensors = _to_cute_swap(a, b, out, bias)
    compile_kwargs = {"mailbox_in_stages": True} if mailbox_in_stages else {}
    if b_stages:
        compile_kwargs["b_stages"] = b_stages
    if mailbox_interleaved:
        compile_kwargs["mailbox_interleaved"] = True
    if not launch_dependents_early:
        compile_kwargs["launch_dependents_early"] = False
    compiled = _get_compiled_splitk_kernel(
        dtype=a.dtype,
        tactic=tactic,
        use_pdl=pdl,
        has_bias=has_bias,
        leading_dims=cute_tensors[4],
        reserve_full_smem=reserve_full_smem,
        **compile_kwargs,
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(a.device).cuda_stream)
    if has_bias:
        compiled(*cute_tensors[:4], stream)
    else:
        compiled(*cute_tensors[:3], stream)
    return out


def run_splitk_row_rms(
    x: _torch.Tensor,
    w: _torch.Tensor,
    gamma: _torch.Tensor,
    out: _torch.Tensor,
    pdl: bool,
    launch: WideLaunch,
    eps: float,
) -> _torch.Tensor:
    """``RMSNorm(x) @ w^T`` in one launch, the norm folded into the GEMM (``RowRmsB``).

    Computes ``(x * gamma) @ w^T * rsqrt(mean(x^2) + eps)`` per row: x times the norm weight is
    rounded to BF16 before the GEMM and every output row is scaled by its rsqrt in FP32 before the
    BF16 store, where the separate RMSNorm rounds ``x * rsqrt * gamma`` to BF16 and the GEMM reads
    that.

    Args:
        x: The raw RMSNorm input [M, K] BF16, 33 <= M <= 64: unit inner stride, a row stride of at
            least K and a multiple of 8 elements, 32-byte aligned (a column slice of a wider
            output qualifies).
        w: Weight [N, K] BF16, contiguous.
        gamma: The RMSNorm weight [K] BF16, contiguous, 16-byte aligned.
        out: Output [M, N] BF16, contiguous.
        pdl: Whether to use Programmatic Dependent Launch.
        launch: ``row_rms_launch`` for this shape.
        eps: The RMSNorm epsilon.

    Returns:
        The ``out`` tensor.
    """
    tensors = (x, w, gamma, out)
    if any(not isinstance(t, _torch.Tensor) for t in tensors):
        raise ValueError("x, w, gamma and out must be torch tensors")
    if any(
        t.dtype != _torch.bfloat16 or t.device != x.device or t.device.type != "cuda"
        for t in tensors
    ):
        raise ValueError("x, w, gamma and out must be BF16 tensors on one CUDA device")
    if x.ndim != 2 or w.ndim != 2 or out.ndim != 2 or gamma.ndim != 1:
        raise ValueError("x, w and out must be 2D and gamma 1D")
    m, k = x.shape
    n = w.shape[0]
    if (
        x.stride(1) != 1
        or x.stride(0) < k
        or x.stride(0) % 8
        or x.data_ptr() % 32
        or not w.is_contiguous()
        or w.shape[1] != k
        or w.data_ptr() % 32
        or not out.is_contiguous()
        or out.shape != (m, n)
        or out.data_ptr() % 32
        or not gamma.is_contiguous()
        or gamma.shape[0] != k
        or gamma.data_ptr() % 16
    ):
        raise ValueError(
            f"run_splitk_row_rms: unsupported operands x {tuple(x.shape)} {x.stride()}, "
            f"w {tuple(w.shape)}, gamma {tuple(gamma.shape)}, out {tuple(out.shape)}"
        )
    validate_tactic(launch.tactic, m, n, k, b_stages=launch.b_stages)
    if not row_rms_b_supported(launch.tactic, launch.b_stages, k):
        raise ValueError(f"run_splitk_row_rms: the launch does not serve RowRmsB: {launch}")
    # detach(): module parameters (the weight, the norm weight) cannot be exported through DLPack
    # otherwise.
    cute_tensors = _to_cute_swap(x.detach(), w.detach().t(), out.detach(), None)
    compiled = _get_compiled_splitk_kernel(
        dtype=x.dtype,
        tactic=launch.tactic,
        use_pdl=pdl,
        has_bias=False,
        leading_dims=cute_tensors[4],
        b_stages=launch.b_stages,
        launch_dependents_early=launch.launch_dependents_early,
        row_rms=RowRmsB(k=k, eps=float(eps)),
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream)
    compiled(*cute_tensors[:3], _from_dlpack_dynamic(gamma.detach(), 0), stream)
    return out
