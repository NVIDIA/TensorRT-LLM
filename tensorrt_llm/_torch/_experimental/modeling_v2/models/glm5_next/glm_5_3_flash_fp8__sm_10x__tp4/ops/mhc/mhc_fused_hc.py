# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The boundary between two mHC-wrapped sublayers in one call (``mhc_fused_hc``).

``torch.ops.trtllm.mhc_fused_hc``, the op ``mHC.fused_hc`` calls: the previous sublayer's
post-mapping into the residual streams, then the next sublayer's pre-mapping of the new streams,
with that sublayer's input RMSNorm folded in. For ``M`` rows of ``mult = 4`` streams of ``hidden``
values it computes ``mhc_post_mapping`` followed by ``mhc_gemm_sqrsum`` and ``mhc_big_fuse`` on the
BF16 streams it stores (each of those modules states its step's formula)::

    streams = bf16(post[:, :, None] * x[:, None, :] + comb.mT @ residual)  # returned
    mixes = streams.view(M, 4 * hidden) @ fn.T  # [M, 24]
    sqrsum = streams.view(M, 4 * hidden).square().sum(-1)  # [M]
    post_out, comb_out, layer_input = mhc_big_fuse(mixes, sqrsum, streams, ...)

The op runs one of four backends, picked by ``tactic``:

* ``fused_half_mma``: two kernels. A tcgen05 TF32 GEMM fused with the post-mapping writes the
  streams and, per hidden split, a partial mixer product and sum of squares of the BF16-rounded
  streams; ``mhcBigFuseKernel`` sums the splits in a fixed order and finishes the boundary.
* ``fused_half_fma``: two kernels. The same on CUDA-core FP32 FMA, over the FP32 streams before
  their BF16 rounding, with ``hidden`` split into at most 8 slices.
* ``fused_all_mma``: one kernel (plus a counter-zeroing kernel when split): the TF32 GEMM, a
  barrier per 64 rows, then the split sum and the rest of the boundary inline.
* ``fused_all_fma``: one kernel (plus a workspace-zeroing kernel): FP32 FMA, whose at most two
  hidden splits are added onto one zeroed slot -- two addends sum the same in either order -- and
  whose last CTA of each row group finishes the boundary.

The TF32 backends read ``fn`` at TF32 precision. The checkpoint's mixer weights are BF16 values,
which TF32 holds exactly, as it does the BF16 streams. Every tactic gives the same bits run to
run; tactics sum in different orders (and the FMA backends give the mixer the unrounded streams),
so they differ from each other in the last bits.

``tactic`` is ``(backend, tile_n, num_k_splits, bigfuse_block_size, tile_m)``. ``fused_hc_tactic``
picks it as ``mHC.fused_hc`` does: the autotuner over ``fused_hc_tactics`` (one tactic for every TP
rank), the tensor-core fallback on a cache miss, and ``DETERMINISTIC_TACTIC`` under
``FORCE_DETERMINISTIC=1`` or ``FORCE_MHC_KERNEL_DETERMINISTIC=1``. ``fused_hc_scratch`` allocates
the three workspaces a tactic needs.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    ConstraintSpec,
    DistributedTuningStrategy,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from tensorrt_llm._utils import get_sm_version

from .mhc_big_fuse import GLM_SCALARS, mhc_big_fuse
from .mhc_gemm_sqrsum import mhc_gemm_sqrsum
from .mhc_post_mapping import mhc_post_mapping

#: GLM-5.3-Flash's hyper-connection: 4 streams of 4096, 24 mixer outputs.
MULT = 4
HIDDEN = 4096
MIX = (2 + MULT) * MULT

#: ``tactic[0]`` -> the op's backend code.
BACKENDS = {"fused_half_mma": 0, "fused_half_fma": 1, "fused_all_mma": 2, "fused_all_fma": 3}

#: The tactic under deterministic mode (``deterministic_requested``) and the fallback without the
#: tensor-core backends:
#: half-fused FMA, two columns per CTA, no hidden split, 256-thread BigFuse. It has no cross-CTA
#: floating-point atomics and one reduction order for every row count.
DETERMINISTIC_TACTIC = ("fused_half_fma", 2, 1, 256, 1)

#: The autotuner's key for this op's tactic choice.
TUNING_OP = "glm5_next::mhc_fused_hc"

#: A tactic: ``(backend, tile_n, num_k_splits, bigfuse_block_size, tile_m)``.
Tactic = tuple[str, int, int, int, int]

#: The op's arguments besides the tensors the autotuner sees: the RMSNorm weight and the scalars.
_RunnerArg = torch.Tensor | int | float | None

# (tile_n, num_k_splits) pairs of the half-fused FMA kernel (C++ pickFhcFma).
_HALF_FMA_TN_KS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (1, 8),
    (2, 1),
    (2, 2),
    (2, 4),
    (2, 8),
    (3, 1),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 2),
    (6, 1),
    (8, 1),
    (12, 1),
    (24, 1),
)
# (tile_n, num_k_splits, tile_m) triples of the all-in-one FMA kernel (C++ pickFhcFmaAllInOne).
_ALL_FMA_TN_KS_TM = tuple(
    (tn, ks, tm)
    for tm in (1, 2, 4)
    for tn, ks in ((1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1), (6, 1), (8, 1), (12, 1), (24, 1))
)
# Split counts the TF32 kernels are instantiated for; a hidden size takes those that divide its
# 64-wide tiles evenly.
_MMA_KS = (1, 2, 4, 5, 7, 8, 10, 14, 16, 20, 28, 32, 40, 56, 64, 80, 112)
_MMA_HIDDEN_SIZES = (4096, 5120, 7168)
_MMA_BLOCK_M = 64
_BIGFUSE_BLOCK_SIZES = (128, 256, 512)
# The candidate filters cap a grid in units of B200's 148 SMs, on every SKU.
_GRID_UNIT = 148


@lru_cache(maxsize=1)
def fused_hc_mma_enabled() -> bool:
    """Whether the tensor-core backends run here: an SM 10.x GPU and a build with DeepGEMM."""
    return 100 <= get_sm_version() < 110 and bool(torch.ops.trtllm.mhc_fused_hc_mma_enabled())


def _mma_split_options(hidden: int) -> tuple[int, ...]:
    if hidden not in _MMA_HIDDEN_SIZES:
        return ()
    tiles = hidden // 64
    return tuple(ks for ks in _MMA_KS if ks <= tiles and tiles % ks == 0)


def _mma_split_ladder(rows: int) -> int:
    """The half-fused TF32 kernel's default split count (C++ ``mhcFusedHcMmaKSplits``)."""
    for max_rows, splits in ((512, 16), (1024, 8), (2048, 4), (4096, 2)):
        if rows <= max_rows:
            return splits
    return 1


def _mma_bigfuse_block_size(rows: int) -> int:
    """The half-fused TF32 kernel's default BigFuse CTA size (C++ ``selectBigFuseBS``)."""
    if rows >= 4096:
        return 128
    return 256 if rows >= 256 else 512


def deterministic_requested() -> bool:
    """Whether the environment asks for run-to-run identical mHC results, as ``mHC.fused_hc``
    reads it: ``FORCE_DETERMINISTIC=1`` or ``FORCE_MHC_KERNEL_DETERMINISTIC=1``."""
    return (
        os.environ.get("FORCE_DETERMINISTIC", "0") == "1"
        or os.environ.get("FORCE_MHC_KERNEL_DETERMINISTIC", "0") == "1"
    )


def default_fused_hc_tactic(rows: int, hidden: int, mma_enabled: bool) -> Tactic:
    """The tactic of a call the autotuner has no entry for: ``mHC.fused_hc``'s fallback, resolved.

    The half-fused TF32 kernel at its own default split count and BigFuse CTA size where the
    tensor-core backends run, ``DETERMINISTIC_TACTIC`` elsewhere.
    """
    if mma_enabled and hidden in _MMA_HIDDEN_SIZES:
        return ("fused_half_mma", 0, _mma_split_ladder(rows), _mma_bigfuse_block_size(rows), 1)
    return DETERMINISTIC_TACTIC


def fused_hc_tactics(
    rows: int, hidden: int, mma_enabled: bool, pin_small_m_half_fma: bool
) -> list[Tactic]:
    """The tactics ``mHC.fused_hc``'s autotuner profiles for ``rows`` rows (a tuning bucket).

    Up to 32 rows: the all-in-one FMA kernel at every tile shape whose grid stays within four
    waves of 148 CTAs, and the half-fused FMA kernel at every tile width, split count and BigFuse
    CTA size (split ones only while ``rows * 24 / tile_n`` is under 296). From 32 rows, with
    tensor cores: the half-fused TF32 kernel at the largest split count of at most
    ``128 / ceil(rows / 64)`` and its measured BigFuse CTA sizes, and from 64 rows the all-in-one
    TF32 kernel at that split count. Above 32 rows without them: the half-fused FMA kernel unsplit.

    ``pin_small_m_half_fma`` (SM 10.3) keeps the small-row half-fused FMA candidates to one split
    count (4 up to 4 rows, else 2) and one BigFuse CTA size (512), so that those candidates differ
    only in tile width, which assigns columns to CTAs without changing any sum.
    """
    tactics: dict[Tactic, None] = {}
    mma_splits = _mma_split_options(hidden)
    mma_ok = mma_enabled and bool(mma_splits)
    if rows <= 32:
        for tn, ks, tm in _ALL_FMA_TN_KS_TM:
            if -(-rows // tm) * (MIX // tn) * ks <= _GRID_UNIT * 4:
                tactics[("fused_all_fma", tn, ks, 0, tm)] = None
        pinned_ks = (4 if rows <= 4 else 2) if pin_small_m_half_fma else None
        block_sizes = (512,) if pin_small_m_half_fma else _BIGFUSE_BLOCK_SIZES
        for tn, ks in _HALF_FMA_TN_KS:
            if pinned_ks is not None and ks != pinned_ks:
                continue
            if ks > 1 and rows * (MIX // tn) >= _GRID_UNIT * 2:
                continue
            for bs in block_sizes:
                tactics[("fused_half_fma", tn, ks, bs, 1)] = None
    if mma_ok and rows >= 32:
        m_tiles = -(-rows // _MMA_BLOCK_M)
        fitting = [ks for ks in mma_splits if ks <= max(1, 128 // m_tiles)]
        ks = fitting[-1] if fitting else mma_splits[0]
        if m_tiles * ks <= _GRID_UNIT * 4:
            if rows < 512:
                block_sizes = (512,)
            elif rows == 1024:
                block_sizes = (128, 512)
            elif rows >= 8192:
                block_sizes = (128, 256)
            else:
                block_sizes = (256,)
            for bs in block_sizes:
                tactics[("fused_half_mma", 0, ks, bs, 1)] = None
            if rows >= 64:
                tactics[("fused_all_mma", 0, ks, 0, 1)] = None
    if not mma_ok and rows > 32:
        bs = 512 if rows < 512 else 256
        for tn, ks in _HALF_FMA_TN_KS:
            if ks == 1:
                tactics[("fused_half_fma", tn, ks, bs, 1)] = None
    return list(tactics)


def _splits(rows: int, tactic: Sequence) -> int:
    backend, _, num_k_splits, _, _ = tactic
    if backend == "fused_half_mma":
        return num_k_splits if num_k_splits > 0 else _mma_split_ladder(rows)
    if backend in ("fused_all_mma", "fused_half_fma"):
        return max(num_k_splits, 1)
    return 1  # fused_all_fma adds its at most two splits onto one slot


def fused_hc_scratch_shapes(
    rows: int, tactic: Sequence
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Shapes of ``(y_acc_ws, r_acc_ws, done_counter_ws)`` for ``rows`` rows under ``tactic``.

    One FP32 ``[M, 24]`` mixer partial and ``[M]`` sum-of-squares partial per hidden split
    (``[splits, M, 24]`` and ``[splits, M]`` when split; the all-in-one FMA kernel always keeps one
    slot), and int32 done counters: one per 64 rows for ``fused_all_mma``, one per ``tile_m`` rows
    for ``fused_all_fma``, one unused counter for the half-fused backends.
    """
    backend, _, _, _, tile_m = tactic
    splits = _splits(rows, tactic)
    if backend == "fused_all_mma":
        counters = -(-rows // _MMA_BLOCK_M)
    elif backend == "fused_all_fma":
        counters = -(-rows // max(1, tile_m))
    else:
        counters = 1
    if splits > 1:
        return (splits, rows, MIX), (splits, rows), (counters,)
    return (rows, MIX), (rows,), (counters,)


def fused_hc_scratch(
    rows: int, tactic: Sequence, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """New ``(y_acc_ws, r_acc_ws, done_counter_ws)`` for ``rows`` rows under ``tactic``.

    Uninitialized: the kernels write every slot they read, and the all-in-one backends zero what
    they accumulate into.
    """
    y_shape, r_shape, done_shape = fused_hc_scratch_shapes(rows, tactic)
    return (
        torch.empty(y_shape, dtype=torch.float32, device=device),
        torch.empty(r_shape, dtype=torch.float32, device=device),
        torch.empty(done_shape, dtype=torch.int32, device=device),
    )


def _pins_small_m_half_fma(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) == (10, 3)


def _tuning_buckets(rows: int) -> tuple[int, ...]:
    buckets = (1, 2, 4, 8, 16, 32, 64, 128)
    if rows >= 128:
        buckets += tuple(range(256, max(1024, min(rows, 8192)) + 1, 128))
    return buckets


def _tuning_bucket(rows: int) -> int:
    if rows <= 128:
        bucket = 1
        while bucket < rows:
            bucket *= 2
        return bucket
    return -(-rows // 128) * 128


def _rows_of_x(shapes: Sequence[Sequence[int]]) -> int:
    return shapes[0][0]


class _FusedHcRunner(TunableRunner):
    """The autotuner's view of ``mhc_fused_hc``: the candidates of a row bucket, and one call each.

    The inputs are ``[x, residual, post, comb, fn, scale, base]``; the rest of the op's arguments
    arrive as keyword arguments. Rows are bucketed by powers of two up to 128, then in steps of 128.

    The TP ranks share the profiling (``PARALLEL``, as ``mHC.fused_hc``'s runner does): each rank
    times its share of the candidates and every rank keeps the fastest of all, so every rank runs
    the same tactic. In decode every rank holds the same rows, and a tactic sums in its own order,
    so ranks on different tactics would carry streams that differ in the last bits and could route
    a row to different experts. The ranks call the autotuner in lockstep: the target pads a
    sequence-parallel step's rows to the same count on every rank.
    """

    tuning_config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=0,
                dim_idx=0,
                gen_tuning_buckets=_tuning_buckets,
                map_to_tuning_buckets=_tuning_bucket,
            ),
        ),
        constraint_specs=tuple(
            ConstraintSpec(input_idx=i, dim_idx=0, infer_shape=_rows_of_x) for i in (1, 2, 3)
        ),
        distributed_tuning_strategy=DistributedTuningStrategy.PARALLEL,
    )

    def __init__(self, hidden: int) -> None:
        self.hidden = hidden

    def unique_id(self) -> tuple[int, int]:
        return (MULT, self.hidden)

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile, **kwargs: _RunnerArg
    ) -> list[Tactic]:
        x = inputs[0]
        return fused_hc_tactics(
            x.shape[0], self.hidden, fused_hc_mma_enabled(), _pins_small_m_half_fma(x.device)
        )

    def forward(
        self, inputs: list[torch.Tensor], *, tactic: Sequence | int = -1, **kwargs: _RunnerArg
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = inputs[0]
        rows = x.shape[0]
        if tactic == -1:
            tactic = default_fused_hc_tactic(rows, self.hidden, fused_hc_mma_enabled())
        return mhc_fused_hc(
            *inputs,
            kwargs["norm_weight"],
            kwargs["sinkhorn_iters"],
            kwargs["norm_eps"],
            kwargs["hc_eps"],
            kwargs["sinkhorn_eps"],
            kwargs["post_mult"],
            kwargs["layer_input_eps"],
            tactic,
            *fused_hc_scratch(rows, tactic, x.device),
        )


_RUNNERS: dict[int, _FusedHcRunner] = {}


def fused_hc_tactic(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: Optional[torch.Tensor],
    sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    sinkhorn_eps: float,
    post_mult: float,
    layer_input_eps: float,
) -> Tactic:
    """The tactic for an ``mhc_fused_hc`` call with these arguments, chosen as ``mHC.fused_hc`` does.

    ``DETERMINISTIC_TACTIC`` when ``deterministic_requested()``. Otherwise the autotuner's choice
    under ``TUNING_OP``. In tuning mode it profiles ``fused_hc_tactics`` at every row bucket --
    powers of two up to 128 and, from a call of 128 rows, multiples of 128 up to ``max(1024,
    min(M, 8192))`` -- running the op on these weights and on bucket-sized activations, with the
    candidates shared out over the TP ranks and the fastest kept on all of them; every rank must
    make the call, with the same row count. Outside it, it looks up the call's row bucket (rows
    rounded up to a power of two, or above 128 to a multiple of 128) and returns
    ``default_fused_hc_tactic`` on a miss. The arguments are ``mhc_fused_hc``'s without the
    tactic and the workspaces.
    """
    if deterministic_requested():
        return DETERMINISTIC_TACTIC
    rows, hidden = x.shape
    runner = _RUNNERS.get(hidden)
    if runner is None:
        runner = _RUNNERS[hidden] = _FusedHcRunner(hidden)
    _, tactic = AutoTuner.get().choose_one(
        TUNING_OP,
        [runner],
        _FusedHcRunner.tuning_config,
        [x, residual, post, comb, fn, scale, base],
        norm_weight=norm_weight,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        sinkhorn_eps=sinkhorn_eps,
        post_mult=post_mult,
        layer_input_eps=layer_input_eps,
    )
    if tactic == -1:
        return default_fused_hc_tactic(rows, hidden, fused_hc_mma_enabled())
    return tuple(tactic)


class MhcFusedHc(OpWrapper):
    """The previous sublayer's post-mapping, then the next sublayer's pre-mapping and input RMSNorm.

    Args:
        x: ``[M, hidden]`` BF16, contiguous: the previous sublayer's output, after its TP reduction.
        residual: ``[M, 4, hidden]`` BF16, contiguous: the streams the previous sublayer read.
        post: ``[M, 4]`` or ``[M, 4, 1]`` FP32, contiguous: the previous sublayer's post coefficients.
        comb: ``[M, 4, 4]`` FP32, contiguous: the previous sublayer's Sinkhorn-normalized combination,
            ``[source stream, destination stream]``.
        fn: ``[24, 4 * hidden]`` FP32, contiguous: the next sublayer's mixer weights.
        scale: ``[3]`` FP32, contiguous: the next sublayer's mixer scales (pre, post, comb).
        base: ``[24]`` FP32, contiguous: the next sublayer's mixer biases.
        norm_weight: ``[hidden]`` BF16, contiguous: the next sublayer's input RMSNorm weight; or
            None for the unnormalized layer input.
        sinkhorn_iters: Sinkhorn iterations, the initial column normalization included (20).
        norm_eps: epsilon of the mixer's RMS over the flattened streams (``rms_norm_eps``).
        hc_eps: added to ``pre`` (``hc_eps``).
        sinkhorn_eps: Sinkhorn epsilon (``hc_eps``).
        post_mult: multiplier of the ``post`` sigmoid (2.0).
        layer_input_eps: epsilon of the fused input RMSNorm (``rms_norm_eps``).
        tactic: ``(backend, tile_n, num_k_splits, bigfuse_block_size, tile_m)``, e.g. from
            ``fused_hc_tactic``; see ``fused_hc_tactics`` for the candidates.
        y_acc_ws: FP32 mixer partials, ``fused_hc_scratch_shapes(M, tactic)[0]`` or larger.
        r_acc_ws: FP32 sum-of-squares partials, ``fused_hc_scratch_shapes(M, tactic)[1]`` or larger.
        done_counter_ws: int32 counters, ``fused_hc_scratch_shapes(M, tactic)[2]`` or larger.

    Returns:
        ``(residual_out, post_out, comb_out, layer_input)``: the new streams (BF16, ``residual``'s
        shape), the next sublayer's ``post`` ``[M, 4]`` and ``comb`` ``[M, 4, 4]`` (FP32), and its
        layer input ``[M, hidden]`` (BF16, RMS-normalized when ``norm_weight`` is given) -- what
        ``mHC.fused_hc`` returns, ``post`` without its trailing unit dimension. All new tensors.
    """

    ARCHS = frozenset({Arch.SM_100})

    _GLM = dict(mult=MULT, hidden=HIDDEN, **GLM_SCALARS)
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode of one request without MTP: FMA candidates only, one CTA row",
            spec=dict(rows=1, norm=True, **_GLM),
        ),
        Cell(why="MTP-3 verify of one request (4 rows)", spec=dict(rows=4, norm=True, **_GLM)),
        Cell(why="MTP-3 verify of two requests (8 rows)", spec=dict(rows=8, norm=True, **_GLM)),
        Cell(
            why="MTP-3 verify of 8 requests (32 rows): both FMA and TF32 candidates",
            spec=dict(rows=32, norm=True, **_GLM),
        ),
        Cell(
            why="MTP-3 verify of 64 requests (256 rows): TF32 candidates at 32 hidden splits",
            spec=dict(rows=256, norm=True, **_GLM),
        ),
        Cell(
            why="a quarter of a 13,312-token prompt (sequence parallel): TF32 at 2 hidden splits",
            spec=dict(rows=3328, norm=True, **_GLM),
        ),
        Cell(
            why="MTP-3 verify of one request without the fused norm: the unnormalized variants",
            spec=dict(rows=4, norm=False, **_GLM),
        ),
    )

    note = """
    The cells are driven with every tactic `fused_hc_tactic` can return at their row count: the
    autotuner's candidates, the cache-miss fallback and `DETERMINISTIC_TACTIC`. The tensor-core
    backends need an SM 10.x GPU, a build with DeepGEMM (`fused_hc_mma_enabled`) and hidden 4096,
    5120 or 7168; the FMA backends take any hidden divisible by 64. The kernels are compiled for 4
    streams. The lagged pre-mix variant (`pre_mix_ext` / `pre_mix_out`) is not used.

    `fused_hc_tactic` tunes as `mHC.fused_hc` does: the TP ranks share the profiling and keep one
    tactic, so their streams stay bitwise equal. `FORCE_DETERMINISTIC=1` or
    `FORCE_MHC_KERNEL_DETERMINISTIC=1` puts every rank on `DETERMINISTIC_TACTIC`, whose sums do not
    depend on the row count either. The candidates of a row count do not depend on the weights or
    the scalars, and a step's boundaries share their shapes, so a tactic can be chosen once per
    step.

    The op checks the split workspaces of the first three backends against the split count it
    runs, and neither the all-in-one FMA kernel's workspaces nor any counter array; `is_valid`
    covers those. Nothing needs zeroing beforehand, so successive calls on one stream can share
    the workspaces; concurrent calls cannot. The FMA kernels and BigFuse launch with programmatic
    dependent launch when `TRTLLM_ENABLE_PDL` allows it. A call of no rows launches nothing.
    Outputs are allocated per call, from the graph's private pool under CUDA graph capture; the
    tensor-core backends cache their TMA descriptors by address and shape.
    """

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        tactic: Sequence,
        y_acc_ws: torch.Tensor,
        r_acc_ws: torch.Tensor,
        done_counter_ws: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, hidden = x.shape
        backend, tile_n, num_k_splits, bigfuse_block_size, tile_m = tactic
        residual_out = torch.empty_like(residual, memory_format=torch.contiguous_format)
        post_out = torch.empty((rows, MULT), dtype=torch.float32, device=x.device)
        comb_out = torch.empty((rows, MULT, MULT), dtype=torch.float32, device=x.device)
        layer_input = torch.empty((rows, hidden), dtype=torch.bfloat16, device=x.device)
        torch.ops.trtllm.mhc_fused_hc(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            residual_out,
            post_out,
            comb_out,
            layer_input,
            y_acc_ws,
            r_acc_ws,
            done_counter_ws,
            rows,
            hidden,
            MULT,
            norm_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            sinkhorn_iters,
            BACKENDS[backend],
            tile_n,
            num_k_splits,
            bigfuse_block_size,
            tile_m,
            norm_weight,
            layer_input_eps,
            None,
            None,
        )
        return residual_out, post_out, comb_out, layer_input

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        tactic: Sequence,
        y_acc_ws: torch.Tensor,
        r_acc_ws: torch.Tensor,
        done_counter_ws: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """The unfused boundary from the three entries' references; the tactic and workspaces are
        ignored.

        ``mhc_post_mapping``'s streams, rounded to BF16 as the next sublayer reads them, then
        ``mhc_gemm_sqrsum`` (a true-FP32 mixer product) and ``mhc_big_fuse`` on those streams.
        """
        rows = x.shape[0]
        streams = mhc_post_mapping.reference(x, residual, post, comb)
        mixes, sqrsum = mhc_gemm_sqrsum.reference(streams.reshape(rows, -1), fn, 0)
        post_out, comb_out, layer_input = mhc_big_fuse.reference(
            mixes,
            sqrsum,
            streams.reshape(rows, MULT, -1),
            scale,
            base,
            norm_weight,
            sinkhorn_iters,
            norm_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            layer_input_eps,
            0,
        )
        return streams, post_out, comb_out, layer_input

    def is_valid(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        fn: torch.Tensor,
        scale: torch.Tensor,
        base: torch.Tensor,
        norm_weight: Optional[torch.Tensor],
        sinkhorn_iters: int,
        norm_eps: float,
        hc_eps: float,
        sinkhorn_eps: float,
        post_mult: float,
        layer_input_eps: float,
        tactic: Sequence,
        y_acc_ws: torch.Tensor,
        r_acc_ws: torch.Tensor,
        done_counter_ws: torch.Tensor,
    ) -> None:
        assert x.dim() == 2, f"x is [M, hidden]; got {tuple(x.shape)}"
        rows, hidden = x.shape
        assert residual.numel() == rows * MULT * hidden and residual.shape[-1] == hidden, (
            f"residual must hold [{rows}, {MULT}, {hidden}]; got {tuple(residual.shape)}"
        )
        assert post.numel() == rows * MULT and comb.numel() == rows * MULT * MULT, (
            f"post holds [{rows}, {MULT}] and comb [{rows}, {MULT}, {MULT}]"
        )
        assert fn.shape == (MIX, MULT * hidden), (
            f"fn is [{MIX}, {MULT * hidden}]; got {tuple(fn.shape)}"
        )
        assert scale.numel() >= 3 and base.numel() >= MIX, "scale holds 3 values and base 24"
        for name, t in (
            ("x", x),
            ("residual", residual),
            ("post", post),
            ("comb", comb),
            ("fn", fn),
            ("scale", scale),
            ("base", base),
            ("y_acc_ws", y_acc_ws),
            ("r_acc_ws", r_acc_ws),
            ("done_counter_ws", done_counter_ws),
        ):
            assert t.is_contiguous(), f"{name} is read with dense row-major strides"
        y_shape, r_shape, done_shape = fused_hc_scratch_shapes(rows, tactic)
        y_need, r_need = math.prod(y_shape), math.prod(r_shape)
        assert y_acc_ws.numel() >= y_need and r_acc_ws.numel() >= r_need, (
            f"{tuple(tactic)} at {rows} rows writes {y_need} mixer and {r_need} sum-of-squares "
            f"partials; got {y_acc_ws.numel()} and {r_acc_ws.numel()}"
        )
        assert done_counter_ws.numel() >= done_shape[0], (
            f"{tuple(tactic)} at {rows} rows counts in {done_shape[0]} slots; "
            f"got {done_counter_ws.numel()}"
        )


mhc_fused_hc = MhcFusedHc()
