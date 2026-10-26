# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``wide_projection_gathers``: decode steps of 9..32 rows (two to four requests at
K = 7) on the 8-row step's projection chain, sm103 TEP4.

At 1..8 rows the built-in MoE computes the pool (8192 -> 4096) and unpool (4096 -> 8192)
projections as column slices per rank and ``projection_gathers`` folds their gathers into
the consumers: the unpool slice into the post-FFN boundary
(``ops/comm/unpool_gather_boundary``), the pool slice into the pre-expert RMSNorm + NVFP4
quantization (``ops/comm/pool_gather_fp4quant``). From 9 rows the built-in multiplies the
whole replicated weights on every rank (a 67 MB read per projection per rank at TP4,
against 16.8 MB for a slice) and the next layer's QKV runs the built-in low-M dispatch's
per-rank pick after the boundary has finished.

``install`` builds this path's two wide inboxes (``ProjectionInbox`` with ``max_rows`` 32,
one per site, separate from the 8-row inboxes, whose calls and kernels stay as they are),
hands them and one ``WideProjections`` to every layer's re-classed MoE and post-FFN join
(``projection_gathers``' classes read them), and gives the dense GEMM plan a 9..32-row
table (``TP4_WIDE_KERNELS``: flashinfer's split-K for the QKV projection and the two
slices). ``DecodeTarget`` arms the ``WideProjections`` for the forward of a decode step of
9..32 rows (``takes``). While armed:

* the pool and unpool projections return the rank's slice, multiplied by the plan's
  split-K kernel (the call the 8-row path makes, at these rows);
* the post-FFN join gathers the unpool slice inside its boundary kernel on the wide unpool
  inbox -- at TP4 and these rows the boundary runs clusters of four CTAs of 256 threads, so
  each CTA owns exactly one rank's 2048 columns, as each CTA owns part of one rank's slice
  at 1..8 rows -- with the trigger ``boundary_early_release`` gave the layer (``entry`` on
  the converted edges): the next QKV, now flashinfer's split-K with its weight loads
  before its dependency wait, streams its weight while the boundary waits on its peers;
* the pre-expert quantization gathers the pool slice on the wide pool inbox (on the MoE's
  preparation stream, where the built-in runs it);
* ``capture_fold``'s capture layers store the DFlash capture from that boundary, as at 1..8.

Unarmed calls -- prefill and mixed steps, decode steps of 1..8 or more than 32 rows -- are
the classes behind, unchanged. The choice reads the step's row count, shapes, dtypes and
contiguity only, so every rank takes the same path at every call.

Numerics: the gathers and their consumers are the built-in kernels' bits on the gathered
rows (certified at 16, 24 and 32 rows against the built-in gather workspace followed by the
built-in cluster boundary / flashinfer's ``rmsnorm_fp4quant``). The three GEMMs are another
kernel than the built-in's at these rows (FP32 accumulation, rounded once): their outputs
differ from the built-in's by the summation order only, held to ``dense_gemms``' error
budget against the kernels the built-in runs (``test_modeling_v2_dense_gemm.py``).
"""

from __future__ import annotations

import contextlib
from typing import Dict, Iterator, Mapping, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

from ...ops.comm import pool_gather_fp4quant as _pool
from ...ops.comm import unpool_gather_boundary as _unpool
from ...ops.comm.projection_inbox import MAX_ROWS, WIDE_ROWS, ProjectionInbox
from ...ops.gemm.dense_gemm import kernel_error
from ..dense_gemms.dispatcher import WIDE_MIN_ROWS, DenseGemmPlan
from ..projection_gathers.gathers import MuseSparkOpenTepGatedResidual, MuseSparkOpenTepGatherMoE

#: The rows of the steps this path arms: two to four requests at K = 7 (and the row counts
#: between, which no K = 7 decode step has but an eager step may).
MIN_ROWS, MAX_ROWS_WIDE = MAX_ROWS + 1, WIDE_ROWS

#: Kernel per site by row count, 9..32, as ``(last row count, kernel)`` steps, sm103 at TP4
#: (per-rank shapes: QKV ``[4608, 8192]``, pooler slice ``[1024, 8192]``, unpooler slice
#: ``[2048, 4096]``), each the fastest at 16, 24 and 32 rows of flashinfer's split-K
#: (``mma_m`` 64/128 x ``mma_n`` 16/32 x ``split_k`` 1/2/4 at the deepest pipeline) and
#: cuBLAS, behind a predecessor releasing at exit and, for split-K, at entry (exposed time
#: per call, weights read from HBM; ``test_microbenchmark_wide_rows``): released at entry,
#: the split-K QKV is faster than cuBLAS at each of those row counts, and each split-K slice
#: is faster than cuBLAS multiplying the whole replicated weight. flashinfer's split-K
#: throughout: it loads the weight before its dependency wait, which the early-released QKV
#: needs. The attention output projection keeps the built-in dispatch: behind the built-in
#: output gate, which releases at exit at these rows, split-K is no faster than cuBLAS.
TP4_WIDE_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((16, "splitk:64,16,2,11"), (WIDE_ROWS, "splitk:64,32,2,9")),
    "pool": ((WIDE_ROWS, "splitk:64,16,4,10"),),
    "unpool": ((16, "splitk:64,16,4,10"), (WIDE_ROWS, "splitk:64,16,2,11")),
}


def wide_kernel_steps(
    capability: Optional[Tuple[int, int]],
) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    """The 9..32-row table of the device of ``capability`` (None: the target's own sm103,
    as ``dense_gemms.device_capability`` reads a host without CUDA).

    Raises:
        ValueError: on sm107, whose TEP4 composition serves its 9..32-row projections on
            ``wide_dense_gemms``' R200 table instead (``dense_gemms`` ``SM107_TP4_WIDE_KERNELS``),
            and on any other device.
    """
    capability = (10, 3) if capability is None else tuple(capability)
    if capability == (10, 3):
        return TP4_WIDE_KERNELS
    elif capability == (10, 7):
        raise ValueError(
            "[musespark_open] wide_projection_gathers has no sm107 table: sm107 TEP4 runs its "
            "9..32-row projections on wide_dense_gemms"
        )
    else:
        raise ValueError(
            f"[musespark_open] wide_projection_gathers has no table for SM {capability} "
            "(measured on sm103 only)"
        )


#: The per-rank ``(K, N)`` of each site at TP4.
TP4_SHAPES: Mapping[str, Tuple[int, int]] = {
    "qkv": (8192, 4608),
    "pool": (8192, 1024),
    "unpool": (4096, 2048),
}


def wide_kernels(
    steps: Mapping[str, Tuple[Tuple[int, str], ...]] = TP4_WIDE_KERNELS,
) -> Dict[str, Tuple[str, ...]]:
    """``steps`` expanded to one kernel per row count 9..32 (``kernels[site][rows - 9]``)."""
    kernels = {}
    for site, table in steps.items():
        by_rows, first = [], MIN_ROWS
        for last, kernel in table:
            by_rows += [kernel] * (last - first + 1)
            first = last + 1
        if len(by_rows) != MAX_ROWS_WIDE - MIN_ROWS + 1:
            raise ValueError(
                f"{site}: the steps {table} do not cover rows {MIN_ROWS}..{MAX_ROWS_WIDE}"
            )
        kernels[site] = tuple(by_rows)
    return kernels


class WideProjections:
    """The decode steps of 9..32 rows on the wide projection chain: armed per forward by
    ``DecodeTarget``.

    It also logs, once per row count and site, whether an armed call took the fused path
    or fell back to the built-in gather (and so the built-in consumer), so a served log
    shows what the captured decode steps run.
    """

    def __init__(self, plan: DenseGemmPlan) -> None:
        self.plan = plan
        self._armed = False
        self._reported = set()

    @staticmethod
    def takes(rows: int) -> bool:
        """Whether a decode step of ``rows`` rows runs on the wide chain."""
        return MIN_ROWS <= rows <= MAX_ROWS_WIDE

    @property
    def is_armed(self) -> bool:
        return self._armed

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Arm the layers' projections, joins and the dense GEMM plan for one decode forward."""
        self._armed = True
        self.plan.wide_armed = True
        try:
            yield
        finally:
            self._armed = False
            self.plan.wide_armed = False

    def report(self, site: str, rows: int, fused: bool) -> None:
        """Log the first armed call of ``site`` at ``rows`` rows: fused, or the built-in gather.

        Both are expected in a served step: the pool site of a layer whose experts are not
        NVFP4 (the built-in quantizes nothing there, so nothing can take the gather) gathers
        with the built-in workspace, as at 1..8 rows.
        """
        if (site, rows, fused) in self._reported:
            return
        self._reported.add((site, rows, fused))
        if fused:
            logger.info(
                f"[musespark_open] wide_projection_gathers: a {rows}-row decode step gathers the "
                f"{site} slice in its consumer"
            )
        else:
            logger.info(
                f"[musespark_open] wide_projection_gathers: an armed {rows}-row {site} call gathers "
                "with the built-in workspace and runs the built-in consumer"
            )


def check(model: torch.nn.Module, plan: Optional[DenseGemmPlan]) -> None:
    """Whether ``install`` may run on ``model`` with ``dense_gemms``' ``plan``; changes nothing.

    Raises:
        ValueError: if the plan's device has no 9..32-row table (``wide_kernel_steps``: sm103
            only), the model is not TP4, a layer's MoE or post-FFN join is not
            ``projection_gathers``' class with both sites fused, ``dense_gemms`` is not
            installed, or its plan prefetches (``dense_gemm_prefetch``): that prefetch's
            row exemptions (``skip_prefetch_rows``) are derived from the 1..8-row table, so
            at 9..32 rows it would fork an L2 prefetch of the next QKV weight whose join
            gates the early-released split-K QKV.
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    if mapping.tp_size != 4:
        raise ValueError(
            f"[musespark_open] wide_projection_gathers serves TP4 (one tray), got TP{mapping.tp_size}"
        )
    if plan is None:
        raise ValueError(
            "[musespark_open] wide_projection_gathers runs its GEMMs on dense_gemms' plan, which "
            "is not installed"
        )
    if plan.prefetch:
        raise ValueError(
            "[musespark_open] wide_projection_gathers does not compose with dense_gemm_prefetch"
        )
    for layer in layers:
        moe, join = layer.feed_forward, layer.add_ffn
        if not isinstance(moe, MuseSparkOpenTepGatherMoE) or moe._pool_inbox is None:
            raise ValueError(
                f"[musespark_open] wide_projection_gathers widens projection_gathers' pool site, which "
                f"layer {layer.layer_idx} does not run ({type(moe).__name__})"
            )
        if not isinstance(join, MuseSparkOpenTepGatedResidual) or join._gather_inbox is None:
            raise ValueError(
                f"[musespark_open] wide_projection_gathers widens projection_gathers' unpool site, "
                f"which layer {layer.layer_idx} does not run ({type(join).__name__})"
            )
    for site, by_rows in wide_kernels(wide_kernel_steps(plan.capability)).items():
        k, n = TP4_SHAPES[site]
        for rows, kernel in enumerate(by_rows, start=WIDE_MIN_ROWS):
            error = kernel_error(kernel, rows, k, n)
            if error is not None:
                raise ValueError(
                    f"[musespark_open] wide_projection_gathers: {site} kernel {kernel} cannot serve "
                    f"{rows} x {(k, n)}: {error}"
                )


def install(model: torch.nn.Module, plan: DenseGemmPlan) -> WideProjections:
    """Build the wide inboxes and hand them, the plan's wide table and the arming to every layer.

    Collective over the TP group (the inboxes are fabric memory); call ``check`` first,
    agreed over the group.
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    world = mapping.tp_size
    unpool_inbox = ProjectionInbox(
        mapping, row_words=_unpool.local_columns(world) // 2, max_rows=WIDE_ROWS
    )
    pool_inbox = ProjectionInbox(
        mapping, row_words=_pool.local_columns(world) // 2, max_rows=WIDE_ROWS
    )
    kernels = wide_kernels(wide_kernel_steps(plan.capability))
    plan.set_wide_kernels(kernels)
    wide = WideProjections(plan)
    for layer in layers:
        moe, join = layer.feed_forward, layer.add_ffn
        moe._wide = wide
        moe._wide_pool_inbox = pool_inbox
        join._wide = wide
        join._wide_gather_inbox = unpool_inbox
    logger.info(
        f"[musespark_open] wide_projection_gathers: decode steps of {MIN_ROWS}..{MAX_ROWS_WIDE} rows "
        f"in {len(layers)} layers: pool/unpool slices with their gathers fused into the "
        "pre-expert quantization and the post-FFN boundary (32-row inboxes); split-K for "
        + ", ".join(
            f"{site} {'|'.join(dict.fromkeys(by_rows))}" for site, by_rows in kernels.items()
        )
    )
    return wide
