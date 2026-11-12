# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast path ``wide_projection_gathers``: decode steps of 9..32 rows (two to four requests at
K = 7) on the 8-row step's projection chain, sm103 TEP4 and TEP8.

At 1..8 rows the built-in MoE computes the pool (8192 -> 4096) and unpool (4096 -> 8192)
projections as column slices per rank and ``projection_gathers`` folds their gathers into
the consumers: the unpool slice into the post-FFN boundary
(``ops/comm/unpool_gather_boundary``), the pool slice into the pre-expert RMSNorm + NVFP4
quantization (``ops/comm/pool_gather_fp4quant``). From 9 rows the built-in multiplies the
whole replicated weights on every rank (a 67 MB read per projection per rank, against
16.8 MB for a TP4 slice and 8.4 MB for a TP8 one) and the next layer's QKV runs after the
built-in boundary has finished.

``install`` builds this path's two wide inboxes (``ProjectionInbox`` with ``max_rows`` 32,
one per site, separate from the 8-row inboxes, whose calls and kernels stay as they are),
hands them and one ``WideProjections`` to every layer's re-classed MoE and post-FFN join
(``projection_gathers``' classes read them), and gives the dense GEMM plan the 9..32-row
table of the device and TP size (``WIDE_KERNELS_BY_SM_TP``: flashinfer's split-K for the
two slices, and at TP4 for the QKV projection; at TP8 ``wide_dense_gemms`` owns the QKV
cells, which ``DenseGemmPlan.kernel_for`` reads first). ``DecodeTarget`` arms the
``WideProjections`` for the forward of a decode step of 9..32 rows (``takes``). While armed:

* the pool and unpool projections return the rank's slice, multiplied by the plan's
  split-K kernel (the call the 8-row path makes, at these rows);
* the post-FFN join gathers the unpool slice inside its boundary kernel on the wide unpool
  inbox -- at these rows the boundary runs clusters of four CTAs of 256 threads (2048
  columns each): at TP4 a CTA owns exactly one rank's slice, at TP8 two whole 1024-column
  slices, one per half of its warps, so it waits on two peers -- with the trigger
  ``boundary_early_release`` gave the layer (``entry`` on the converted edges): the next
  QKV, flashinfer's split-K with its weight loads before its dependency wait, streams its
  weight while the boundary waits on its peers;
* the pre-expert quantization gathers the pool slice on the wide pool inbox (on the MoE's
  preparation stream, where the built-in runs it);
* ``capture_fold``'s capture layers store the DFlash capture from that boundary, as at 1..8.

With ``dense_gemm_prefetch`` (served at TP8, not at TP4) the unpool slice's dispatcher would
also fork the L2 prefetch of the next QKV weight at these rows, and that fork's join would gate
the early-released QKV. ``boundary_early_release`` drops it per site, at the
``drop_converted_prefetches`` call (``wide``), on every converted edge at the 9..32-row counts
whose QKV cell loads its weight before its wait; ``check`` refuses a plan where a converted
layer's unpooler still forks there (``prefetch_conflicts``). The 1..8-row exemptions and
``next_qkv_prefetch``'s fork rows stay the 1..8-row table's.

Unarmed calls -- prefill and mixed steps, decode steps of 1..8 or more than 32 rows -- are
the classes behind, unchanged. The choice reads the step's row count, shapes, dtypes and
contiguity only, so every rank takes the same path at every call.

Numerics: the gathers and their consumers are the built-in kernels' bits on the gathered
rows (certified at 16, 24 and 32 rows against the built-in gather workspace followed by the
built-in cluster boundary / flashinfer's ``rmsnorm_fp4quant``; at TP8 over eight ranks on two
trays, CUDA-graph capture included). The
GEMMs are another kernel than the built-in's at these rows (FP32 accumulation, rounded
once): their outputs differ from the built-in's by the summation order only, held to
``dense_gemms``' error budget against the kernels the built-in runs
(``test_modeling_v2_dense_gemm.py``). At TP8 the slice cells keep the ``split_k`` 4 of the
8-row cells (``DEFAULT_KERNELS``), so every row of a 9..32-row step gets the slice bits it
gets in a one-request step, which the 1..8-row path already serves.
"""

from __future__ import annotations

import contextlib
from typing import Dict, FrozenSet, Iterator, List, Mapping, Optional, Tuple

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


#: The same at TP8 (two trays), per-rank slices pooler ``[512, 8192]`` and unpooler
#: ``[1024, 4096]``: flashinfer's split-K with the ``split_k`` 4 of the slices' 8-row cells
#: (``dense_gemms.DEFAULT_KERNELS``, ``splitk:64,8,4,12``), so every row of a wide call gets the
#: bits it gets in a one-request step. ``mma_m`` 64 (8 / 16 column tiles, 32 / 64 CTAs in
#: clusters of 4: the pooler's grid stays small beside the router on the other stream);
#: ``mma_n`` 16 / 32 holds the rows of two / three or four requests in one tile, at the deepest
#: pipeline the shared memory takes at ``split_k`` 4 (10 / 8 stages, as for ``wide_dense_gemms``'
#: TP8 QKV cells). No QKV row: at TP8 ``wide_dense_gemms`` serves QKV at 9..32 rows
#: (``dense_gemms.SM103_TP8_WIDE_KERNELS``), and ``kernel_for`` would read its cell first anyway.
#: Chosen from the shapes, not from a per-call sweep.
TP8_WIDE_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "pool": ((16, "splitk:64,16,4,10"), (WIDE_ROWS, "splitk:64,32,4,8")),
    "unpool": ((16, "splitk:64,16,4,10"), (WIDE_ROWS, "splitk:64,32,4,8")),
}

#: The 9..32-row table of each ``(compute capability, TP size)`` this path serves.
WIDE_KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {
    ((10, 3), 4): TP4_WIDE_KERNELS,
    ((10, 3), 8): TP8_WIDE_KERNELS,
}


def wide_kernel_steps(
    capability: Optional[Tuple[int, int]], tp_size: int = 4
) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    """The 9..32-row table of the device of ``capability`` (None: the target's own sm103,
    as ``dense_gemms.device_capability`` reads a host without CUDA) at ``tp_size``
    (``WIDE_KERNELS_BY_SM_TP``).

    Raises:
        ValueError: on any other device or TP size.
    """
    capability = (10, 3) if capability is None else tuple(capability)
    steps = WIDE_KERNELS_BY_SM_TP.get((capability, int(tp_size)))
    if steps is not None:
        return steps
    raise ValueError(
        f"[musespark_open] wide_projection_gathers has no table for SM {capability} at "
        f"TP{tp_size} (measured on sm103 at TP4 and TP8)"
    )


#: The per-rank ``(K, N)`` of each site at TP4.
TP4_SHAPES: Mapping[str, Tuple[int, int]] = {
    "qkv": (8192, 4608),
    "pool": (8192, 1024),
    "unpool": (4096, 2048),
}
#: The per-rank ``(K, N)`` of each site at TP8 (``dense_gemm``'s ``QKV_SHAPE`` / ``POOL_SHAPE`` /
#: ``UNPOOL_SHAPE``).
TP8_SHAPES: Mapping[str, Tuple[int, int]] = {
    "qkv": (8192, 2304),
    "pool": (8192, 512),
    "unpool": (4096, 1024),
}
SHAPES_BY_TP: Mapping[int, Mapping[str, Tuple[int, int]]] = {4: TP4_SHAPES, 8: TP8_SHAPES}


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


def prefetch_conflicts(
    model: torch.nn.Module,
    plan: DenseGemmPlan,
    kernels: Optional[Mapping[str, Tuple[str, ...]]] = None,
) -> List[Tuple[int, Tuple[int, ...]]]:
    """The layers, each with its row counts, whose unpooler would fork ``dense_gemm_prefetch``'s
    L2 prefetch of the next QKV weight on a 9..32-row step whose QKV loads that weight before its
    wait (``wide_weights_before_wait_rows``; ``kernels`` is this path's armed table), behind a
    boundary that releases that QKV early (a trigger other than ``"exit"``). The fork's join
    would make the prefetch kernel, which never triggers early, a predecessor of the QKV it was
    meant to serve. ``boundary_early_release`` drops those forks per site
    (``drop_converted_prefetches(..., wide=True)``); empty when it did, when the plan does not
    prefetch, or without programmatic dependent launch (then the QKV cannot start early and the
    prefetch is what streams its weight). Reads only the dispatchers' sites.
    """
    # Imported here: the trigger module pulls in the attention classes it re-classes.
    from ..boundary_early_release.triggers import wide_weights_before_wait_rows

    if plan is None or not plan.prefetch or not plan.enable_pdl:
        return []
    rows = wide_weights_before_wait_rows(plan, kernels)
    if not rows:
        return []
    conflicts = []
    layers = list(model.layers)
    for index, layer in enumerate(layers[:-1]):
        if getattr(getattr(layer, "add_ffn", None), "_boundary_trigger", "exit") == "exit":
            continue
        dispatcher = getattr(layer.feed_forward.unpooler, "_low_m_gemm_dispatcher", None)
        site = getattr(dispatcher, "site", None)
        if site is None or site.next_weight is None:
            continue
        missing: FrozenSet[int] = rows - frozenset(site.skip_prefetch_rows)
        if missing:
            conflicts.append((index, tuple(sorted(missing))))
    return conflicts


def check(model: torch.nn.Module, plan: Optional[DenseGemmPlan]) -> None:
    """Whether ``install`` may run on ``model`` with ``dense_gemms``' ``plan``; changes nothing.

    Raises:
        ValueError: if ``dense_gemms`` is not installed, the plan's device and the model's TP
            size have no 9..32-row table (``wide_kernel_steps``: sm103 at TP4 and TP8), a
            layer's MoE or post-FFN join is not ``projection_gathers``' class with both sites
            fused, a cell cannot serve its site's shape, or a converted layer's unpooler would
            still fork the next QKV weight's L2 prefetch at a 9..32-row count whose QKV loads
            that weight before its wait (``prefetch_conflicts``): that fork's join would gate
            the early-released split-K QKV.
    """
    layers = list(model.layers)
    mapping = layers[0].mapping
    if plan is None:
        raise ValueError(
            "[musespark_open] wide_projection_gathers runs its GEMMs on dense_gemms' plan, which "
            "is not installed"
        )
    tp_size = int(mapping.tp_size)
    steps = wide_kernel_steps(plan.capability, tp_size)
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
    kernels = wide_kernels(steps)
    for site, by_rows in kernels.items():
        k, n = SHAPES_BY_TP[tp_size][site]
        for rows, kernel in enumerate(by_rows, start=WIDE_MIN_ROWS):
            error = kernel_error(kernel, rows, k, n)
            if error is not None:
                raise ValueError(
                    f"[musespark_open] wide_projection_gathers: {site} kernel {kernel} cannot serve "
                    f"{rows} x {(k, n)}: {error}"
                )
    conflicts = prefetch_conflicts(model, plan, kernels)
    if conflicts:
        index, rows = conflicts[0]
        raise ValueError(
            "[musespark_open] wide_projection_gathers: the unpooler of "
            f"{len(conflicts)} layer(s) behind an early-released boundary (first: layer {index}) "
            f"still forks dense_gemm_prefetch's next-QKV prefetch at rows {list(rows)}, where the "
            "QKV loads its weight before its wait; boundary_early_release drops it there only "
            "with drop_converted_prefetches(..., wide=True)"
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
    kernels = wide_kernels(wide_kernel_steps(plan.capability, world))
    plan.set_wide_kernels(kernels)
    wide = WideProjections(plan)
    for layer in layers:
        moe, join = layer.feed_forward, layer.add_ffn
        moe._wide = wide
        moe._wide_pool_inbox = pool_inbox
        join._wide = wide
        join._wide_gather_inbox = unpool_inbox
    # A 9..32-row boundary CTA holds 2048 columns: one TP4 slice, or two TP8 slices.
    peers = 2048 // _unpool.local_columns(world)
    if "qkv" in plan.wide:
        qkv = "qkv by wide_dense_gemms " + "|".join(dict.fromkeys(plan.wide["qkv"]))
    elif "qkv" not in kernels:
        qkv = "qkv on the built-in dispatch"
    else:
        qkv = None
    logger.info(
        f"[musespark_open] wide_projection_gathers: TP{world} decode steps of "
        f"{MIN_ROWS}..{MAX_ROWS_WIDE} rows in {len(layers)} layers: pool/unpool slices with their "
        "gathers fused into the pre-expert quantization and the post-FFN boundary (32-row inboxes, "
        f"{peers} peer(s) per boundary CTA); split-K for "
        + ", ".join(
            f"{site} {'|'.join(dict.fromkeys(by_rows))}" for site, by_rows in kernels.items()
        )
        + (f"; {qkv}" if qkv else "")
    )
    return wide
