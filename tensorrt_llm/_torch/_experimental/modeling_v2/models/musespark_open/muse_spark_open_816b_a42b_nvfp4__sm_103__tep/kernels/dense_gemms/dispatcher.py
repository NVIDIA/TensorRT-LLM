# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``dense_gemms`` / ``dense_gemm_prefetch``: the dense projections through ``ops/gemm/dense_gemm``.

Where the built-in model chooses the GEMM: the built-in ``Linear`` (QKV and attention
output projections) and ``MuseSparkOpenMoE._replicated_projection`` (the rank's
pooler and unpooler slices) both end in ``apply_low_m_gemm(module, x, weight,
bias)``, which asks ``module._low_m_gemm_dispatcher`` -- an autotuner race whose
winner each rank keeps for itself. The engine assigns that attribute to every
``Linear`` after the weights are loaded (``prepare_low_m_gemm``), which is after
this target is built.

So ``install`` re-classes the four projections of every layer as
``MuseSparkOpenTepDenseLinear``, whose ``_low_m_gemm_dispatcher`` slot always
yields a ``DenseGemmDispatcher`` and keeps whatever the engine assigns to it as
that dispatcher's fallback. A call that fits -- BF16, no bias, at most 8 rows,
the site's shape -- runs ``dense_gemm`` with the kernel this fast path names for
the site and row count, the same on every rank; any other call goes to the built-in dispatcher
unchanged. ``wide_dense_gemms`` gives the sites its table names (``WIDE_KERNELS_BY_SM_TP``:
QKV at tep8) a kernel for the 9..32-row calls as well, on every call of those rows. A
fast path may also give the plan a table for 9..32 rows
(``DenseGemmPlan.set_wide_kernels``, ``wide_projection_gathers``): while it arms
the plan (``wide_armed``, one decode forward), a call of those rows at a site the
table names runs that table's kernel. Nothing else in the layer changes, so the other fast paths
that re-class the attention, the decoder layer or the MoE compose with this one:
their GEMM calls still land in these dispatchers.

Above 32 rows the built-in ``Linear`` never asks a low-M dispatcher, so a fast path
that acts on larger decode steps arms the plan instead (``DenseGemmPlan.batch``, set
for one decode forward by ``batch_dense_gemms``): while it is set, a re-classed
projection's ``apply_linear`` hands the call to it with the built-in GEMM as a callable.
With ``tep8_batch_splitk`` (``batch_splitk``, sm103 TEP8 only) and no such
path armed, ``apply_linear`` runs a 33..160-row call of a site ``SM103_TP8_BATCH_KERNELS``
names on ``ops/gemm/batch_splitk_gemm`` instead of the built-in cuBLAS call; the pooler and
unpooler slices of those rows run it through ``projection_publish`` (``batch_splitk_apply``).

With ``tep8_attn_block_splitk`` (sm103 TEP8 only) the
attention block's two projections of a 9..32-row decode call run on ``ops/gemm/batch_splitk_gemm``
(this target's copy of flashinfer's split-K kernel, launched with programmatic dependent launch)
at the cells ``SM103_TP8_ATTN_BLOCK_KERNELS`` names: the attention output projection at split_k 1
(one K slice, cuBLAS's summation order) instead of the built-in cuBLAS call, so its weight k-tiles
are loaded into shared memory before its ``griddepcontrol.wait``, under the attention that
launches it early (the sliding-window cluster block, the full layers' combine); and QKV at split_k
4 (flashinfer's K slices and order) with a deeper ring where the copy's reduce-scatter mailbox
leaves room. The forks and joins are the served ones.

Bits: the o_proj cells reproduce the built-in cuBLAS call. **The QKV cells do not, wherever the
served 9..32-row call is that same cuBLAS call** -- they sum K in four slices instead of one. They
keep the served bits only where a wide table is armed at those rows with a split_k 4 cell, and on
sm103 the path that arms one (``wide_projection_gathers``) is TEP4-only, so at TEP8 this lever
changes the decoded text. Its lever is lossless-numeric for that reason, not bitwise.

With ``dense_gemm_prefetch`` a projection that ran also forks a side stream that asks the L2 to
fetch the next projection's weight (QKV -> the same layer's attention output ->
the same layer's pooler slice; the unpooler -> the next layer's QKV), so the
next GEMM finds its weight in L2 after the latency-bound kernels in between
(prep and attention; the all-reduce and cluster boundary; the unpool gather and
boundary). The unpooler's own weight is not prefetched: between the pooler and
the unpooler the experts stream far more than the L2 holds. ``dense_gemm_prefetch`` therefore
needs the sharded projections, whose pooler GEMM reads only the rank's slice.

Named forks (``NamedFork``, ``DenseGemmPlan.add_fork``): side-stream work other
fast paths register under a name, each on a CUDA stream of its own, joined by the
dispatchers of one named site only (``join_role``) -- e.g. a prefetch forked
after a layer's expert kernel that the next layer's QKV dispatcher joins and the
unpooler's dispatcher in between does not -- and by the target at the end of
its forward. With no named fork registered the dispatchers join nothing by name.

The fork and its join (``DenseGemmPlan``): the next projection's dispatcher
joins the outstanding fork before anything else, and the target joins whatever
is still outstanding at the end of its forward, so a fork made inside a CUDA
graph capture is joined inside that capture, as the capture requires. A fork
records the capture it was made in (none when eager); a join waits for it only
from the same capture state. A fork from another state -- an eager step that
ended between a fork and its consumer, a capture that was abandoned, or a
breakable graph's segment that already joined it at its end -- is dropped
instead: the prefetch reads an immutable weight and writes nothing anyone
reads, so nothing depends on its completion, and a capture may not wait on
work outside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

import torch
from cuda.bindings import runtime as _cudart

from tensorrt_llm._torch.modules import low_m_gemm as _low_m_gemm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

from ...dispatch import LAUNCH_PDL
from ...ops.gemm.dense_gemm import (
    CUBLAS,
    DIRECT,
    MAX_ROWS,
    O_PROJ_SHAPE,
    QKV_SHAPE,
    SPLITK,
    WIDE_MAX_ROWS,
    dense_gemm,
    kernel_error,
    l2_prefetch,
    parse_kernel,
)

#: The first row count of a plan's wide table (``DenseGemmPlan.set_wide_kernels``).
WIDE_MIN_ROWS = MAX_ROWS + 1

#: The four projections of a layer, in the order a step runs them.
SITES = ("qkv", "o_proj", "pool", "unpool")

#: Kernel per site by row count on sm103 at tep8 (two trays, sharded pooler and
#: unpooler), as ``(last row count, kernel)`` steps, chosen in
#: 62-layer chains of cold weights behind predecessors that release at exit: the
#: direct kernel is the fastest for every shape at 1-2 rows and for the pooler and
#: unpooler slices at 3-4 rows; split-K 4 for QKV from 3 rows, and it ties with
#: the direct kernel on the pooler and beats it on the unpooler at 8 rows;
#: cuBLAS for the attention output projection from 3 rows. Rows 5-7 follow the
#: built-in autotuner's crossover. At 8 rows the pooler and unpooler run the
#: split-K-4 family every rank's autotuner picks, so their bits match the built-in's.
DEFAULT_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((2, "direct:256,1"), (MAX_ROWS, "splitk:64,8,4,12")),
    "o_proj": ((2, "direct:64,4"), (MAX_ROWS, CUBLAS)),
    "pool": ((2, "direct:256,1"), (6, "direct:128,2"), (MAX_ROWS, "splitk:64,8,4,12")),
    "unpool": ((2, "direct:256,1"), (5, "direct:64,2"), (MAX_ROWS, "splitk:64,8,4,12")),
}

#: The same on sm103 at tep4 (one tray), per-rank shapes QKV ``[4608, 8192]``, o_proj
#: ``[8192, 2048]`` and the sharded pooler / unpooler slices ``[1024, 8192]`` /
#: ``[2048, 4096]``, chosen the same way (``TP4_REPLICATED_PROJECTION_KERNELS``
#: replaces the pooler and unpooler entries when they are replicated). The direct
#: kernel is the fastest for every shape at 1-2 rows and for the pooler and
#: unpooler slices up to 5 and 4 rows; from 3 rows QKV runs split-K 2 (its N of
#: 4608 already fills the GPU with 72 column tiles), the slices split-K 4, and the
#: attention output projection cuBLAS, as at tep8.
TP4_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((2, "direct:256,2"), (MAX_ROWS, "splitk:64,8,2,10")),
    "o_proj": ((2, "direct:128,4"), (MAX_ROWS, CUBLAS)),
    "pool": ((2, "direct:256,2"), (5, "direct:256,4"), (MAX_ROWS, "splitk:64,8,4,10")),
    "unpool": ((2, "direct:128,2"), (4, "direct:128,4"), (MAX_ROWS, "splitk:64,8,4,10")),
}

#: The tep4 pooler and unpooler on the whole replicated weights ``[4096, 8192]`` /
#: ``[8192, 4096]``: the built-in model's tep4 projections, which it shards only at
#: tep8. Direct at 1-2 rows; from 3 rows split-K 2 for the pooler, cuBLAS for the
#: unpooler (K = 4096 leaves split-K no slices to spread).
TP4_REPLICATED_PROJECTION_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "pool": ((2, "direct:256,2"), (MAX_ROWS, "splitk:64,8,2,10")),
    "unpool": ((2, "direct:128,2"), (MAX_ROWS, CUBLAS)),
}

#: ``wide_dense_gemms`` on sm103 at tep8 (two trays): the QKV projection ``[2304, 8192]`` of the
#: 9..32-row calls of a decode step of 2..4 requests, as ``(last row count, kernel)`` steps from 9
#: rows. flashinfer's split-K with the ``split_k`` 4 of QKV's 8-row cell in ``DEFAULT_KERNELS`` --
#: required, not preferred: ``install``'s split-K check refuses any other, and with it every row of a
#: wide call gets the bits it gets in a one-request step. ``mma_m`` 64 gives 36 column tiles, 144
#: CTAs in clusters of 4 (one wave, the grid of the cuBLAS split-K these calls ran); ``mma_n`` 16 /
#: 32 holds the rows of two / three or four requests in one tile, at the deepest pipeline the shared
#: memory takes at ``split_k`` 4 (10 / 8 stages; the owner CTA also holds its three peers' partials,
#: which is why the tep4 sweeps' deepest split-K-4 pick is 10 at ``mma_n`` 16 where split-K 2 reaches
#: 11). It replaces the built-in low-M dispatch's cache-miss pick, cuBLAS's split-K 4 plus its
#: reduce kernel. Served here, QKV also forks the
#: attention output projection's L2 prefetch (``dense_gemm_prefetch``) at these rows, as at 1..8;
#: o_proj's dispatcher joins it before the built-in GEMM it keeps. The attention output projection
#: is not named: its 8-row cell is cuBLAS, which no split-K cell reproduces bit for bit. Nor are the
#: pooler and unpooler: above 8 rows the built-in tep8 model multiplies their whole replicated
#: weights, a shape these sites do not take; ``wide_projection_gathers`` names their slices in its
#: own TP8 table, armed per decode forward (``set_wide_kernels``), and leaves QKV to this one.
#: Chosen from the shapes, not from a per-call sweep; ``install``'s split-K check refuses a cell
#: that would change the bits.
SM103_TP8_WIDE_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((16, "splitk:64,16,4,10"), (WIDE_MAX_ROWS, "splitk:64,32,4,8")),
}

#: The ``wide_dense_gemms`` table of each ``(compute capability, TP size)`` it is measured for.
WIDE_KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {
    ((10, 3), 8): SM103_TP8_WIDE_KERNELS,
}

#: The sites whose wide cells are measured on the rank's slices (``sharded_projections``).
_SLICE_SITES = frozenset({"pool", "unpool"})


def wide_kernels(table: Mapping[str, Tuple[Tuple[int, str], ...]]) -> Dict[str, Tuple[str, ...]]:
    """The kernel of each site of a ``wide_dense_gemms`` table (``WIDE_KERNELS_BY_SM_TP``) for rows
    9..32 (index ``rows - 9``)."""
    out = {}
    for site, steps in table.items():
        kernels, first = [], MAX_ROWS + 1
        for last, kernel in steps:
            kernels += [kernel] * (last - first + 1)
            first = last + 1
        assert len(kernels) == WIDE_MAX_ROWS - MAX_ROWS, steps
        out[site] = tuple(kernels)
    return out


def wide_kernel_table(
    capability: Tuple[int, int],
    tp_size: int,
    sharded: bool,
    kernels: Mapping[str, Tuple[str, ...]],
) -> Dict[str, Tuple[str, ...]]:
    """``wide_dense_gemms``' kernel per row count 9..32 of each site its table names on
    ``capability`` at ``tp_size`` (``WIDE_KERNELS_BY_SM_TP``), checked against ``kernels``, the
    plan's 1..8-row table.

    Every wide cell must be flashinfer's split-K with the ``split_k`` of its site's 8-row cell:
    split-K tactics of one ``split_k`` sum the same K slices in the same order, so every row of a
    wide call gets the bits that row gets in a one-request step. Reads only the tables.

    Raises:
        ValueError: if ``WIDE_KERNELS_BY_SM_TP`` has no table for ``(capability, tp_size)``; if the
            table names the pooler or unpooler (cells measured on the rank's slices) and the
            projections are not ``sharded``; or if a wide cell does not sum the K slices of its
            site's 8-row cell.
    """
    key = (tuple(capability), int(tp_size))
    steps = WIDE_KERNELS_BY_SM_TP.get(key)
    if steps is None or (not sharded and _SLICE_SITES & set(steps)):
        measured = ", ".join(
            f"SM {sm} at TP{tp}"
            + (" with the sharded pooler and unpooler" if _SLICE_SITES & set(cells) else "")
            for (sm, tp), cells in sorted(WIDE_KERNELS_BY_SM_TP.items())
        )
        raise ValueError(
            f"[musespark_open] wide_dense_gemms is measured for {measured}; got SM {key[0]} at "
            f"TP{key[1]}, {'sharded' if sharded else 'replicated'} projections"
        )
    table = wide_kernels(steps)
    for role, wide_cells in table.items():
        family, fields = parse_kernel(kernels[role][MAX_ROWS - 1])
        for kernel in wide_cells:
            wide_family, wide_fields = parse_kernel(kernel)
            if family != SPLITK or wide_family != SPLITK or wide_fields[2] != fields[2]:
                raise ValueError(
                    f"[musespark_open] wide_dense_gemms: {role} kernel {kernel} does not sum "
                    f"the K slices of the {MAX_ROWS}-row kernel {kernels[role][MAX_ROWS - 1]}"
                )
    return table


#: The measured table of each ``(compute capability, TP size)``, with the pooler and
#: unpooler on the rank's slices (``kernel_steps`` swaps in the replicated tep4 cells).
KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {
    ((10, 3), 8): DEFAULT_KERNELS,
    ((10, 3), 4): TP4_KERNELS,
}

#: The pooler and unpooler cells of each tep4 table on the whole replicated weights.
TP4_REPLICATED_PROJECTION_KERNELS_BY_SM: Mapping[
    Tuple[int, int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {
    (10, 3): TP4_REPLICATED_PROJECTION_KERNELS,
}


#: ``batch_splitk``: the switch (read once, at ``install``). Off by default; on sm103 at TP8 with
#: the sharded pooler and unpooler, ``1`` runs the 33..160-row decode calls (5..20 requests at
#: K = 7) of the sites ``SM103_TP8_BATCH_KERNELS`` names on ``ops/gemm/batch_splitk_gemm`` (a
#: cluster split-K kernel whose kernel-N tile holds up to 256 rows, reducing in DSMEM) instead
#: of the built-in cuBLAS call (a split-K ``nvjet`` kernel plus ``cublasLt::splitKreduce`` for
#: most of these shapes and rows). Anywhere else it changes nothing but a warning.
#: The row band of ``batch_splitk``: above the 9..32-row wide band, up to the 20-request step.
BATCH_MIN_ROWS = WIDE_MAX_ROWS + 1
BATCH_MAX_ROWS = 160
#: ``tep8_batch_splitk``: whether ``install`` builds that table. Off keeps the built-in calls.
_BATCH_SPLITK = False


def configure(*, batch_splitk: bool = False, attn_block_splitk: bool = False) -> None:
    """Set whether ``install`` arms the batched split-K tables.

    ``lever_settings.apply`` sets them from the model's plan before any install; the defaults
    keep the built-in cuBLAS calls above the wide band and the served calls inside it.

    Args:
        batch_splitk: the ``tep8_batch_splitk`` lever (``_BATCH_SPLITK``).
        attn_block_splitk: the ``tep8_attn_block_splitk`` lever (``_ATTN_BLOCK_SPLITK``).
    """
    global _BATCH_SPLITK, _ATTN_BLOCK_SPLITK
    _BATCH_SPLITK = bool(batch_splitk)
    _ATTN_BLOCK_SPLITK = bool(attn_block_splitk)


#: ``batch_splitk`` on sm103 at TP8 (per-rank QKV ``[2304, 8192]``, pooler slice ``[512, 8192]``,
#: unpooler slice ``[1024, 4096]``), as ``(last row count, kernel)`` steps from 33 rows; rows past
#: a site's last step keep the built-in call. ``bsplitk:<mma_m>,<mma_n>,<split_k>,<stages>``:
#: ``mma_n`` public rows per CTA (a call takes ``ceil(rows / mma_n)`` row tiles). Every cell is
#: split_k 4, the split_k of each site's 8-row cell in ``DEFAULT_KERNELS`` and of the 9..32-row
#: kernels: K is summed in the same four slices in the same order, so every row gets the bits it
#: gets in a one-request step (``install`` checks the split_k). Chosen in 62-layer chains of cold
#: weights against the served cuBLAS call on one GB300 (microbench ``mb-h58-bsk-2``, each cell
#: the fastest split_k-4 tactic launched without PDL; checked through the overlay in
#: ``mb-h58-bsk-verify``): QKV one row tile (144 CTAs, the most clusters of 4 that fit) up to 128
#: rows, -0.1..+2.2 us against cuBLAS without PDL and +1.1..+3.4 us with it (from 136 rows
#: cuBLAS's two-CTA tiles tie or win); the pooler slice four row tiles (128 CTAs) and the
#: unpooler slice two (128 CTAs), +0.2..+2.6 us without PDL at every row count. The attention
#: output projection is not named: its split_k 1 cells (cuBLAS's bits) are slower than cuBLAS
#: without PDL at every row count.
SM103_TP8_BATCH_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": (
        (40, "bsplitk:64,40,4,8"),
        (48, "bsplitk:64,48,4,7"),
        (56, "bsplitk:64,56,4,7"),
        (64, "bsplitk:64,64,4,6"),
        (72, "bsplitk:64,72,4,6"),
        (80, "bsplitk:64,80,4,5"),
        (88, "bsplitk:64,88,4,5"),
        (96, "bsplitk:64,96,4,5"),
        (104, "bsplitk:64,104,4,4"),
        (112, "bsplitk:64,112,4,4"),
        (120, "bsplitk:64,120,4,4"),
        (128, "bsplitk:64,128,4,4"),
    ),
    "pool": (
        (48, "bsplitk:64,24,4,10"),
        (64, "bsplitk:64,16,4,11"),
        (96, "bsplitk:64,24,4,10"),
        (128, "bsplitk:64,32,4,9"),
        (160, "bsplitk:64,40,4,8"),
    ),
    "unpool": (
        (48, "bsplitk:64,24,4,8"),
        (64, "bsplitk:64,32,4,8"),
        (80, "bsplitk:64,40,4,8"),
        (96, "bsplitk:64,48,4,7"),
        (112, "bsplitk:64,56,4,7"),
        (128, "bsplitk:64,64,4,6"),
        (144, "bsplitk:64,72,4,6"),
        (160, "bsplitk:64,80,4,5"),
    ),
}

#: The ``batch_splitk`` table of each ``(compute capability, TP size)`` it is measured for.
BATCH_KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {((10, 3), 8): SM103_TP8_BATCH_KERNELS}


#: ``tep8_attn_block_splitk``, pushed here by ``lever_settings.apply`` (default off). On sm103 at
#: TP8 with the sharded pooler and unpooler it runs the 9..32-row calls of the sites and rows
#: ``SM103_TP8_ATTN_BLOCK_KERNELS`` names on ``ops/gemm/batch_splitk_gemm``; anywhere else it
#: changes nothing but a warning.
_ATTN_BLOCK_SPLITK = False

#: ``attn_block_splitk`` on sm103 at TP8, as ``(last row count, kernel)`` steps from 9 rows (``None``:
#: the served call at those rows). ``bsplitk:<mma_m>,<mma_n>,<split_k>,<stages>`` with one row tile
#: per call (``mma_n`` covers the rows), so each weight byte is streamed once.
#:
#: * o_proj ``[8192, 1024]``: split_k 1, the bits of the built-in cuBLAS call (which does not split
#:   K = 1024 at these rows); ``mma_m`` 64, 128 CTAs. Launched with PDL right behind the attention's
#:   early trigger, it loads its weight k-tiles before its wait.
#: * QKV ``[2304, 8192]``: split_k 4, the served cell's K slices and order (the reduce-scatter
#:   copy is bitwise with flashinfer's kernel at equal split_k), one more stage than flashinfer's
#:   cell and ``mma_n`` 24 at 17..24 rows; 9..16 rows keep flashinfer's cell (the copy was slower there).
#:
#: Measured on the E-MK chain of kernel stack 3 (62 layers, 8 ranks, two tray pairs; mb-emk-h217-1..4).
SM103_TP8_ATTN_BLOCK_KERNELS: Mapping[str, Tuple[Tuple[int, Optional[str]], ...]] = {
    "qkv": ((16, None), (24, "bsplitk:64,24,4,10"), (WIDE_MAX_ROWS, "bsplitk:64,32,4,9")),
    "o_proj": (
        (16, "bsplitk:64,16,1,8"),
        (24, "bsplitk:64,24,1,8"),
        (WIDE_MAX_ROWS, "bsplitk:64,32,1,8"),
    ),
}

#: The ``attn_block_splitk`` table of each ``(compute capability, TP size)`` it is measured for.
ATTN_BLOCK_KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, Optional[str]], ...]]
] = {((10, 3), 8): SM103_TP8_ATTN_BLOCK_KERNELS}

#: The ``split_k`` an ``attn_block_splitk`` cell of each site must have: QKV's 8-row cell's (flashinfer
#: split-K 4), and 1 for o_proj, whose 8-row cell is cuBLAS, which sums K = 1024 in one slice.
_ATTN_BLOCK_SPLIT_K = {"qkv": 4, "o_proj": 1}


def attn_block_kernels(
    table: Mapping[str, Tuple[Tuple[int, Optional[str]], ...]] = SM103_TP8_ATTN_BLOCK_KERNELS,
) -> Dict[str, Tuple[Optional[str], ...]]:
    """The ``attn_block_splitk`` kernel of each site per row count 9..32 (index ``rows - WIDE_MIN_ROWS``),
    ``None`` where the site keeps its served call."""
    out = {}
    for site, steps in table.items():
        kernels: List[Optional[str]] = []
        first = WIDE_MIN_ROWS
        for last, kernel in steps:
            assert first <= last <= WIDE_MAX_ROWS, steps
            kernels += [kernel] * (last - first + 1)
            first = last + 1
        assert first == WIDE_MAX_ROWS + 1, steps
        out[site] = tuple(kernels)
    return out


def attn_block_kernel_table(
    capability: Tuple[int, int], tp_size: int, sharded: bool
) -> Dict[str, Tuple[Optional[str], ...]]:
    """``attn_block_splitk``'s kernel per row count 9..32 of each site its table names on ``capability``
    at ``tp_size``, each cell checked: its ``split_k`` (``_ATTN_BLOCK_SPLIT_K``), one row tile at every
    row count it serves, and the site's shape (``batch_splitk_gemm.validate``). Reads only the tables.

    Raises:
        ValueError: if ``ATTN_BLOCK_KERNELS_BY_SM_TP`` has no table for ``(capability, tp_size)``, the
            projections are not ``sharded``, or a cell fails a check.
    """
    from ...ops.gemm import batch_splitk_gemm as _bsk

    key = (tuple(capability), int(tp_size))
    steps = ATTN_BLOCK_KERNELS_BY_SM_TP.get(key)
    if steps is None or not sharded:
        raise ValueError(
            f"[musespark_open] tep8_attn_block_splitk is measured for SM {(10, 3)} at TP8 with the "
            f"sharded "
            f"pooler and unpooler; got SM {key[0]} at TP{key[1]}, "
            f"{'sharded' if sharded else 'replicated'} projections"
        )
    shapes = {"qkv": QKV_SHAPE, "o_proj": O_PROJ_SHAPE}
    table = attn_block_kernels(steps)
    for role, cells in table.items():
        k, n = shapes[role]
        for rows, kernel in enumerate(cells, start=WIDE_MIN_ROWS):
            if kernel is None:
                continue
            tactic = _bsk.parse(kernel)
            if tactic.split_k != _ATTN_BLOCK_SPLIT_K[role]:
                raise ValueError(
                    f"[musespark_open] tep8_attn_block_splitk: {role} kernel {kernel} has split_k "
                    f"{tactic.split_k}; the site's bits need {_ATTN_BLOCK_SPLIT_K[role]}"
                )
            if rows > tactic.mma_n:
                raise ValueError(
                    f"[musespark_open] tep8_attn_block_splitk: {role} kernel {kernel} takes {rows} rows in "
                    f"more than one row tile"
                )
            _bsk.validate(tactic, rows, n, k)
    return table


def batch_kernels(
    table: Mapping[str, Tuple[Tuple[int, str], ...]] = SM103_TP8_BATCH_KERNELS,
) -> Dict[str, Tuple[Optional[str], ...]]:
    """The kernel of each site of a ``batch_splitk`` table per row count ``BATCH_MIN_ROWS..
    BATCH_MAX_ROWS`` (index ``rows - BATCH_MIN_ROWS``); ``None`` past the site's last step."""
    out = {}
    for site, steps in table.items():
        kernels: List[Optional[str]] = []
        first = BATCH_MIN_ROWS
        for last, kernel in steps:
            assert first <= last <= BATCH_MAX_ROWS, steps
            kernels += [kernel] * (last - first + 1)
            first = last + 1
        kernels += [None] * (BATCH_MAX_ROWS - first + 1)
        out[site] = tuple(kernels)
    return out


def batch_kernel_table(
    capability: Tuple[int, int],
    tp_size: int,
    sharded: bool,
    kernels: Mapping[str, Tuple[str, ...]],
) -> Dict[str, Tuple[Optional[str], ...]]:
    """``batch_splitk``'s kernel per row count 33..160 of each site its table names on ``capability``
    at ``tp_size``, checked against ``kernels`` (the plan's 1..8-row table): every cell must be a
    split-K tactic with the ``split_k`` of its site's 8-row cell. Reads only the tables.

    Raises:
        ValueError: if ``BATCH_KERNELS_BY_SM_TP`` has no table for ``(capability, tp_size)``, the
            projections are not ``sharded``, or a cell does not sum the K slices of its site's
            8-row cell.
    """
    from ...ops.gemm import batch_splitk_gemm as _bsk

    key = (tuple(capability), int(tp_size))
    steps = BATCH_KERNELS_BY_SM_TP.get(key)
    if steps is None or not sharded:
        raise ValueError(
            f"[musespark_open] batch_splitk is measured for SM {(10, 3)} at TP8 with the sharded "
            f"pooler and unpooler; got SM {key[0]} at TP{key[1]}, "
            f"{'sharded' if sharded else 'replicated'} projections"
        )
    table = batch_kernels(steps)
    for role, cells in table.items():
        family, fields = parse_kernel(kernels[role][MAX_ROWS - 1])
        for kernel in cells:
            if kernel is None:
                continue
            if family != SPLITK or _bsk.parse(kernel).split_k != fields[2]:
                raise ValueError(
                    f"[musespark_open] batch_splitk: {role} kernel {kernel} does not sum the K "
                    f"slices of the {MAX_ROWS}-row kernel {kernels[role][MAX_ROWS - 1]}"
                )
    return table


def _by_rows(steps: Tuple[Tuple[int, str], ...]) -> List[str]:
    kernels, first = [], 1
    for last, kernel in steps:
        kernels += [kernel] * (last - first + 1)
        first = last + 1
    assert len(kernels) == MAX_ROWS, steps
    return kernels


def kernel_steps(
    capability: Tuple[int, int], tp_size: int, sharded: bool
) -> Mapping[str, Tuple[Tuple[int, str], ...]]:
    """The ``(last row count, kernel)`` steps of each site on ``capability`` at ``tp_size``.

    ``sharded``: whether the pooler and unpooler GEMMs read the rank's slice
    (``sharded_projections``) rather than the whole replicated weight.

    Raises:
        ValueError: if ``KERNELS_BY_SM_TP`` has no table for ``(capability, tp_size)``.
    """
    key = (tuple(capability), int(tp_size))
    table = KERNELS_BY_SM_TP.get(key)
    if table is None:
        raise ValueError(
            f"[musespark_open] dense_gemms: no kernel table for SM {key[0]} at TP {key[1]} "
            f"(supported: {sorted(KERNELS_BY_SM_TP)})"
        )
    if key[1] == 4 and not sharded:
        table = {**table, **TP4_REPLICATED_PROJECTION_KERNELS_BY_SM[key[0]]}
    return table


def default_kernels(
    tp_size: int = 8, sharded: bool = True, capability: Tuple[int, int] = (10, 3)
) -> Dict[str, Tuple[str, ...]]:
    """The kernel of each site for rows 1..8 on ``capability`` at ``tp_size`` (``kernel_steps``)."""
    return {
        site: tuple(_by_rows(steps))
        for site, steps in kernel_steps(capability, tp_size, sharded).items()
    }


def device_capability() -> Tuple[int, int]:
    """The compute capability whose kernel table ``install`` uses: the current CUDA device's.

    On a host without CUDA, where no kernel launches (the wiring's unit tests), the
    target's own sm103.
    """
    if not torch.cuda.is_available():
        return (10, 3)
    return tuple(torch.cuda.get_device_capability())


def sharded_projections(model: torch.nn.Module) -> bool:
    """Whether the built-in model runs the sharded pooler and unpooler projections.

    The built-in constructor gives every MoE its projection-gather workspaces
    exactly when it shards them.
    """
    return all(
        getattr(layer.feed_forward, "_pool_gather_workspace", None) is not None
        and getattr(layer.feed_forward, "_unpool_gather_workspace", None) is not None
        for layer in model.layers
    )


def describe(kernels: Mapping[str, Tuple[str, ...]]) -> str:
    """``site=kernel@a-b,...`` for a log line."""
    parts = []
    for site in SITES:
        runs, first = [], 1
        for row in range(2, MAX_ROWS + 2):
            if row > MAX_ROWS or kernels[site][row - 1] != kernels[site][first - 1]:
                span = f"{first}" if first == row - 1 else f"{first}-{row - 1}"
                runs.append(f"{kernels[site][first - 1]}@{span}")
                first = row
        parts.append(f"{site}=" + "|".join(runs))
    return ", ".join(parts)


def current_capture_id() -> Optional[int]:
    """The id of the CUDA graph capture the current stream belongs to, or ``None`` when eager.

    A stream that joined a capture (by waiting on a capturing stream) reports the
    id of that capture, so the main stream and a side stream it forked agree.
    """
    if not torch.cuda.is_current_stream_capturing():
        return None
    _, capture_id, *_ = CUASSERT(
        _cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream().cuda_stream)
    )
    return int(capture_id)


def dedicated_stream(device: torch.device) -> torch.cuda.ExternalStream:
    """A CUDA stream of its own (non-blocking, default priority), never one of PyTorch's pool.

    PyTorch hands out its pooled streams round robin, and a CUDA graph capture
    takes its capturing stream from the same pool, so a pooled side stream can be
    the capturing stream itself; a fork and join on it are then in-order work on
    one stream. A stream made here can never be that stream. It lives as long as
    the process.
    """
    with torch.cuda.device(device):
        handle = CUASSERT(_cudart.cudaStreamCreateWithFlags(_cudart.cudaStreamNonBlocking))[0]
    return torch.cuda.ExternalStream(int(handle), device=device)


class NamedFork:
    """Side-stream work the critical path waits for only at one named consumer.

    ``fork(launch)`` makes the fork's own stream wait for the current stream's
    work so far and calls ``launch()`` with that stream current; ``join()`` makes
    the current stream wait for everything forked since the last join. A fork
    names the dense projection site whose dispatcher joins it before its GEMM
    (``join_role``, one of ``SITES``; ``DenseGemmPlan.join_role``), so a fork made
    after, say, a layer's expert kernel is joined by the next layer's QKV
    dispatcher and by no dispatcher in between; ``join_role=None`` leaves the
    join to its owner (``join()``). The target joins every fork still outstanding
    at the end of its forward (``DenseGemmPlan.join_all``), so a fork made inside
    a CUDA graph capture is joined inside it.

    The capture rules are ``DenseGemmPlan``'s for the prefetch fork: a fork
    records the capture state it was made in, a join waits for it only from the
    same state and otherwise drops it (``dropped``); work forked from another
    state cannot be waited on, so a fork may carry only work nothing reads
    (an L2 prefetch, for one). A fork while a fork from another capture state
    is still outstanding drops that one first. Forks made one after another
    without a join are ordered by the side stream itself and covered by the
    next join. The stream and ``sink`` (an int32 scratch word the launch may
    use) are made at the first fork outside a capture -- the engine's eager
    warm-up -- and a fork made inside a capture before that launches nothing
    (``skipped``).

    In a captured graph the join is a dependency of the consumer's kernel: a
    full edge in front of a plain launch, a programmatic edge in front of a
    kernel launched with programmatic dependent launch, so the
    consumer launches once every CTA of the forked kernels has exited or
    triggered, and its ``griddepcontrol.wait`` returns once they completed.
    """

    def __init__(self, name: str, join_role: Optional[str]) -> None:
        if join_role is not None and join_role not in SITES:
            raise ValueError(f"fork {name!r}: join_role {join_role!r} is not one of {SITES}")
        self.name = name
        self.join_role = join_role
        self.pending = False
        self.forks = 0
        self.joins = 0
        self.dropped = 0
        self.skipped = 0
        self._fork_capture: Optional[int] = None
        self._stream: Optional[torch.cuda.ExternalStream] = None
        self.sink: Optional[torch.Tensor] = None

    @property
    def stream(self) -> Optional[torch.cuda.ExternalStream]:
        """The fork's own stream, or ``None`` before the first fork outside a capture."""
        return self._stream

    def fork(self, launch: Callable[[], None], device: Optional[torch.device] = None) -> bool:
        """Run ``launch()`` on the fork's stream after the current stream's work so far.

        Returns whether it ran (False: inside a capture before the stream exists).
        """
        capture = current_capture_id()
        if self._stream is None:
            if capture is not None:
                # A stream and the sink cannot be made inside a capture; the
                # engine's eager warm-up makes them before it captures.
                self.skipped += 1
                return False
            if device is None:
                device = torch.device("cuda", torch.cuda.current_device())
            self._stream = dedicated_stream(device)
            self.sink = torch.zeros(1, dtype=torch.int32, device=device)
        if self.pending and self._fork_capture != capture:
            self.pending = False
            self.dropped += 1
        self._stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self._stream):
            launch()
        self._fork_capture = capture
        self.pending = True
        self.forks += 1
        return True

    def join(self) -> None:
        """Make the current stream wait for the outstanding forks made in the same capture
        state; drop them if they were made in another."""
        if not self.pending:
            return
        self.pending = False
        if current_capture_id() == self._fork_capture:
            torch.cuda.current_stream().wait_stream(self._stream)
            self.joins += 1
        else:
            self.dropped += 1


class DenseGemmPlan:
    """What every site of one model shares: the kernel per site and row count, PDL, the
    prefetch stream.

    ``kernels[site][rows - 1]`` is the kernel of a call of ``rows`` rows.
    ``wide_kernels[site][rows - WIDE_MIN_ROWS]`` is the kernel of a call of 9..32 rows
    while ``wide_armed`` (only for the sites the wide table names; none by default).
    ``pending`` is True between a prefetch fork and the join that follows it: the
    next projection's dispatcher, or the target's at the end of its forward.
    ``dropped`` counts the forks a join dropped because they were made in another
    capture state (see the module docstring).
    """

    def __init__(
        self,
        kernels: Mapping[str, Tuple[str, ...]],
        prefetch: bool,
        enable_pdl: bool,
        capability: Optional[Tuple[int, int]] = None,
        wide: Optional[Mapping[str, Tuple[str, ...]]] = None,
        attn_block: Optional[Mapping[str, Tuple[Optional[str], ...]]] = None,
    ) -> None:
        self.kernels = dict(kernels)
        #: ``wide_dense_gemms``: the kernel of a site's 9..32-row calls (``wide[site][rows - 9]``);
        #: a site it does not name keeps the built-in dispatch at those rows.
        self.wide = dict(wide or {})
        #: ``attn_block_splitk``: the batch split-K cell of a site's 9..32-row calls
        #: (``attn_block[site][rows - WIDE_MIN_ROWS]``, ``None`` where the served call runs).
        #: Empty when the lever is off, which is what ``attn_block_kernel_for`` reads.
        self.attn_block = dict(attn_block or {})
        #: The 9..32-row table (``set_wide_kernels``) and whether the current forward arms it.
        self.wide_kernels: Dict[str, Tuple[str, ...]] = {}
        self.wide_armed = False
        #: The fast path armed for the current decode forward above 32 rows (``batch_dense_gemms``'s
        #: ``BatchDenseGemms``), whose ``projection`` every re-classed projection's
        #: ``apply_linear`` calls; ``None`` otherwise.
        self.batch = None
        #: ``batch_splitk``: the kernel of a site's 33..160-row calls
        #: (``batch_splitk[site][rows - BATCH_MIN_ROWS]``, ``None`` past its last step); empty
        #: when off (``set_batch_splitk_kernels``).
        self.batch_splitk: Dict[str, Tuple[Optional[str], ...]] = {}
        self.prefetch = prefetch
        self.enable_pdl = enable_pdl
        #: The compute capability of the device the plan serves (``install`` reads it once);
        #: what an arch-specific producer policy keys on.
        self.capability = None if capability is None else tuple(capability)
        self.tp_size: Optional[int] = None
        #: Named forks by name (``add_fork``), each joined by its role's dispatcher.
        self.named_forks: Dict[str, NamedFork] = {}
        self.pending = False
        self.dropped = 0
        self._fork_capture: Optional[int] = None
        self._stream: Optional[torch.cuda.Stream] = None
        self._sink: Optional[torch.Tensor] = None
        #: Whether the side stream was the capturing stream at the first fork made in a
        #: capture (None before one). PyTorch hands out pooled streams round robin, and
        #: the capture may take the same one: the fork and join are then in-order work on
        #: one stream, the same result with the prefetch on the critical path.
        self.side_stream_is_capture_stream: Optional[bool] = None

    def kernel_for(self, role: str, rows: int) -> Optional[str]:
        """The kernel of a ``rows``-row call at site ``role``, or None where the plan has none
        (the built-in dispatch): ``kernels`` at 1..8 rows; at 9..32 rows the ``wide`` table
        (``wide_dense_gemms``) for the sites it names, else ``wide_kernels`` while ``wide_armed``
        (``wide_projection_gathers``)."""
        if 1 <= rows <= MAX_ROWS:
            return self.kernels[role][rows - 1]
        if not WIDE_MIN_ROWS <= rows <= WIDE_MAX_ROWS:
            return None
        wide = self.wide.get(role)
        if wide is not None:
            return wide[rows - WIDE_MIN_ROWS]
        if self.wide_armed:
            wide = self.wide_kernels.get(role)
            return None if wide is None else wide[rows - WIDE_MIN_ROWS]
        return None

    def kernel(self, role: str, rows: int) -> Optional[str]:
        """``kernel_for``: the kernel of a ``role`` call of ``rows`` rows, or ``None``."""
        return self.kernel_for(role, rows)

    def attn_block_kernel_for(self, role: str, rows: int) -> Optional[str]:
        """``attn_block_splitk``'s cell of a 9..32-row ``role`` call, or ``None`` (the served call)."""
        cells = self.attn_block.get(role)
        if cells is None or not WIDE_MIN_ROWS <= rows <= WIDE_MAX_ROWS:
            return None
        return cells[rows - WIDE_MIN_ROWS]

    def batch_kernel_for(self, role: str, rows: int) -> Optional[str]:
        """``batch_splitk``'s kernel of a 33..160-row ``role`` call, or ``None`` (the built-in call)."""
        cells = self.batch_splitk.get(role)
        if cells is None or not BATCH_MIN_ROWS <= rows <= BATCH_MAX_ROWS:
            return None
        return cells[rows - BATCH_MIN_ROWS]

    def set_batch_splitk_kernels(self, kernels: Mapping[str, Tuple[Optional[str], ...]]) -> None:
        """Give sites a ``batch_splitk`` kernel per row count 33..160 (``batch_kernel_table``).

        Raises:
            ValueError: for an unknown site or a table that does not cover 33..160 rows.
        """
        rows = BATCH_MAX_ROWS - BATCH_MIN_ROWS + 1
        for role, table in kernels.items():
            if role not in SITES or len(table) != rows:
                raise ValueError(
                    f"[musespark_open] dense_gemms: a batch_splitk table names {rows} kernels per "
                    f"site of {SITES}, got {role!r} with {len(table)}"
                )
        self.batch_splitk = {role: tuple(table) for role, table in kernels.items()}

    def set_wide_kernels(self, kernels: Mapping[str, Tuple[str, ...]]) -> None:
        """Give sites a kernel per row count 9..32 (``kernels[site][rows - WIDE_MIN_ROWS]``).

        Raises:
            ValueError: for an unknown site or a table that does not cover 9..32 rows.
        """
        rows = WIDE_MAX_ROWS - WIDE_MIN_ROWS + 1
        for role, table in kernels.items():
            if role not in SITES or len(table) != rows:
                raise ValueError(
                    f"[musespark_open] dense_gemms: a wide table names {rows} kernels per site of "
                    f"{SITES}, got {role!r} with {len(table)}"
                )
            for kernel in table:
                parse_kernel(kernel)
        self.wide_kernels = {role: tuple(table) for role, table in kernels.items()}

    def fork_prefetch(self, weight: torch.Tensor) -> None:
        """Ask the L2 for ``weight`` on the side stream, after the current stream's work so far."""
        capture = current_capture_id()
        if self._stream is None:
            if capture is not None:
                # A stream and the sink cannot be made inside a capture; the
                # engine's eager warm-up makes them before it captures.
                return
            self._stream = torch.cuda.Stream(device=weight.device)
            self._sink = torch.zeros(1, dtype=torch.int32, device=weight.device)
        current = torch.cuda.current_stream(weight.device)
        if capture is not None and self.side_stream_is_capture_stream is None:
            self.side_stream_is_capture_stream = self._stream.cuda_stream == current.cuda_stream
            if self.side_stream_is_capture_stream:
                logger.warning(
                    "[musespark_open] dense_gemm_prefetch: the prefetch side stream is the capturing stream; its forks "
                    "and joins are in-order work on that stream"
                )
            else:
                logger.info(
                    "[musespark_open] dense_gemm_prefetch: the prefetch side stream is not the capturing stream"
                )
        self._stream.wait_stream(current)
        with torch.cuda.stream(self._stream):
            l2_prefetch(weight, self._sink)
        self._fork_capture = capture
        self.pending = True

    def join(self) -> None:
        """Make the current stream wait for an outstanding prefetch fork made in the same
        capture state; drop one made in another."""
        if not self.pending:
            return
        self.pending = False
        if current_capture_id() == self._fork_capture:
            torch.cuda.current_stream().wait_stream(self._stream)
        else:
            self.dropped += 1

    def add_fork(self, name: str, join_role: Optional[str]) -> NamedFork:
        """Register a named fork joined by the ``join_role`` site's dispatchers (see ``NamedFork``).

        Raises:
            ValueError: if ``name`` is taken or ``join_role`` is not a site.
        """
        if name in self.named_forks:
            raise ValueError(f"a fork named {name!r} is already registered")
        fork = NamedFork(name, join_role)
        self.named_forks[name] = fork
        return fork

    def join_role(self, role: str) -> None:
        """Join every named fork whose consumer is the ``role`` site; the others stay outstanding."""
        for fork in self.named_forks.values():
            if fork.join_role == role:
                fork.join()

    def join_all(self) -> None:
        """Join the prefetch fork and every named fork (a target's end of forward)."""
        self.join()
        for fork in self.named_forks.values():
            fork.join()


@dataclass
class _Site:
    plan: DenseGemmPlan
    role: str
    layer_idx: int
    #: ``(K, N)`` of the weight the site's GEMM is handed.
    shape: Tuple[int, int]
    #: The weight to prefetch after this site's GEMM (``dense_gemm_prefetch``), read at call time.
    next_weight: Optional[Callable[[], torch.Tensor]]
    #: Row counts whose call forks no prefetch: there the next GEMM streams its weight
    #: another way (``boundary_early_release``: launched early by its predecessor, it loads the weight
    #: before its own dependency wait), or the fork's join would gate the next GEMM
    #: (``direct_publish``: o_proj publishes into the hop, which then ends before the pool weight's
    #: prefetch does).
    skip_prefetch_rows: FrozenSet[int] = frozenset()


def _keeps_prefetch(site: "_Site", rows: int) -> bool:
    """Whether ``site``'s call of ``rows`` rows forks its prefetch (the module's rule)."""
    plan = site.plan
    if not (plan.prefetch and site.next_weight is not None and rows not in site.skip_prefetch_rows):
        return False
    return True


#: The kernel families of ``ops/gemm/dense_gemm``.
KERNEL_FAMILIES = frozenset({CUBLAS, SPLITK, DIRECT})
#: flashinfer's families: split-K (the owner CTA of each cluster reduces the K slices
#: through DSMEM and stores the rounded tile once) and the direct SIMT kernel (thread 0
#: of each CTA stores each of its outputs once), both from the epilogue, after the
#: kernel's dependency wait.
FLASHINFER_FAMILIES = frozenset({SPLITK, DIRECT})


@dataclass(frozen=True)
class PublishPolicy:
    """How a site's GEMM fills a direct-transport slot: one policy per site, claimed at install.

    A fast path that has a projection publish into a collective's transport claims that
    site's dispatcher with its policy (``claim_publish``) when it installs, and arms the
    dispatcher per forward by setting ``DenseGemmDispatcher.publish`` to the rank's slot.
    ``unpool_direct`` claims the unpool sites (flashinfer's kernels store into the slot, cuBLAS is
    copied); ``direct_publish`` claims the attention output sites with the device's producer
    families storing into the slot (cuBLAS on sm103) and everything else copied.

    Attributes:
        owner: the claiming policy's name (messages, and the one-owner-per-site check).
        direct: kernel families whose call stores straight into the slot: each must store
            every element exactly once, after its dependency wait, as the owner certified.
        copy: ``copy(slot, product)`` stores a product computed elsewhere into the slot
            from an SM kernel, one store per element (never the copy engine).
    """

    owner: str
    direct: FrozenSet[str]
    copy: Callable[[torch.Tensor, torch.Tensor], None]


def claim_publish(dispatcher, policy: PublishPolicy) -> None:
    """Give ``dispatcher``'s site the producer ``policy``; at install, before any arming.

    Raises:
        ValueError: if another feature already claimed the site (one producer contract
            per slot writer), or the policy names an unknown kernel family.
    """
    unknown = sorted(policy.direct - KERNEL_FAMILIES)
    if unknown:
        raise ValueError(f"[musespark_open] {policy.owner}: unknown kernel families {unknown}")
    held = getattr(dispatcher, "publish_policy", None)
    if held is not None and held.owner != policy.owner:
        site = getattr(dispatcher, "site", None)
        where = f"layer {site.layer_idx} {site.role}" if site is not None else repr(dispatcher)
        raise ValueError(
            f"[musespark_open] {policy.owner}: the {where} GEMM already publishes for {held.owner}; "
            "a site has one producer policy, so the two features cannot share it"
        )
    dispatcher.publish_policy = policy


class DenseGemmDispatcher:
    """Sits in a projection's ``_low_m_gemm_dispatcher`` slot; same ``apply`` signature.

    ``builtin`` is the engine's per-model ``LowMGemmDispatcher`` (or ``None`` until
    the engine assigns it, when the module-level fallback dispatcher serves, as
    ``apply_low_m_gemm`` would).
    """

    def __init__(self, site: _Site) -> None:
        self.site = site
        self.builtin = None
        #: While set (by an armed transport, for one decode forward), a call writes the
        #: product into this tensor -- the rank's slot of a multicast transport -- and
        #: returns it, instead of a fresh output, following ``publish_policy``. ``None``
        #: otherwise.
        self.publish: Optional[torch.Tensor] = None
        #: The producer policy the feature that owns this site's slot claimed at install
        #: (``claim_publish``); ``None`` while no feature publishes from this site.
        self.publish_policy: Optional[PublishPolicy] = None
        #: While set (by a fast path that already produced this call's product another way,
        #: for one call), the next call of its row count and width returns this tensor after the
        #: joins and the prefetch fork, without a GEMM, and clears it. ``None`` otherwise.
        self.substitute: Optional[torch.Tensor] = None

    @staticmethod
    def is_decode_call(input_tensor: torch.Tensor) -> bool:
        """A BF16 inference call of 1-8 rows on the GPU: a call this feature exists to serve."""
        rows = input_tensor.numel() // max(int(input_tensor.shape[-1]), 1)
        return (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and 1 <= rows <= MAX_ROWS
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )

    def fits(self, input_tensor: torch.Tensor, weight: torch.Tensor, bias) -> bool:
        """Whether ``dense_gemm`` serves this call: BF16, no bias, the site's shape, a row count
        the plan has a kernel for (``DenseGemmPlan.kernel_for``)."""
        if bias is not None or torch.is_grad_enabled() or torch.compiler.is_compiling():
            return False
        if not (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.dim() == 2
            and input_tensor.dim() >= 1
            and (int(weight.shape[1]), int(weight.shape[0])) == self.site.shape
            and int(input_tensor.shape[-1]) == int(weight.shape[1])
            and input_tensor.is_contiguous()
            and weight.is_contiguous()
        ):
            return False
        rows = input_tensor.numel() // int(weight.shape[1])
        # The split-K kernel's TMA needs 32-byte aligned operands.
        return (
            self.site.plan.kernel_for(self.site.role, rows) is not None
            and input_tensor.data_ptr() % 32 == 0
            and weight.data_ptr() % 32 == 0
        )

    def apply(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        force_active: bool = False,
    ) -> Optional[torch.Tensor]:
        plan = self.site.plan
        plan.join()
        if plan.named_forks:
            plan.join_role(self.site.role)
        if self.substitute is not None:
            return self._apply_substitute(input_tensor, weight)
        if self.publish is not None:
            return self._apply_publish(module, input_tensor, weight, bias, force_active)
        if plan.attn_block:
            out = self._attn_block_apply(input_tensor, weight, bias)
            if out is not None:
                return out
        if not self.fits(input_tensor, weight, bias):
            if input_tensor.dim() >= 1 and self.is_decode_call(input_tensor):
                # The built-in dispatch picks its kernel per rank: say so once.
                logger.warning_once(
                    f"[musespark_open] dense_gemms: a {self.site.role} call of layer {self.site.layer_idx} did not "
                    f"fit dense_gemm (input {tuple(input_tensor.shape)} "
                    f"contiguous={input_tensor.is_contiguous()}, weight {tuple(weight.shape)} "
                    f"{weight.dtype}, bias={bias is not None}); it runs the built-in low-M "
                    "dispatch, whose kernel is chosen per rank",
                    key=f"musespark_open.dense_gemms.decline_{self.site.role}",
                )
            builtin = self.builtin if self.builtin is not None else _low_m_gemm._DISPATCHER
            return builtin.apply(module, input_tensor, weight, bias, force_active=force_active)
        k = int(weight.shape[1])
        x = input_tensor.view(-1, k)
        rows = int(x.shape[0])
        kernel = plan.kernel_for(self.site.role, rows)
        out = dense_gemm(x, weight, kernel, plan.enable_pdl)
        if _keeps_prefetch(self.site, rows):
            plan.fork_prefetch(self.site.next_weight())
        return out.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def _apply_substitute(self, input_tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """``apply`` with ``self.substitute`` as the product: no GEMM; the prefetch fork as ``apply``.

        Raises:
            RuntimeError: if the call's rows or width are not the substitute's.
        """
        product, self.substitute = self.substitute, None
        k = int(weight.shape[1])
        rows = input_tensor.numel() // max(k, 1)
        if tuple(product.shape) != (rows, int(weight.shape[0])):
            raise RuntimeError(
                f"[musespark_open] layer {self.site.layer_idx} {self.site.role} was handed a "
                f"{tuple(product.shape)} product for a call of {rows} rows x {int(weight.shape[0])}"
            )
        if _keeps_prefetch(self.site, rows):
            self.site.plan.fork_prefetch(self.site.next_weight())
        return product.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def _attn_block_apply(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """``attn_block_splitk``: a 9..32-row QKV or o_proj call on the plan's batch split-K cell for its
        rows, or ``None`` (the served call): BF16, no bias, the site's shape, contiguous, 32-byte aligned,
        not compiled or differentiated. Joins happened in ``apply``. Forks what the served call forks:
        QKV (``fits``' path) the next weight's prefetch; o_proj (the built-in call's path) nothing."""
        plan = self.site.plan
        if bias is not None or torch.is_grad_enabled() or torch.compiler.is_compiling():
            return None
        if not (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.dim() == 2
            and input_tensor.dim() >= 1
            and (int(weight.shape[1]), int(weight.shape[0])) == self.site.shape
            and int(input_tensor.shape[-1]) == int(weight.shape[1])
            and input_tensor.is_contiguous()
            and weight.is_contiguous()
            and input_tensor.data_ptr() % 32 == 0
            and weight.data_ptr() % 32 == 0
        ):
            return None
        k = int(weight.shape[1])
        rows = input_tensor.numel() // k
        kernel = plan.attn_block_kernel_for(self.site.role, rows)
        if kernel is None:
            return None
        from ...ops.gemm import batch_splitk_gemm as _bsk

        x = input_tensor.view(-1, k)
        out = torch.empty((rows, int(weight.shape[0])), dtype=x.dtype, device=x.device)
        _bsk.run(x, weight, out, _bsk.parse(kernel), plan.enable_pdl)
        if (
            self.site.role != "o_proj"
            and plan.prefetch
            and self.site.next_weight is not None
            and rows not in self.site.skip_prefetch_rows
        ):
            plan.fork_prefetch(self.site.next_weight())
        logger.info_once(
            f"[musespark_open] tep8_attn_block_splitk: a {rows}-row {self.site.role} call (layer "
            f"{self.site.layer_idx}) ran {kernel}",
            key=f"musespark_open.attn_block_splitk.{self.site.role}.{rows}",
        )
        return out.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def batch_splitk_apply(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        out: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """A 33..160-row call on the plan's ``batch_splitk`` kernel for its rows, or ``None`` (the caller
        runs the built-in call): BF16, no bias, the site's shape, contiguous, 32-byte aligned, not
        compiled or differentiated. ``out`` (``projection_publish``'s transport slot) receives the
        product; the kernel stores each element of it once, after the reduction. Joins the plan's
        forks the site's dispatcher would have joined before its GEMM."""
        plan = self.site.plan
        if not plan.batch_splitk or bias is not None:
            return None
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return None
        if not (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.dim() == 2
            and input_tensor.dim() >= 1
            and (int(weight.shape[1]), int(weight.shape[0])) == self.site.shape
            and int(input_tensor.shape[-1]) == int(weight.shape[1])
            and input_tensor.is_contiguous()
            and weight.is_contiguous()
            and input_tensor.data_ptr() % 32 == 0
            and weight.data_ptr() % 32 == 0
        ):
            return None
        k = int(weight.shape[1])
        rows = input_tensor.numel() // k
        kernel = plan.batch_kernel_for(self.site.role, rows)
        if kernel is None:
            return None
        from ...ops.gemm import batch_splitk_gemm as _bsk

        plan.join()
        if plan.named_forks:
            plan.join_role(self.site.role)
        x = input_tensor.view(-1, k)
        if out is None:
            out = torch.empty((rows, int(weight.shape[0])), dtype=x.dtype, device=x.device)
        _bsk.run(x, weight, out, _bsk.parse(kernel), plan.enable_pdl)
        logger.info_once(
            f"[musespark_open] batch_splitk: a {rows}-row {self.site.role} call (layer "
            f"{self.site.layer_idx}) ran {kernel}",
            key=f"musespark_open.batch_splitk.{self.site.role}",
        )
        return out.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def _apply_publish(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        force_active: bool,
    ) -> torch.Tensor:
        """``apply`` with the product written into ``self.publish``, by the site's policy.

        Every rank's consumer polls the slot, so this call must fill it whatever
        this rank's layout: a call that fits runs the plan's kernel for its rows
        with the slot as its output when ``publish_policy.direct`` names that
        kernel's family (its epilogue stores are the publication, one store per
        element); any other call -- a family the policy does not let store into
        the slot, or a rank-local layout ``fits`` declines -- runs ``apply``
        disarmed and the policy's ``copy`` stores the product into the slot. Same
        bits either way. The prefetch fork follows ``apply``'s rule,
        ``skip_prefetch_rows`` included.

        Raises:
            RuntimeError: if no fast path claimed the site (``claim_publish``), or the
                call's shape is not the slot's (a transport is armed for the step's
                row count only).
        """
        publish = self.publish
        policy = self.publish_policy
        if policy is None:
            raise RuntimeError(
                f"[musespark_open] layer {self.site.layer_idx} {self.site.role} was armed to publish, but "
                "no fast path claimed the site's producer policy (claim_publish at install)"
            )
        k = int(weight.shape[1])
        rows = input_tensor.numel() // max(k, 1)
        if rows != int(publish.shape[0]) or int(weight.shape[0]) != int(publish.shape[1]):
            raise RuntimeError(
                f"[musespark_open] {policy.owner}: layer {self.site.layer_idx} {self.site.role} was armed "
                f"to publish {tuple(publish.shape)} but got {rows} rows x {int(weight.shape[0])}"
            )
        plan = self.site.plan
        kernel = plan.kernel_for(self.site.role, rows)
        if self.fits(input_tensor, weight, bias) and parse_kernel(kernel)[0] in policy.direct:
            dense_gemm(input_tensor.view(-1, k), weight, kernel, plan.enable_pdl, out=publish)
            if _keeps_prefetch(self.site, rows):
                plan.fork_prefetch(self.site.next_weight())
        else:
            logger.warning_once(
                f"[musespark_open] {policy.owner}: a {self.site.role} call (layer {self.site.layer_idx}, "
                f"kernel {kernel}, input {tuple(input_tensor.shape)} "
                f"contiguous={input_tensor.is_contiguous()}) does not store into the transport "
                f"slot itself (stores: {sorted(policy.direct)}); its product is copied there",
                key=f"musespark_open.{policy.owner}_copy_{self.site.role}",
            )
            self.publish = None
            try:
                out = self.apply(module, input_tensor, weight, bias, force_active=force_active)
            finally:
                self.publish = publish
            if out is None:
                out = torch.nn.functional.linear(input_tensor, weight)
            policy.copy(publish, out)
        return publish.view(*input_tensor.shape[:-1], int(weight.shape[0]))


class MuseSparkOpenTepDenseLinear(Linear):
    """A built-in ``Linear`` whose low-M dispatcher slot always yields this feature's dispatcher.

    Never constructed directly: ``adopt`` re-classes a built-in ``Linear`` in
    place and adds one plain attribute. What the engine assigns to the slot
    becomes the dispatcher's fallback for the calls it does not serve.
    """

    @property
    def _low_m_gemm_dispatcher(self) -> DenseGemmDispatcher:
        return self.__dict__["_low_m_dispatcher"]

    @_low_m_gemm_dispatcher.setter
    def _low_m_gemm_dispatcher(self, builtin) -> None:
        self.__dict__["_low_m_dispatcher"].builtin = builtin

    def apply_linear(self, input, bias, lora_params=None, layer_idx=None):
        """The built-in ``apply_linear``; while a fast path arms the plan (``DenseGemmPlan.batch``)
        the call goes through that path's ``projection`` with the built-in call as a callable
        and this module (when the call carries no bias)."""
        dispatcher = self.__dict__["_low_m_dispatcher"]
        batch = dispatcher.site.plan.batch
        lora = getattr(self, "lora", None) is not None and bool(lora_params)
        if batch is None or lora:
            if dispatcher.site.plan.batch_splitk and not lora and _builtin_is_cublas(self):
                out = dispatcher.batch_splitk_apply(input, self.weight, bias)
                if out is not None:
                    return out
            return Linear.apply_linear(self, input, bias, lora_params, layer_idx)
        return batch.projection(
            dispatcher.site,
            input,
            lambda: Linear.apply_linear(self, input, bias, lora_params, layer_idx),
            self if bias is None else None,
        )

    @classmethod
    def check(cls, linear: torch.nn.Module, site: str) -> None:
        """Whether ``adopt`` may re-class this module; changes nothing.

        Raises:
            TypeError: if ``linear`` is not exactly the built-in ``Linear``.
            ValueError: if its calls cannot reach the low-M dispatch (a custom
                cuBLAS GEMM) or carry a bias (``dense_gemm`` has none).
        """
        if type(linear) is not Linear:
            raise TypeError(
                f"dense_gemms re-classes the built-in Linear of {site}, got {type(linear).__name__}"
            )
        if getattr(linear, "use_custom_cublas_mm", False):
            raise ValueError(f"{site} runs a custom cuBLAS GEMM, which skips the low-M dispatch")
        if getattr(linear, "bias", None) is not None:
            raise ValueError(f"{site} carries a bias")

    @classmethod
    def adopt(
        cls, linear: Linear, dispatcher: DenseGemmDispatcher
    ) -> "MuseSparkOpenTepDenseLinear":
        builtin = linear.__dict__.pop("_low_m_gemm_dispatcher", None)
        linear.__class__ = cls
        linear.__dict__["_low_m_dispatcher"] = dispatcher
        dispatcher.builtin = builtin
        return linear


def _builtin_is_cublas(linear: Linear) -> bool:
    """Whether the built-in call of ``linear`` above 32 rows is ``F.linear`` on its BF16 weight
    (``UnquantizedLinearMethod.apply``): no custom cuBLAS GEMM, no CuTe DSL BF16 GEMM."""
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

    return (
        isinstance(getattr(linear, "quant_method", None), UnquantizedLinearMethod)
        and not getattr(linear, "use_custom_cublas_mm", False)
        and not getattr(linear, "use_cute_dsl_bf16_gemm", False)
    )


def _projection_weight(moe, linear: Linear, sharded: bool) -> Callable[[], torch.Tensor]:
    """The weight a pooler / unpooler GEMM is handed: the rank's row slice when sharded."""
    if not sharded:
        return lambda: linear.weight

    def rank_slice() -> torch.Tensor:
        columns = linear.weight.shape[0] // moe.mapping.tp_size
        begin = moe.mapping.tp_rank * columns
        return linear.weight[begin : begin + columns]

    return rank_slice


def install(
    model: torch.nn.Module,
    prefetch: bool,
    kernels: Optional[Mapping[str, Tuple[str, ...]]] = None,
    wide: bool = False,
) -> DenseGemmPlan:
    """Route every layer's four projections through ``dense_gemm``; return the shared plan.

    ``kernels`` is the kernel of each site per row count; when None, the table of the
    device's compute capability (``device_capability``, read once here), the model's TP
    size and its projection layout (``default_kernels``). ``wide`` (``wide_dense_gemms``, tep8's
    QKV) adds the TP size's 9..32-row table (``wide_kernel_table``) for the sites it names.
    Every module is checked before any is re-classed, so a refusal leaves the model as it was.
    Only the weights' shapes are read, so this runs on a model whose weights are still on the
    meta device.

    Raises:
        ValueError: if the low-M dispatch is off (``TRTLLM_LOW_M_GEMM_BACKEND``), so
            the built-in ``Linear`` never asks a dispatcher; if no ``kernels`` are given
            and the device and TP size have no table (``kernel_steps``); if ``prefetch`` is asked
            without the sharded projections (``sharded_projections``), whose
            pooler GEMM would read the whole replicated weight; if ``wide`` is asked where
            ``wide_kernel_table`` refuses (no table for the device and TP size, slices
            without the sharded projections, a cell off its 8-row ``split_k``); if a
            projection cannot be re-classed (see
            ``MuseSparkOpenTepDenseLinear.check``); or if a named kernel cannot
            serve its site's shape.
        TypeError: if a projection is not the built-in ``Linear``.
    """
    if not _low_m_gemm.LOW_M_GEMM_ACTIVE:
        raise ValueError(
            "dense_gemms serves the low-M GEMM dispatch, which TRTLLM_LOW_M_GEMM_BACKEND leaves off"
        )
    capability = device_capability()
    sharded = sharded_projections(model)
    tp_size = int(model.layers[0].feed_forward.mapping.tp_size)
    kernels = default_kernels(tp_size, sharded, capability) if kernels is None else dict(kernels)
    if prefetch and not sharded:
        raise ValueError(
            "dense_gemm_prefetch prefetches the rank's pooler slice and needs the sharded projections "
            "(MUSESPARK_OPEN_SHARDED_PROJECTIONS=1); without them the pooler GEMM reads the "
            "whole replicated weight"
        )
    wide_table = wide_kernel_table(capability, tp_size, sharded, kernels) if wide else {}
    batch_table: Mapping[str, Tuple[Optional[str], ...]] = {}
    if _BATCH_SPLITK:
        if (tuple(capability), tp_size) in BATCH_KERNELS_BY_SM_TP and sharded:
            batch_table = batch_kernel_table(capability, tp_size, sharded, kernels)
        else:
            logger.warning(
                "[musespark_open] tep8_batch_splitk is measured for SM (10, 3) at "
                f"TP8 with the sharded pooler and unpooler; SM {tuple(capability)} at TP{tp_size} "
                f"({'sharded' if sharded else 'replicated'}) keeps the built-in calls"
            )
    attn_table: Mapping[str, Tuple[Optional[str], ...]] = {}
    if _ATTN_BLOCK_SPLITK:
        if (tuple(capability), tp_size) in ATTN_BLOCK_KERNELS_BY_SM_TP and sharded:
            attn_table = attn_block_kernel_table(capability, tp_size, sharded)
        else:
            logger.warning(
                "[musespark_open] tep8_attn_block_splitk has no table for SM "
                f"{tuple(capability)} at TP{tp_size} "
                f"({'sharded' if sharded else 'replicated'}); the served calls run"
            )
    plan = DenseGemmPlan(
        kernels,
        prefetch=prefetch,
        enable_pdl=LAUNCH_PDL,
        capability=capability,
        wide=wide_table,
        attn_block=attn_table,
    )
    plan.tp_size = tp_size
    if batch_table:
        plan.set_batch_splitk_kernels(batch_table)

    layers = list(model.layers)
    planned = []
    for index, layer in enumerate(layers):
        attention, moe = layer.self_attn, layer.feed_forward
        weights = {
            "qkv": lambda a=attention: a.qkv_proj.weight,
            "o_proj": lambda a=attention: a.o_proj.weight,
            "pool": _projection_weight(moe, moe.pooler, sharded),
            "unpool": _projection_weight(moe, moe.unpooler, sharded),
        }
        modules = {
            "qkv": attention.qkv_proj,
            "o_proj": attention.o_proj,
            "pool": moe.pooler,
            "unpool": moe.unpooler,
        }
        for role in SITES:
            where = f"layer {index} {role}"
            MuseSparkOpenTepDenseLinear.check(modules[role], where)
            # Shapes only: the loader builds the model under MetaInitMode, whose meta
            # weights refuse any other operation (a slice among them).
            n, k = (int(d) for d in modules[role].weight.shape)
            if sharded and role in ("pool", "unpool"):
                n //= moe.mapping.tp_size
            shape = (k, n)
            for rows, kernel in enumerate(kernels[role], start=1):
                error = kernel_error(kernel, rows, *shape)
                if error is not None:
                    raise ValueError(
                        f"{where}: kernel {kernel} cannot serve {rows} x {shape}: {error}"
                    )
            for rows, kernel in enumerate(wide_table.get(role, ()), start=MAX_ROWS + 1):
                error = kernel_error(kernel, rows, *shape)
                if error is not None:
                    raise ValueError(
                        f"{where}: kernel {kernel} cannot serve {rows} x {shape}: {error}"
                    )
            planned.append((index, role, modules[role], shape, weights))

    next_of = {"qkv": "o_proj", "o_proj": "pool"}
    for index, role, linear, shape, weights in planned:
        if role in next_of:
            next_weight = weights[next_of[role]]
        elif role == "unpool" and index + 1 < len(layers):
            following = layers[index + 1].self_attn
            next_weight = lambda a=following: a.qkv_proj.weight  # noqa: E731
        else:
            next_weight = None
        site = _Site(plan, role, index, shape, next_weight)
        MuseSparkOpenTepDenseLinear.adopt(linear, DenseGemmDispatcher(site))

    logger.info(
        f"[musespark_open] dense_gemms: dense projections of all {len(layers)} layers (SM {capability}, "
        f"TP{tp_size}, "
        f"{'sharded' if sharded else 'replicated'} pooler/unpooler) on calls of at most "
        f"{MAX_ROWS} rows by {describe(kernels)}; "
        f"L2 prefetch of the next weight {'on' if prefetch else 'off'}"
    )
    if wide_table:
        table = WIDE_KERNELS_BY_SM_TP[(tuple(capability), tp_size)]
        steps = "; ".join(
            f"{site}=" + ", ".join(f"{k}@..{last}" for last, k in table[site]) for site in table
        )
        builtin = ", ".join(site for site in SITES if site not in table)
        logger.info(
            f"[musespark_open] wide_dense_gemms: calls of {MAX_ROWS + 1}..{WIDE_MAX_ROWS} rows by "
            f"{steps}; each row gets the bits of its {MAX_ROWS}-row kernel"
            + (f"; {builtin} keep the built-in dispatch at these rows" if builtin else "")
        )
    return plan
