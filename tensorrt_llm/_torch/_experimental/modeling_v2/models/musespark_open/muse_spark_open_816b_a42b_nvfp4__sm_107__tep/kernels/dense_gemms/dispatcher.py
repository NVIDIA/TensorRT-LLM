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
unchanged. ``wide_dense_gemms`` also gives the plan a table for 9..32 rows (``DenseGemmPlan.wide``):
a call of those rows at a site the table names runs that table's kernel. Nothing else in the layer
changes, so the other fast paths
that re-class the attention, the decoder layer or the MoE compose with this one:
their GEMM calls still land in these dispatchers.

Above 32 rows the built-in ``Linear`` never asks a low-M dispatcher, so a fast path
that acts on larger decode steps arms the plan instead (``DenseGemmPlan.batch``, set
for one decode forward by ``batch_dense_gemms``): while it is set, a re-classed
projection's ``apply_linear`` hands the call to it with the built-in GEMM as a callable.

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
from typing import Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

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
    POOL_SHAPE,
    QKV_SHAPE,
    SPLITK,
    UNPOOL_SHAPE,
    WIDE_MAX_ROWS,
    dense_gemm,
    kernel_error,
    l2_prefetch,
    parse_kernel,
)

#: The first row count of a plan's wide table (``DenseGemmPlan.wide``).
WIDE_MIN_ROWS = MAX_ROWS + 1

#: ``tep4_narrow_gemms`` (``configure``): site -> ``[[last row count, kernel], ...]`` replacing tep4
#: sites' 1..8-row steps (``kernel_steps``); None keeps the measured tables below.
_TEP4_NARROW_GEMMS: Optional[Mapping] = None

#: The four projections of a layer, in the order a step runs them.
SITES = ("qkv", "o_proj", "pool", "unpool")

#: ``pool_splitk8`` (``configure``) lets the pooler's ``wide_dense_gemms`` kernels (9..32 rows) sum
#: K in 8 slices (split_k 8, which the split-K kernel serves only with its reduce-scatter epilogue,
#: ``splitk_gemm``'s default ``rs_split_k=(8,)``) instead of the 8-row cell's 4: a wide call's pool
#: rows then no longer get their one-request bits (numerics-gated, not bitwise). Every other site,
#: and the pooler with it off, keeps ``install``'s same-split_k check. Read at install.
POOL_SPLITK8 = False

#: The compute capability the kernel tables below are measured on (sm107, 212 SMs).
CAPABILITY: Tuple[int, int] = (10, 7)

#: Kernel per site by row count at tep8 (two trays, sharded pooler and unpooler), as
#: ``(last row count, kernel)`` steps, chosen in 62-layer chains of cold weights behind
#: predecessors that release at exit: the direct kernel at two or four outputs per CTA for QKV
#: at 1-3 rows and for the unpooler slice at 1-7 rows; split-K 4 for QKV from 4 rows (36 x 4
#: CTAs is the widest split-K grid at its width); flashinfer's split-K ``(64, 8, 1, 12)`` for the
#: attention output projection from 3 rows, where it is faster than cuBLAS. The pooler slice runs
#: the direct kernel at 1-6 rows and split-K 4 from 7 rows, within the measurement spread of the
#: fastest.
SM107_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": (
        (1, "direct:256,2"),
        (2, "direct:128,4"),
        (3, "direct:64,2"),
        (MAX_ROWS, "splitk:64,8,4,12"),
    ),
    "o_proj": ((2, "direct:64,4"), (MAX_ROWS, "splitk:64,8,1,12")),
    "pool": ((2, "direct:256,1"), (6, "direct:128,2"), (MAX_ROWS, "splitk:64,8,4,12")),
    "unpool": ((1, "direct:256,2"), (7, "direct:128,2"), (MAX_ROWS, "splitk:64,8,4,12")),
}

#: The tep4 table (one tray), per-rank shapes QKV ``[4608, 8192]``, o_proj ``[8192, 2048]`` and
#: the sharded pooler / unpooler slices ``[1024, 8192]`` / ``[2048, 4096]``, chosen the same way
#: (``SM107_TP4_REPLICATED_PROJECTION_KERNELS`` replaces the pooler and unpooler entries when they
#: are replicated): the direct kernel at 1-3 rows for QKV, 1-2 for the attention output
#: projection, 1-6 for the pooler slice and 1-5 for the unpooler slice; then split-K 2 for QKV,
#: flashinfer's split-K ``(64, 8, 1, 11)`` for the attention output projection (faster than
#: cuBLAS here) and split-K 4 for the slices.
SM107_TP4_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": (
        (1, "direct:256,2"),
        (2, "direct:64,4"),
        (3, "direct:32,2"),
        (MAX_ROWS, "splitk:64,8,2,12"),
    ),
    "o_proj": ((2, "direct:128,4"), (MAX_ROWS, "splitk:64,8,1,11")),
    "pool": ((1, "direct:256,2"), (6, "direct:128,2"), (MAX_ROWS, "splitk:64,8,4,11")),
    "unpool": ((3, "direct:128,2"), (5, "direct:64,2"), (MAX_ROWS, "splitk:64,8,4,10")),
}

#: The tep4 pooler and unpooler on the whole replicated weights ``[4096, 8192]`` /
#: ``[8192, 4096]`` (the built-in model's tep4 projections, which it shards only at tep8): direct
#: at 1-3 and 1-2 rows, then split-K 2 for the pooler and flashinfer's split-K ``(64, 8, 1, 12)``
#: for the unpooler.
SM107_TP4_REPLICATED_PROJECTION_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "pool": (
        (1, "direct:256,2"),
        (2, "direct:64,4"),
        (3, "direct:32,2"),
        (MAX_ROWS, "splitk:64,8,2,11"),
    ),
    "unpool": ((2, "direct:128,2"), (MAX_ROWS, "splitk:64,8,1,12")),
}

#: ``wide_dense_gemms``: the tep4 kernels of the 9..32-row calls of a decode step of 2..4
#: requests, as ``(last row count, kernel)`` steps from 9 rows. Each is flashinfer's split-K with
#: the ``split_k`` of the site's 8-row cell in ``SM107_TP4_KERNELS``, so it sums the same K slices
#: in the same order and every row of a wide call gets the bits that row gets in a one-request
#: step (``install`` checks this). Among those tactics, the fastest in 62-layer chains of cold
#: weights; each beats cuBLAS on its site up to 32 rows, except the attention output projection
#: at 9..16 rows, where cuBLAS is measured slightly faster.
SM107_TP4_WIDE_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((16, "splitk:64,16,2,11"), (WIDE_MAX_ROWS, "splitk:64,32,2,9")),
    "o_proj": ((16, "splitk:64,16,1,10"), (WIDE_MAX_ROWS, "splitk:64,32,1,9")),
    "pool": ((16, "splitk:64,8,4,11"), (WIDE_MAX_ROWS, "splitk:64,16,4,10")),
    "unpool": ((16, "splitk:64,16,4,9"), (WIDE_MAX_ROWS, "splitk:64,32,4,8")),
}


#: ``wide_dense_gemms`` at tep8: the 9..32-row calls of a decode step of 2..4 requests on
#: flashinfer's split-K with the ``split_k`` of the site's 8-row cell in ``SM107_KERNELS`` (4 for QKV
#: and the pooler and unpooler slices, 1 for the attention output projection), so every row of a
#: wide call keeps the bits it gets in a one-request step. Tile widths and stages follow the tep4
#: table.
SM107_TP8_WIDE_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": ((16, "splitk:64,16,4,11"), (WIDE_MAX_ROWS, "splitk:64,32,4,9")),
    "o_proj": ((16, "splitk:64,16,1,10"), (WIDE_MAX_ROWS, "splitk:64,32,1,9")),
    "pool": ((16, "splitk:64,8,4,11"), (WIDE_MAX_ROWS, "splitk:64,16,4,10")),
    "unpool": ((16, "splitk:64,16,4,9"), (WIDE_MAX_ROWS, "splitk:64,32,4,8")),
}

#: ``tep4_wide_gemms`` (``configure``): site -> ``[[last row count, kernel], ...]`` replacing tep4
#: sites of ``SM107_TP4_WIDE_KERNELS``; None keeps that table. ``install`` still checks each
#: kernel's split_k against the site's 8-row cell (same bits).
_TEP4_WIDE_GEMMS: Optional[Mapping] = None


def wide_steps(tp_size: int = 4) -> Dict[str, Tuple[Tuple[int, str], ...]]:
    """The ``(last row count, kernel)`` steps of each ``wide_dense_gemms`` site at ``tp_size``."""
    if tp_size == 4:
        steps = dict(SM107_TP4_WIDE_KERNELS)
        if _TEP4_WIDE_GEMMS:
            for site, cells in _TEP4_WIDE_GEMMS.items():
                if site not in SITES:
                    raise ValueError(
                        f"[musespark_open] dense_gemms tep4_wide_gemms: unknown site {site!r}"
                    )
                steps[site] = tuple((int(last), str(kernel)) for last, kernel in cells)
        return steps
    steps = dict(SM107_TP8_WIDE_KERNELS)
    # The deepest pipeline each tactic's shared memory allows on this device (split_k 4 holds
    # more partials than the tep4 cells' split_k 2, so the tep4 stage counts can overflow).
    return {
        site: tuple((last, _fit_stages(kernel, last, *_TP8_SHAPES[site])) for last, kernel in cells)
        for site, cells in steps.items()
    }


#: The per-rank ``(K, N)`` of each site at tep8 (``ops/gemm/dense_gemm``).
_TP8_SHAPES = {"qkv": QKV_SHAPE, "o_proj": O_PROJ_SHAPE, "pool": POOL_SHAPE, "unpool": UNPOOL_SHAPE}


def _fit_stages(kernel: str, rows: int, k: int, n: int) -> str:
    """``kernel`` with its split-K ``stages`` lowered until flashinfer accepts it at ``rows x K -> N``."""
    family, fields = parse_kernel(kernel)
    if family != SPLITK:
        return kernel
    mma_m, mma_n, split_k, stages = fields
    while (
        stages > 2
        and kernel_error(f"splitk:{mma_m},{mma_n},{split_k},{stages}", rows, k, n) is not None
    ):
        stages -= 1
    return f"splitk:{mma_m},{mma_n},{split_k},{stages}"


def wide_kernels(tp_size: int = 4) -> Dict[str, Tuple[str, ...]]:
    """The kernel of each ``wide_dense_gemms`` site for rows 9..32 (index ``rows - 9``)."""
    out = {}
    for site, steps in wide_steps(tp_size).items():
        kernels, first = [], MAX_ROWS + 1
        for last, kernel in steps:
            kernels += [kernel] * (last - first + 1)
            first = last + 1
        assert len(kernels) == WIDE_MAX_ROWS - MAX_ROWS, steps
        out[site] = tuple(kernels)
    return out


#: The measured table of each TP size on ``CAPABILITY``, with the pooler and unpooler on the
#: rank's slices (``kernel_steps`` swaps in the replicated tep4 cells).
KERNELS_BY_TP: Mapping[int, Mapping[str, Tuple[Tuple[int, str], ...]]] = {
    8: SM107_KERNELS,
    4: SM107_TP4_KERNELS,
}


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
        ValueError: if ``capability`` is not ``CAPABILITY`` or ``KERNELS_BY_TP`` has no table
            for ``tp_size``.
    """
    key = (tuple(capability), int(tp_size))
    table = KERNELS_BY_TP.get(key[1]) if key[0] == CAPABILITY else None
    if table is None:
        raise ValueError(
            f"[musespark_open] dense_gemms: no kernel table for SM {key[0]} at TP {key[1]} "
            f"(supported: SM {CAPABILITY} at TP {sorted(KERNELS_BY_TP)})"
        )
    if key[1] == 4 and not sharded:
        table = {**table, **SM107_TP4_REPLICATED_PROJECTION_KERNELS}
    # ``tep4_narrow_gemms`` (``configure``; site -> [[last row count, kernel], ...] covering rows
    # 1..8) replaces tep4 sites' 1..8-row steps, as ``tep4_wide_gemms`` does for 9..32 rows: deeper
    # split-K stages with the split-K kernel's ``oversized_smem`` (same split_k, same bits).
    narrow = _TEP4_NARROW_GEMMS if key[1] == 4 else None
    if narrow:
        where = "[musespark_open] dense_gemms tep4_narrow_gemms"
        table = dict(table)
        for site, cells in narrow.items():
            if site not in SITES:
                raise ValueError(f"{where}: unknown site {site!r}")
            steps = tuple((int(last), str(kernel)) for last, kernel in cells)
            if not steps or steps[-1][0] != MAX_ROWS:
                raise ValueError(f"{where}: {site} steps must end at {MAX_ROWS} rows")
            old_k = [parse_kernel(k) for _, k in table[site]]
            for _, kernel in steps:
                family, fields = parse_kernel(kernel)
                if family == SPLITK and not any(
                    f == SPLITK and o[2] == fields[2] for f, o in old_k
                ):
                    raise ValueError(
                        f"{where}: {site} {kernel} changes the split_k of the table's split-K "
                        "cells (bits)"
                    )
            table[site] = steps
    return table


def default_kernels(
    tp_size: int = 8, sharded: bool = True, capability: Tuple[int, int] = CAPABILITY
) -> Dict[str, Tuple[str, ...]]:
    """The kernel of each site for rows 1..8 on ``capability`` at ``tp_size`` (``kernel_steps``)."""
    return {
        site: tuple(_by_rows(steps))
        for site, steps in kernel_steps(capability, tp_size, sharded).items()
    }


def device_capability() -> Tuple[int, int]:
    """The compute capability whose kernel table ``install`` uses: the current CUDA device's.

    On a host without CUDA, where no kernel launches (the wiring's unit tests), the
    target's own ``CAPABILITY``.
    """
    if not torch.cuda.is_available():
        return CAPABILITY
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
    ``wide[site][rows - WIDE_MIN_ROWS]`` is the kernel of a call of 9..32 rows (only for the
    sites the wide table names; none by default).
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
    ) -> None:
        self.kernels = dict(kernels)
        #: ``wide_dense_gemms``: the kernel of a site's 9..32-row calls (``wide[site][rows - 9]``);
        #: a site it does not name keeps the built-in dispatch at those rows.
        self.wide = dict(wide or {})
        #: The fast path armed for the current decode forward above 32 rows (``batch_dense_gemms``'s
        #: ``BatchDenseGemms``), whose ``projection`` every re-classed projection's
        #: ``apply_linear`` calls; ``None`` otherwise.
        self.batch = None
        self.prefetch = prefetch
        self.enable_pdl = enable_pdl
        #: The compute capability of the device the plan serves (``install`` reads it once);
        #: what an arch-specific producer policy keys on.
        self.capability = None if capability is None else tuple(capability)
        #: The model's TP size (``install`` sets it); the TP8 wide64 path keys on it.
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
        (the built-in dispatch): ``kernels`` at 1..8 rows; at 9..32 rows the construction's
        ``wide`` table (``wide_dense_gemms``) for the sites it names."""
        if 1 <= rows <= MAX_ROWS:
            return self.kernels[role][rows - 1]
        if not WIDE_MIN_ROWS <= rows <= WIDE_MAX_ROWS:
            return None
        wide = self.wide.get(role)
        if wide is not None:
            return wide[rows - WIDE_MIN_ROWS]
        return None

    def kernel(self, role: str, rows: int) -> Optional[str]:
        """``kernel_for``: the kernel of a ``role`` call of ``rows`` rows, or ``None``."""
        return self.kernel_for(role, rows)

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


#: The site whose weight a site's fork prefetches (the dispatcher's ``next_of``; unpool -> the next layer's QKV).
PREFETCH_CONSUMER = {"qkv": "o_proj", "o_proj": "pool", "unpool": "qkv"}


def dropped_prefetch_consumers(names: Sequence[str] = ()) -> FrozenSet[str]:
    """The consumer sites whose prefetch ``names`` (``drop_weight_prefetch``) drops; empty for none.

    Raises:
        ValueError: for a name that is not ``qkv``, ``o_proj``, ``pool`` or ``all``.
    """
    sites = {name.strip() for name in names if name.strip()}
    if not sites:
        return frozenset()
    if "all" in sites:
        return frozenset(PREFETCH_CONSUMER.values())
    unknown = sites - set(PREFETCH_CONSUMER.values())
    if unknown:
        raise ValueError(
            f"[musespark_open] dense_gemms drop_weight_prefetch: unknown sites {sorted(unknown)}; "
            f"expected names among {sorted(set(PREFETCH_CONSUMER.values()))} or 'all'"
        )
    return frozenset(sites)


#: ``drop_weight_prefetch`` (``configure``) names the consumer sites (``qkv``, ``o_proj``, ``pool``;
#: ``all``) whose ``dense_gemm_prefetch`` fork is not made. The fork's join makes the prefetch
#: kernel a predecessor of the consumer GEMM, and that kernel never triggers early, so the GEMM
#: launches only once every prefetch CTA has exited; at 9..32 rows the prefetch of the next QKV
#: weight (after the unpooler) and of the o_proj weight (after QKV) ends right before the GEMM
#: starts (nsys, TEP4 bs2-3 and TEP8 bs4), so the GEMM's early launch by its predecessor (the
#: boundary's ``entry`` trigger, the combine's after-wait trigger) buys nothing. Dropped only where
#: the consumer's kernel at the call's row count is flashinfer's split-K, which loads its weight
#: before its own wait, and with programmatic dependent launch on -- elsewhere the prefetch is the
#: only early stream of that weight. The prefetch reads an immutable weight and writes nothing
#: anyone reads: same bits either way. ``configure`` resolves the names here
#: (``dropped_prefetch_consumers``); empty by default.
DROPPED_PREFETCH_CONSUMERS: FrozenSet[str] = frozenset()

#: ``prefetch_join`` (``configure``): ``after`` keeps every ``dense_gemm_prefetch`` fork but joins
#: it right after the consumer GEMM's launch instead of before it, wherever this dispatcher itself
#: launches the kernel (its own fast path, or the direct publish into a transport slot): the GEMM no
#: longer waits for the prefetch kernel's exit (it launches at its predecessor's trigger and stages
#: its weight before its wait) and the prefetch still warms the L2 for its post-wait k-tiles; the
#: kernel after the GEMM waits for the prefetch instead. Calls that take any other path join
#: before, as the module does. ``before`` (the default) is the module.
PREFETCH_JOIN_AFTER = False


def _keeps_prefetch(site: "_Site", rows: int) -> bool:
    """Whether ``site``'s call of ``rows`` rows forks its prefetch (the module's rule plus
    ``drop_weight_prefetch``)."""
    plan = site.plan
    if not (plan.prefetch and site.next_weight is not None and rows not in site.skip_prefetch_rows):
        return False
    consumer = PREFETCH_CONSUMER.get(site.role)
    if consumer is None or consumer not in DROPPED_PREFETCH_CONSUMERS or not plan.enable_pdl:
        return True
    kernel = plan.kernel_for(consumer, rows)
    return kernel is None or parse_kernel(kernel)[0] != SPLITK


#: ``tep8_wide64`` (``configure``) runs the 33..64-row TP8 decode calls of the QKV and attention
#: output projections (5..8 requests at K = 7), which the built-in ``Linear`` sends to cuBLAS
#: (``nvjet`` plus a ``splitKreduce`` launch for QKV), on flashinfer's split-K with the ``split_k``
#: of each site's 8-row cell in ``SM107_KERNELS`` (QKV 4, o_proj 1): every row gets the bits it gets
#: in a one-request step, and the kernel stages its weight before its dependency wait. Needs the
#: split-K kernel's ``wide64`` (64-wide tiles, up to 64 rows, conflict-free mailbox). It serves
#: ``TEP8_WIDE64_SERVED_SITES``; the pool and unpool slices at these rows stay with
#: ``projection_publish``, whose census certified cuBLAS (``torch.mm`` into the transport slot) as
#: their producer. ``tep8_wide64_kernels`` (site -> kernel) replaces a site's tactic; it must keep
#: that site's ``split_k``.
TEP8_WIDE64_MIN_ROWS, TEP8_WIDE64_MAX_ROWS = WIDE_MAX_ROWS + 1, 64
#: ``tep8_wide64_min_rows`` (``configure``; the ``tep8_wide64_splitk`` lever's MIN_ROWS) moves
#: ``TEP8_WIDE64_MIN_ROWS`` down from its default ``TEP8_WIDE64_FLOOR`` (33) to as few as
#: ``TEP8_WIDE64_MIN_LIMIT`` rows: the 2..4-request steps' QKV and o_proj then run the wide64 tactics
#: (each site's split_k, as the 9..32-row wide table: the same bits) through ``wide64``, which forks
#: no weight prefetch, as at 33..64 rows. Independently of it, a step the direct hop
#: (``tep8_direct_hop``'s MIN_ROWS) takes below 33 rows publishes its o_proj from ``wide64``'s
#: scatter branch on the served wide64 tactic.
TEP8_WIDE64_FLOOR = TEP8_WIDE64_MIN_ROWS
TEP8_WIDE64_MIN_LIMIT = MAX_ROWS + 1
#: Tactic per site for 33..64 rows (split_k of the 8-row cells); stages are what shared memory
#: holds at 64-wide tiles. Measured (40 and 64 rows, behind a predecessor that releases early): QKV
#: 6.0-6.2 us vs cuBLAS 8.1-8.7 (64-row tiles keep 144 CTAs and stream each weight slab once);
#: o_proj 3.1-3.3 vs 3.4.
SM107_TP8_WIDE64_KERNELS: Mapping[str, str] = {
    "qkv": "splitk:64,64,4,5",
    "o_proj": "splitk:64,64,1,7",
}
#: The sites ``tep8_wide64`` serves.
TEP8_WIDE64_SERVED_SITES: FrozenSet[str] = frozenset({"qkv", "o_proj"})


def _tep8_wide64_kernels(overrides: Optional[Mapping]) -> Dict[str, str]:
    kernels = dict(SM107_TP8_WIDE64_KERNELS)
    for site, kernel in (overrides or {}).items():
        if site not in kernels:
            raise ValueError(
                f"[musespark_open] dense_gemms tep8_wide64_kernels: unknown site {site!r}"
            )
        kernels[site] = str(kernel)
    return kernels


#: The sites and kernels of the TP8 wide64 path (``configure``'s ``tep8_wide64`` and
#: ``tep8_wide64_kernels``); empty by default.
TEP8_WIDE64_SITES: FrozenSet[str] = frozenset()
TEP8_WIDE64_KERNELS: Dict[str, str] = {}

#: ``tep8_wide64_max_rows`` (``configure``; the ``tep8_wide64_splitk`` lever's MAX_ROWS): the last
#: row count of a TP8 QKV call on the wide64 tactic, ``TEP8_WIDE64_MAX_ROWS`` (64) by default. Above
#: 64, up to ``TEP8_WIDE64_WIDE_LIMIT`` (9..16 requests at K = 7, which the built-in ``Linear`` sends
#: to cuBLAS: nvjet split-K plus a splitKreduce launch, ~8.6 us at 96..128 rows), the call runs in ONE
#: launch of the served QKV tactic compiled exactly as its 64-row launch, whose grid's second
#: dimension takes the extra 64-row token tiles (``splitk_gemm.run_splitk_dense_tiles``), as
#: ``projection_publish``'s 65..128-row publication does: split_k 4 (the 8-row cell's) fixes each
#: element's K slices and their reduction order, so every row gets the bits it gets in a 1..64-row
#: call (its one-request bits). ``TEP8_WIDE64_WIDE_SITES`` only: o_proj keeps its 65..128-row paths
#: (hop128 on an armed step, else cuBLAS). ``tep8_wide64_wide_tile`` (WIDE_TILE, output columns x
#: rows, ``TEP8_WIDE64_WIDE_TILE``) moves that launch off the served tile: ``128x64`` (mma_m 128:
#: 18 x 2 x 4 = 144 CTAs at 65..128 rows, where ``64x64`` gives 288 on 212 SMs) or ``64x128``
#: (mma_n 128: one token tile, 36 x 1 x 4 = 144 CTAs, each weight slab read once) or ``128x128``
#: (one token tile, 18 x 1 x split_k CTAs: the activation re-read 18 times instead of 36), its
#: stages lowered to what shared memory holds; same split_k, same bits. On VR the 64-wide tiles lose
#: to cuBLAS at 65..128 rows (W6: 8.5..10.1 us vs 7.0..7.8): they are L2-bound (~13.5 TB/s), the
#: 36 column tiles re-reading the activation (128 rows: 72 MiB of activation for 37.7 MiB of weight).
#: ``tep8_wide64_wide_split_k`` (WIDE_SPLIT_K, ``TEP8_WIDE64_WIDE_SPLIT_K``; 0 = the served tactic's)
#: 8 takes the launch to split_k 8 (the reduce-scatter epilogue): ``128x128`` at split_k 8 is
#: 18 x 8 = 144 CTAs of 512 KiB each, the 64-row launch's grid and bytes per CTA, in one launch with
#: the reduction in-kernel (cuBLAS adds a splitKreduce launch, ~2.4 us at 96..128 rows). A split_k
#: other than the served one sums K in other slices: every row's bits change (numeric, gated).
#: ``2cta`` (``TEP8_WIDE64_WIDE_2CTA``) takes the launch off split-K's 1-CTA family to
#: ``ops/gemm/splitk_gemm_2cta``: tcgen05 ``cta_group::2`` MMA on 128-column x 128-token pair tiles
#: (each CTA stages half the columns and half the tokens: the 64-row launch's bytes per CTA for
#: twice the rows), the K slices reduced over DSMEM in the 1-CTA owner order, the weight slice
#: prefetched into L2 before the PDL wait; at the served split_k 4 every row is bit for bit its
#: 64-row launch's and its 8-row cell's (W14), 36 x 4 = 144 CTAs. It runs the kernel's reduce-scatter
#: epilogue (``TEP8_WIDE64_2CTA_RS``: each K-slice CTA reduces and stores a quarter of the pair tile,
#: the same additions in the same order, so the same bits; 12 A/B stages instead of 8). W14 (vrg-q2rg):
#: 6.66..6.88 us at 65..128 rows against cuBLAS 7.06..8.19 (owner epilogue 7.25..7.63). split_k 4
#: only; WIDE_SPLIT_K 8 is refused with it.
TEP8_WIDE64_WIDE_MAX_ROWS = TEP8_WIDE64_MAX_ROWS
#: (w256: was 128) WIDE_TILE ``2cta`` takes up to 256 rows (17..32 requests): ceil(rows / 128)
#: 128-token pair tiles on the grid, each row the bits of its <= 128-row call at the same split_k.
#: The 1-CTA wide tiles stop at ``TEP8_WIDE64_1CTA_LIMIT``; the PREP_FOLD epilogue at
#: ``TEP8_WIDE64_PREP_FOLD_MAX_ROWS`` (128 by default, one token tile: 129.. rows run the GEMM and the prep
#: apart; ``tep8_wide64_prep_fold_max_rows`` up to 256 folds over both token tiles).
TEP8_WIDE64_WIDE_LIMIT = 4 * TEP8_WIDE64_MAX_ROWS
TEP8_WIDE64_1CTA_LIMIT = 2 * TEP8_WIDE64_MAX_ROWS
TEP8_WIDE64_PREP_FOLD_MAX_ROWS = 2 * TEP8_WIDE64_MAX_ROWS
TEP8_WIDE64_WIDE_SITES: FrozenSet[str] = frozenset({"qkv"})
#: WIDE_TILE's names -> (mma_m, mma_n); ``64x64`` is the served wide64 tile.
TEP8_WIDE64_WIDE_TILES: Mapping[str, Tuple[int, int]] = {
    "64x64": (64, 64),
    "128x64": (128, 64),
    "64x128": (64, 128),
    "128x128": (128, 128),
    "2cta": (128, 128),
}
TEP8_WIDE64_WIDE_TILE = (64, 64)
#: WIDE_TILE ``2cta``: the one launch on ``ops/gemm/splitk_gemm_2cta`` (its pair tile above).
TEP8_WIDE64_2CTA = "2cta"
TEP8_WIDE64_WIDE_2CTA = False
#: The split_k sizes the 2-CTA kernel serves (its cluster (2, split_k) stays within 8 CTAs).
TEP8_WIDE64_2CTA_SPLIT_KS = (2, 4)
#: WIDE_TILE ``2cta`` runs the kernel's reduce-scatter epilogue (bitwise the owner epilogue's output).
TEP8_WIDE64_2CTA_RS = True
#: ... and stages its first weight k-tiles in shared memory before the PDL wait (``prestage``: the
#: same loads, earlier; W14 vrg-q2pg bitwise, a graph of two PDL calls too: 6.99 / 7.22 / 7.40 ->
#: 6.67 / 6.91 / 7.14 us at 64 / 96 / 128 rows).
TEP8_WIDE64_2CTA_PRESTAGE = True
#: ``tep8_wide64_splitk``'s PREP_FOLD (``MUSESPARK_OPEN_LEVER_TEP8_WIDE64_SPLITK__PREP_FOLD=1``): a
#: WIDE_TILE ``2cta`` call armed by ``multi_request_prep_append`` (``DenseGemmDispatcher.prep_fold``)
#: runs ``ops/gemm/splitk_gemm_2cta_prep``: the same GEMM (its token blocks reduce-scattered whole, the
#: same additions in the same order) with the packed decode prep -- q/k norm, RoPE, gate split, E4M3
#: query, FP8 K/V append, FMHA scale words, cu_seqlens -- in its epilogue, the prep's statements on the
#: reduced tile: the prep launch and its dependency edge go. The same bytes as 2cta + the packed prep.
TEP8_WIDE64_PREP_FOLD = False
#: WIDE_SPLIT_K's values: 0 keeps the served tactic's split_k.
TEP8_WIDE64_WIDE_SPLIT_KS = (0, 4, 8)
TEP8_WIDE64_WIDE_SPLIT_K = 0
#: (w256) ``tep8_wide64_splitk``'s WIDE2_SPLIT_K (0: off): the 2cta launch's split_k on calls ABOVE 128 rows
#: (2 token tiles). At split_k 4 such a call is 288 CTAs on 212 SMs (two waves: 13.0..13.5 us at 160..256
#: rows vs 7.0 at 128; qkv_mb.py); split_k 2 is 144 CTAs (9.3..9.5 us). Changes the K-slice order of those
#: rows (order-only, gated); <= 128-row calls keep the served split_k.
TEP8_WIDE64_WIDE2_SPLIT_K = 0
#: ``_wide_tiles_kernel`` per (served kernel, tile, K, N), resolved at install.
_WIDE_TILES_KERNELS: Dict[tuple, str] = {}

#: ``tep8_direct_prefetch_min_rows`` (``configure``; the ``tep8_direct_hop`` lever's
#: PREFETCH_MIN_ROWS, 0 off): on a decode step of at least that many rows (and above
#: ``WIDE_MAX_ROWS``, where the module's own rule forks no prefetch) that the direct publication
#: armed, the QKV call forks the L2 prefetch of its layer's o_proj weight (``fork_direct_prefetch``),
#: as a 1..32-row QKV call does (``dense_gemm_prefetch``), and the o_proj producer's ``plan.join``
#: joins it. Above 32 rows the QKV takes the wide64 split-K or cuBLAS, which fork nothing, so the
#: scatter producer streams its 16 MiB weight slice cold from HBM after its dependency wait: at
#: 9..16 requests its predecessor is the 1.5..2.5 us attention norm-gate on every layer (the CL8 SWA
#: block, behind which the 64-row producer stages its weight before the wait, serves 1..8
#: requests), so the 65..128-row producer's 5.3..6.4 us (bs9..16) is mostly cold weight streaming.
#: The prefetch reads an immutable weight and writes nothing anyone reads: same bits.
DIRECT_PREFETCH_MIN_ROWS = 0
DIRECT_PREFETCH_LIMIT = 128


def _two_max_rows() -> int:
    """Rows of one 2-CTA token tile (``ops/gemm/splitk_gemm_2cta.MAX_ROWS``, 128)."""
    from ...ops.gemm import splitk_gemm_2cta as _two

    return _two.MAX_ROWS


def _2cta_shape_error(n: int, k: int, split_k: int) -> Optional[str]:
    """Why ``ops/gemm/splitk_gemm_2cta`` cannot take an ``N x K`` weight at ``split_k``, or None
    (its pair tile is 128 output columns, its k-tile 64)."""
    if int(split_k) not in TEP8_WIDE64_2CTA_SPLIT_KS:
        return f"split_k {split_k} not in {list(TEP8_WIDE64_2CTA_SPLIT_KS)}"
    if int(n) % 128 or int(k) % (64 * int(split_k)):
        return f"N {n} is not whole 128-column pair tiles or K {k} not whole 64 x {split_k} slices"
    return None


def _wide_tiles_kernel(kernel: str, tile: Tuple[int, int], k: int, n: int, split_k: int = 0) -> str:
    """The kernel of ``tep8_wide64_max_rows``'s one launch: ``kernel`` (a site's served wide64
    tactic) on ``tile`` = (mma_m, mma_n), at ``split_k`` (0: the served tactic's), its stages lowered
    until the 64-row call accepts it.

    ``install`` resolves it (``_WIDE_TILES_KERNELS``), so a bad one fails there, not at a step.

    Raises:
        ValueError: ``kernel`` is not a split-K kernel, or no stage count fits.
    """
    family, fields = parse_kernel(kernel)
    if family != SPLITK:
        raise ValueError(
            f"[musespark_open] tep8_wide64_splitk MAX_ROWS: {kernel} is not a split-K kernel"
        )
    wide = _fit_stages(
        f"splitk:{tile[0]},{tile[1]},{int(split_k) or fields[2]},{fields[3]}",
        TEP8_WIDE64_MAX_ROWS,
        k,
        n,
    )
    error = kernel_error(wide, TEP8_WIDE64_MAX_ROWS, k, n)
    if error is not None:
        raise ValueError(f"[musespark_open] tep8_wide64_splitk MAX_ROWS: {wide}: {error}")
    return wide


def configure(
    *,
    pool_splitk8: bool = False,
    tep4_wide_gemms: Optional[Mapping] = None,
    tep4_narrow_gemms: Optional[Mapping] = None,
    tep8_wide64: bool = False,
    tep8_wide64_kernels: Optional[Mapping] = None,
    tep8_wide64_min_rows: int = TEP8_WIDE64_FLOOR,
    tep8_wide64_max_rows: int = TEP8_WIDE64_MAX_ROWS,
    tep8_wide64_wide_tile: str = "64x64",
    tep8_wide64_wide_split_k: int = 0,
    tep8_wide64_wide2_split_k: int = 0,
    tep8_direct_prefetch_min_rows: int = 0,
    prefetch_join: str = "before",
    drop_weight_prefetch: Tuple[str, ...] = (),
    tep8_wide64_prep_fold: bool = False,
    tep8_wide64_prep_fold_max_rows: int = 128,
) -> None:
    """Set the dispatcher's table overrides and prefetch scheduling; the defaults keep the measured
    tables and the module's prefetch rule.

    ``lever_settings.apply`` sets them from the model's plan before any install.

    Args:
        pool_splitk8: the pooler's wide kernels may sum K in 8 slices (``POOL_SPLITK8``).
        tep4_wide_gemms: site -> ``[[last row count, kernel], ...]`` replacing tep4 sites' 9..32-row
            steps (``wide_steps``); None keeps ``SM107_TP4_WIDE_KERNELS``.
        tep4_narrow_gemms: site -> ``[[last row count, kernel], ...]`` ending at 8 rows, replacing
            tep4 sites' 1..8-row steps (``kernel_steps``); None keeps the measured tables.
        tep8_wide64: the 33..64-row TP8 QKV and o_proj calls on split-K (``TEP8_WIDE64_SITES``).
        tep8_wide64_kernels: site -> kernel replacing ``SM107_TP8_WIDE64_KERNELS`` entries.
        tep8_wide64_min_rows: the first row count of those calls (``TEP8_WIDE64_MIN_ROWS``):
            ``TEP8_WIDE64_MIN_LIMIT``..``TEP8_WIDE64_FLOOR``.
        tep8_wide64_max_rows: the last row count of the QKV calls (``TEP8_WIDE64_WIDE_MAX_ROWS``):
            ``TEP8_WIDE64_MAX_ROWS``..``TEP8_WIDE64_WIDE_LIMIT``; above 64 rows one launch.
        tep8_wide64_wide_tile: the tile of that launch (``TEP8_WIDE64_WIDE_TILE``), a name of
            ``TEP8_WIDE64_WIDE_TILES``; ``2cta`` the 2-CTA kernel (``TEP8_WIDE64_WIDE_2CTA``).
        tep8_wide64_wide_split_k: that launch's split_k (``TEP8_WIDE64_WIDE_SPLIT_K``), one of
            ``TEP8_WIDE64_WIDE_SPLIT_KS`` (0: the served tactic's).
        tep8_direct_prefetch_min_rows: the fewest rows of an armed direct-hop step whose QKV call
            forks the o_proj weight's L2 prefetch (``DIRECT_PREFETCH_MIN_ROWS``): 0 (none) ..
            ``DIRECT_PREFETCH_LIMIT``.
        prefetch_join: join the weight prefetch ``before`` or ``after`` the consumer GEMM's launch
            (``PREFETCH_JOIN_AFTER``).
        drop_weight_prefetch: the consumer sites (or ``all``) whose weight prefetch is not forked
            (``DROPPED_PREFETCH_CONSUMERS``).
        tep8_wide64_prep_fold: armed 2-CTA calls fold the multi-request decode prep into their
            epilogue (``TEP8_WIDE64_PREP_FOLD``); needs ``tep8_wide64_wide_tile`` ``2cta`` and a
            ``tep8_wide64_max_rows`` above 64.

    Raises:
        ValueError: for a ``prefetch_join`` other than ``before`` / ``after``, an unknown
            ``drop_weight_prefetch`` site, or, with ``tep8_wide64``, an unknown
            ``tep8_wide64_kernels`` site, or a ``tep8_wide64_min_rows`` outside
            ``TEP8_WIDE64_MIN_LIMIT``..``TEP8_WIDE64_FLOOR``, a ``tep8_wide64_max_rows`` outside
            ``TEP8_WIDE64_MAX_ROWS``..``TEP8_WIDE64_WIDE_LIMIT``, a ``tep8_wide64_wide_tile`` not
            in ``TEP8_WIDE64_WIDE_TILES``, ``2cta`` with a ``tep8_wide64_wide_split_k`` it does not
            serve, a ``tep8_wide64_max_rows`` above 64 without ``tep8_wide64``, or
            ``tep8_wide64_prep_fold`` without ``2cta`` above 64 rows; nothing is set then.
    """
    global POOL_SPLITK8, _TEP4_WIDE_GEMMS, _TEP4_NARROW_GEMMS, DROPPED_PREFETCH_CONSUMERS
    global PREFETCH_JOIN_AFTER, TEP8_WIDE64_SITES, TEP8_WIDE64_KERNELS, TEP8_WIDE64_MIN_ROWS
    global TEP8_WIDE64_WIDE_MAX_ROWS, TEP8_WIDE64_WIDE_TILE, DIRECT_PREFETCH_MIN_ROWS
    global TEP8_WIDE64_PREP_FOLD_MAX_ROWS
    global TEP8_WIDE64_WIDE_SPLIT_K, TEP8_WIDE64_WIDE_2CTA, TEP8_WIDE64_PREP_FOLD
    if int(tep8_wide64_wide_split_k) not in TEP8_WIDE64_WIDE_SPLIT_KS:
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_wide_split_k must be one of "
            f"{list(TEP8_WIDE64_WIDE_SPLIT_KS)}, got {tep8_wide64_wide_split_k}"
        )
    if not 0 <= int(tep8_direct_prefetch_min_rows) <= DIRECT_PREFETCH_LIMIT:
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_direct_prefetch_min_rows must be in "
            f"0..{DIRECT_PREFETCH_LIMIT}, got {tep8_direct_prefetch_min_rows}"
        )
    if not TEP8_WIDE64_MIN_LIMIT <= int(tep8_wide64_min_rows) <= TEP8_WIDE64_FLOOR:
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_min_rows must be in "
            f"{TEP8_WIDE64_MIN_LIMIT}..{TEP8_WIDE64_FLOOR}, got {tep8_wide64_min_rows}"
        )
    if not TEP8_WIDE64_MAX_ROWS <= int(tep8_wide64_max_rows) <= TEP8_WIDE64_WIDE_LIMIT:
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_max_rows must be in "
            f"{TEP8_WIDE64_MAX_ROWS}..{TEP8_WIDE64_WIDE_LIMIT}, got {tep8_wide64_max_rows}"
        )
    if int(tep8_wide64_max_rows) > TEP8_WIDE64_MAX_ROWS and not tep8_wide64:
        raise ValueError(
            "[musespark_open] dense_gemms: tep8_wide64_max_rows above "
            f"{TEP8_WIDE64_MAX_ROWS} needs tep8_wide64"
        )
    if tep8_wide64_wide_tile not in TEP8_WIDE64_WIDE_TILES:
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_wide_tile must be one of "
            f"{list(TEP8_WIDE64_WIDE_TILES)}, got {tep8_wide64_wide_tile!r}"
        )
    if (
        int(tep8_wide64_max_rows) > TEP8_WIDE64_1CTA_LIMIT
        and tep8_wide64_wide_tile != TEP8_WIDE64_2CTA
    ):
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_max_rows above {TEP8_WIDE64_1CTA_LIMIT} "
            f"needs tep8_wide64_wide_tile 2cta, got {tep8_wide64_wide_tile!r}"
        )
    if (
        tep8_wide64_wide_tile == TEP8_WIDE64_2CTA
        and int(tep8_wide64_wide_split_k)
        and int(tep8_wide64_wide_split_k) not in TEP8_WIDE64_2CTA_SPLIT_KS
    ):
        raise ValueError(
            "[musespark_open] dense_gemms: tep8_wide64_wide_tile 2cta serves split_k "
            f"{list(TEP8_WIDE64_2CTA_SPLIT_KS)} (tep8_wide64_wide_split_k 0 = the served 4), got "
            f"tep8_wide64_wide_split_k {tep8_wide64_wide_split_k}"
        )
    if tep8_wide64_prep_fold and not (
        tep8_wide64
        and tep8_wide64_wide_tile == TEP8_WIDE64_2CTA
        and int(tep8_wide64_max_rows) > TEP8_WIDE64_MAX_ROWS
    ):
        raise ValueError(
            "[musespark_open] dense_gemms: MUSESPARK_OPEN_LEVER_TEP8_WIDE64_SPLITK__PREP_FOLD folds "
            "the decode prep into the 2-CTA QKV launch: it needs WIDE_TILE=2cta and MAX_ROWS above "
            f"{TEP8_WIDE64_MAX_ROWS} (got tile {tep8_wide64_wide_tile!r}, max_rows "
            f"{tep8_wide64_max_rows}, tep8_wide64 {bool(tep8_wide64)})"
        )
    if (
        not 2 * TEP8_WIDE64_MAX_ROWS
        <= int(tep8_wide64_prep_fold_max_rows)
        <= TEP8_WIDE64_WIDE_LIMIT
    ):
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_prep_fold_max_rows must be in "
            f"{2 * TEP8_WIDE64_MAX_ROWS}..{TEP8_WIDE64_WIDE_LIMIT}, got {tep8_wide64_prep_fold_max_rows}"
        )
    join = str(prefetch_join).strip() or "before"
    if join not in ("before", "after"):
        raise ValueError(
            f"[musespark_open] dense_gemms: prefetch_join must be 'before' or 'after', got {join!r}"
        )
    dropped = dropped_prefetch_consumers(drop_weight_prefetch)
    sites = TEP8_WIDE64_SERVED_SITES if tep8_wide64 else frozenset()
    kernels = _tep8_wide64_kernels(tep8_wide64_kernels) if sites else {}
    POOL_SPLITK8 = bool(pool_splitk8)
    _TEP4_WIDE_GEMMS = None if tep4_wide_gemms is None else dict(tep4_wide_gemms)
    _TEP4_NARROW_GEMMS = None if tep4_narrow_gemms is None else dict(tep4_narrow_gemms)
    DROPPED_PREFETCH_CONSUMERS = dropped
    PREFETCH_JOIN_AFTER = join == "after"
    TEP8_WIDE64_SITES = sites
    TEP8_WIDE64_KERNELS = kernels
    TEP8_WIDE64_MIN_ROWS = int(tep8_wide64_min_rows)
    TEP8_WIDE64_WIDE_MAX_ROWS = int(tep8_wide64_max_rows)
    DIRECT_PREFETCH_MIN_ROWS = int(tep8_direct_prefetch_min_rows)
    TEP8_WIDE64_WIDE_TILE = TEP8_WIDE64_WIDE_TILES[tep8_wide64_wide_tile]
    TEP8_WIDE64_WIDE_2CTA = tep8_wide64_wide_tile == TEP8_WIDE64_2CTA
    TEP8_WIDE64_WIDE_SPLIT_K = int(tep8_wide64_wide_split_k)
    global TEP8_WIDE64_WIDE2_SPLIT_K
    if int(tep8_wide64_wide2_split_k) not in (0, *TEP8_WIDE64_2CTA_SPLIT_KS):
        raise ValueError(
            f"[musespark_open] dense_gemms: tep8_wide64_wide2_split_k is 0 or one of "
            f"{list(TEP8_WIDE64_2CTA_SPLIT_KS)}, got {tep8_wide64_wide2_split_k}"
        )
    TEP8_WIDE64_WIDE2_SPLIT_K = int(tep8_wide64_wide2_split_k)
    TEP8_WIDE64_PREP_FOLD = bool(tep8_wide64_prep_fold)
    # w256: the fold over 2 token tiles (129..256 rows) only when asked (PREP_FOLD_MAX_ROWS).
    TEP8_WIDE64_PREP_FOLD_MAX_ROWS = int(tep8_wide64_prep_fold_max_rows)


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
    families storing into the slot (flashinfer's split-K and direct kernels) and everything else
    copied.

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
        #: While set (``TEP8_WIDE64_PREP_FOLD``: by ``multi_request_prep_append`` for one decode
        #: forward of its layer), a WIDE_TILE ``2cta`` call takes it and, unless it declines the
        #: call (``operands``), runs the prep-fold kernel and hands it the prep's outputs
        #: (``finish``). ``None`` otherwise.
        self.prep_fold: Optional[object] = None

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
        deferred = PREFETCH_JOIN_AFTER and self._launches_itself(input_tensor, weight, bias)
        if not deferred:
            plan.join()
        if plan.named_forks:
            plan.join_role(self.site.role)
        if self.substitute is not None:
            if deferred:
                # No GEMM to launch first: the deferred join runs now.
                plan.join()
            return self._apply_substitute(input_tensor, weight)
        if self.publish is not None:
            return self._apply_publish(module, input_tensor, weight, bias, force_active, deferred)
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
        if deferred:
            plan.join()
        if _keeps_prefetch(self.site, rows):
            plan.fork_prefetch(self.site.next_weight())
        return out.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def wide64(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, bias
    ) -> Optional[torch.Tensor]:
        """A 33..64-row TP8 decode call of a ``TEP8_WIDE64_SITES`` site on flashinfer's split-K (QKV up
        to ``TEP8_WIDE64_WIDE_MAX_ROWS``: ``_wide_tiles``), or ``None`` (the caller then runs the
        built-in ``Linear.apply_linear``, i.e. cuBLAS)."""
        site = self.site
        plan = site.plan
        if site.role not in TEP8_WIDE64_SITES or plan.tp_size != 8 or bias is not None:
            return None
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return None
        if not (
            input_tensor.is_cuda
            and input_tensor.dtype == torch.bfloat16
            and weight.dtype == torch.bfloat16
            and weight.dim() == 2
            and input_tensor.dim() >= 1
            and (int(weight.shape[1]), int(weight.shape[0])) == site.shape
            and int(input_tensor.shape[-1]) == int(weight.shape[1])
            and input_tensor.is_contiguous()
            and weight.is_contiguous()
            and input_tensor.data_ptr() % 32 == 0
            and weight.data_ptr() % 32 == 0
        ):
            return None
        k = int(weight.shape[1])
        rows = input_tensor.numel() // k
        if TEP8_WIDE64_MAX_ROWS < rows <= 4 * TEP8_WIDE64_MAX_ROWS and site.role == "o_proj":
            # hop128 (``tep8_direct_hop``'s MAX_ROWS 128; w256: 256): only a step the direct
            # publication armed; every other 65..256-row call keeps cuBLAS, as before.
            return self._direct128(input_tensor, weight, rows, k)
        if rows > TEP8_WIDE64_MAX_ROWS:
            if rows <= TEP8_WIDE64_WIDE_MAX_ROWS and site.role in TEP8_WIDE64_WIDE_SITES:
                # ``tep8_wide64_max_rows``: 65..MAX_ROWS-row QKV in one launch (every row's bits).
                return self._wide_tiles(input_tensor, weight, rows, k)
            return None
        if rows < TEP8_WIDE64_MIN_ROWS and not (
            # ``tep8_direct_hop``'s MIN_ROWS below the wide64 band: an armed step's o_proj still
            # scatters (below); every other call keeps the 9..32-row wide table.
            site.role == "o_proj" and rows > MAX_ROWS and self._direct_target(rows) is not None
        ):
            return None
        plan.join()
        if plan.named_forks:
            plan.join_role(site.role)
        if site.role == "o_proj":
            # On a decode step whose wide post-attention hop runs the direct transport, this layer's
            # o_proj stores its partial straight into the owners' scatter slots
            # (post_attn_hop_wide.DirectPublication): the scatter epilogue of the same split-K
            # kernel, same bits.
            from ...ops.allreduce import post_attn_hop_wide as _wide

            publication = getattr(_wide, "_DIRECT_PUBLICATION", None)
            target = self._direct_target(rows)
            if target is not None:
                view, peer_stride, fb_layout = target
                if rows < TEP8_WIDE64_FLOOR:
                    logger.info_once(
                        f"[musespark_open] tep8_direct_hop: o_proj of {_wide.SERVED_DIRECT_ROWS[0]}.."
                        f"{TEP8_WIDE64_FLOOR - 1}-row steps scatters on the served wide64 tactic "
                        f"{TEP8_WIDE64_KERNELS[site.role]} (direct hop MIN_ROWS)",
                        key="musespark_open.tep8_direct_hop_low_rows",
                    )
                family, fields = parse_kernel(TEP8_WIDE64_KERNELS[site.role])
                if family != "splitk" or fields[2] != 1:
                    raise RuntimeError(
                        "[musespark_open] TEP8 direct hop: o_proj runs "
                        f"{TEP8_WIDE64_KERNELS[site.role]}, not a split_k 1 split-K kernel"
                    )
                _wide.run_splitk_scatter(
                    input_tensor.view(-1, k),
                    weight.t(),
                    view,
                    plan.enable_pdl,
                    _wide.SplitKTactic(*fields),
                    peer_stride,
                    fb_layout,
                    # The workspace's slots per source (16 on hop128's 128-row workspace).
                    publication.target_slots(),
                )
                return view.view(*input_tensor.shape[:-1], int(weight.shape[0]))
        out = dense_gemm(
            input_tensor.view(-1, k), weight, TEP8_WIDE64_KERNELS[site.role], plan.enable_pdl
        )
        return out.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def _direct_target(self, rows: int):
        """``DirectPublication.target`` of this dispatcher's call of ``rows`` rows (the armed step's
        scatter view), or None (no direct publication, not armed, or not this step's rows)."""
        from ...ops.allreduce import post_attn_hop_wide as _wide

        publication = getattr(_wide, "_DIRECT_PUBLICATION", None)
        return publication.target(self, rows) if publication is not None else None

    def fork_direct_prefetch(self, input_tensor: torch.Tensor, weight: torch.Tensor) -> bool:
        """``DIRECT_PREFETCH_MIN_ROWS``: after a QKV call of a decode step of at least that many rows
        (above ``WIDE_MAX_ROWS``) that the direct publication armed, fork the L2 prefetch of the
        layer's o_proj weight (``DenseGemmPlan.fork_prefetch``; the o_proj producer joins it).
        Returns whether it forked."""
        site = self.site
        if site.role != "qkv" or site.next_weight is None or not DIRECT_PREFETCH_MIN_ROWS:
            return False
        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return False
        rows = input_tensor.numel() // max(int(weight.shape[-1]), 1)
        if rows <= WIDE_MAX_ROWS or rows < DIRECT_PREFETCH_MIN_ROWS:
            return False
        from ...ops.allreduce import post_attn_hop_wide as _wide

        publication = getattr(_wide, "_DIRECT_PUBLICATION", None)
        if publication is None or getattr(publication, "rows", None) != rows:
            return False
        site.plan.fork_prefetch(site.next_weight())
        logger.info_once(
            "[musespark_open] tep8_direct_hop PREFETCH_MIN_ROWS: the QKV call of armed "
            f"{max(DIRECT_PREFETCH_MIN_ROWS, WIDE_MAX_ROWS + 1)}..{getattr(_wide, 'DIRECT_MAX_ROWS', 64)}-row "
            "steps forks the L2 prefetch of the layer's o_proj weight (joined by the scatter producer)",
            key="musespark_open.tep8_direct_prefetch",
        )
        return True

    def _wide_tiles(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, rows: int, k: int
    ) -> torch.Tensor:
        """``tep8_wide64_max_rows``: a 65..``TEP8_WIDE64_WIDE_MAX_ROWS``-row call of a
        ``TEP8_WIDE64_WIDE_SITES`` site in ONE launch of its served wide64 tactic on
        ``TEP8_WIDE64_WIDE_TILE`` tiles, the token tiles on the grid's second dimension
        (``splitk_gemm.run_splitk_dense_tiles``): each row gets the bits it gets in a 1..64-row call
        (the same split_k). Like the 33..64-row calls, it joins and forks no weight prefetch.
        WIDE_TILE ``2cta``: ``_wide_2cta`` instead."""
        if TEP8_WIDE64_WIDE_2CTA:
            return self._wide_2cta(input_tensor, weight, rows, k)
        from ...ops.gemm import splitk_gemm as _splitk

        site = self.site
        plan = site.plan
        n = int(weight.shape[0])
        key = (
            TEP8_WIDE64_KERNELS[site.role],
            TEP8_WIDE64_WIDE_TILE,
            k,
            n,
            TEP8_WIDE64_WIDE_SPLIT_K,
        )
        kernel = _WIDE_TILES_KERNELS.get(key)
        if kernel is None:
            kernel = _WIDE_TILES_KERNELS[key] = _wide_tiles_kernel(*key)
        plan.join()
        if plan.named_forks:
            plan.join_role(site.role)
        out = torch.empty((rows, n), dtype=input_tensor.dtype, device=input_tensor.device)
        _splitk.run_splitk_dense_tiles(
            input_tensor.view(-1, k),
            weight.t(),
            out,
            plan.enable_pdl,
            _splitk.SplitKTactic(*parse_kernel(kernel)[1]),
        )
        fields = parse_kernel(kernel)[1]
        same = fields[2] == parse_kernel(TEP8_WIDE64_KERNELS[site.role])[1][2]
        logger.info_once(
            f"[musespark_open] tep8_wide64_splitk: {site.role} of {TEP8_WIDE64_MAX_ROWS + 1}.."
            f"{TEP8_WIDE64_WIDE_MAX_ROWS}-row TP8 decode steps in one launch of {kernel} "
            f"({fields[1]}-row token tiles, "
            + (
                f"the {TEP8_WIDE64_MAX_ROWS}-row call's bits"
                if same
                else f"split_k {fields[2]}: numeric, not the {TEP8_WIDE64_MAX_ROWS}-row call's bits"
            )
            + "; MAX_ROWS)",
            key=f"musespark_open.tep8_wide64_max_rows.{site.role}",
        )
        return out.view(*input_tensor.shape[:-1], n)

    def _wide_2cta(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, rows: int, k: int
    ) -> torch.Tensor:
        """WIDE_TILE ``2cta``: the 65..``TEP8_WIDE64_WIDE_MAX_ROWS``-row call in one launch of
        ``ops/gemm/splitk_gemm_2cta`` at the served tactic's split_k (or WIDE_SPLIT_K): each row the
        bits of its 1..64-row call at that split_k. A build or launch failure raises (no fallback)."""
        site = self.site
        plan = site.plan
        n = int(weight.shape[0])
        split_k = TEP8_WIDE64_WIDE_SPLIT_K or parse_kernel(TEP8_WIDE64_KERNELS[site.role])[1][2]
        if TEP8_WIDE64_WIDE2_SPLIT_K and rows > _two_max_rows():
            split_k = TEP8_WIDE64_WIDE2_SPLIT_K
            logger.info_once(
                f"[musespark_open] tep8_wide64_splitk WIDE2_SPLIT_K: {_two_max_rows() + 1}..-row calls of the 2cta "
                f"launch at split_k {split_k}",
                key="musespark_open.tep8_wide64_wide2_split_k",
            )
        error = _2cta_shape_error(n, k, split_k)
        if error is not None:
            raise ValueError(f"[musespark_open] tep8_wide64_splitk WIDE_TILE 2cta: {error}")
        from ...ops.gemm import splitk_gemm_2cta as _two

        fold = self.prep_fold
        self.prep_fold = None
        plan.join()
        if plan.named_forks:
            plan.join_role(site.role)
        out = torch.empty((rows, n), dtype=input_tensor.dtype, device=input_tensor.device)
        if fold is not None:
            if not (TEP8_WIDE64_PREP_FOLD and TEP8_WIDE64_2CTA_RS):
                raise RuntimeError(
                    "[musespark_open] tep8_wide64_splitk: a prep fold armed with "
                    "MUSESPARK_OPEN_LEVER_TEP8_WIDE64_SPLITK__PREP_FOLD off or the owner epilogue"
                )
            operands = fold.operands(out)
            if operands is not None:
                from ...ops.gemm import splitk_gemm_2cta_prep as _fold

                _fold.run_splitk_2cta_prep_fold(
                    input_tensor.view(-1, k),
                    weight,
                    out,
                    plan.enable_pdl,
                    split_k,
                    TEP8_WIDE64_2CTA_PRESTAGE,
                    **operands,
                )
                fold.finish(out)
                logger.info_once(
                    f"[musespark_open] tep8_wide64_splitk PREP_FOLD: {TEP8_WIDE64_MAX_ROWS + 1}.."
                    f"{TEP8_WIDE64_WIDE_MAX_ROWS}-row multi-request decode steps run the QKV "
                    "projection and the packed decode prep (q/k norm, RoPE, gate split, E4M3 query, "
                    "FP8 K/V append, FMHA scales) in one launch of the 2-CTA split-K kernel "
                    f"(ops/gemm/splitk_gemm_2cta_prep, split_k {split_k}; the same bytes)",
                    key="musespark_open.tep8_wide64_prep_fold",
                )
                return out.view(*input_tensor.shape[:-1], n)
        _two.run_splitk_2cta(
            input_tensor.view(-1, k),
            weight,
            out,
            plan.enable_pdl,
            split_k,
            rs=TEP8_WIDE64_2CTA_RS,
            prestage=TEP8_WIDE64_2CTA_PRESTAGE,
        )
        served = parse_kernel(TEP8_WIDE64_KERNELS[site.role])[1][2]
        logger.info_once(
            f"[musespark_open] tep8_wide64_splitk: {site.role} of {TEP8_WIDE64_MAX_ROWS + 1}.."
            f"{TEP8_WIDE64_WIDE_MAX_ROWS}-row TP8 decode steps in one launch of the 2-CTA split-K "
            f"kernel (ops/gemm/splitk_gemm_2cta, "
            f"{'reduce-scatter' if TEP8_WIDE64_2CTA_RS else 'owner'} epilogue, split_k {split_k}: "
            + (
                f"the {TEP8_WIDE64_MAX_ROWS}-row call's bits"
                if split_k == served
                else f"numeric, not the {TEP8_WIDE64_MAX_ROWS}-row call's bits"
            )
            + "; WIDE_TILE 2cta)",
            key=f"musespark_open.tep8_wide64_2cta.{site.role}",
        )
        return out.view(*input_tensor.shape[:-1], n)

    def _direct128(
        self, input_tensor: torch.Tensor, weight: torch.Tensor, rows: int, k: int
    ) -> Optional[torch.Tensor]:
        """hop128: o_proj of a 65..128-row step the direct publication armed, on the split_k-1
        scatter kernel (``post_attn_hop_wide.DIRECT128_PRODUCER``): ``chunks`` as 64-row launches of
        the served tactic (each chunk's tokens land in their own slots: chunk c through the view
        advanced by 8 c slots), in token order on the current stream (PDL chains them); ``grid`` /
        ``tile128`` as one launch of 64- or 128-token tiles. ``None`` (cuBLAS, as before) when not
        armed."""
        from ...ops.allreduce import post_attn_hop_wide as _wide

        site = self.site
        plan = site.plan
        publication = getattr(_wide, "_DIRECT_PUBLICATION", None)
        if publication is None or not getattr(_wide, "DIRECT128", False):
            return None
        target = publication.target(self, rows)
        if target is None:
            return None
        view, peer_stride, fb_layout = target
        family, fields = parse_kernel(TEP8_WIDE64_KERNELS[site.role])
        if family != "splitk" or fields[2] != 1:
            raise RuntimeError(
                f"[musespark_open] hop128: o_proj runs {TEP8_WIDE64_KERNELS[site.role]}, not a "
                "split_k 1 split-K kernel"
            )
        plan.join()
        if plan.named_forks:
            plan.join_role(site.role)
        a = input_tensor.view(-1, k)
        b = weight.t()
        slots = publication.target_slots()
        mode, stages = _wide.DIRECT128_PRODUCER
        if mode == "2cta" and rows > 128:
            import os as _os

            # B4 is banked at 65..128 rows only; above 128 rows the served one-launch producer (grid) unless
            # MUSESPARK_OPEN_XCHG_2CTA_WIDE=1 (B4 at 129..256 rows too: same split_k-1 bits, unmeasured).
            if _os.environ.get("MUSESPARK_OPEN_XCHG_2CTA_WIDE", "0") != "1":
                mode, stages = "grid", _wide.DIRECT128_PRODUCER_STAGES["grid"]
        if mode == "2cta":
            if not fb_layout:
                raise RuntimeError(
                    "[musespark_open] hop128 producer 2cta: needs the fb-layout workspace"
                )
            import os as _os

            from ...ops.gemm import splitk_gemm_2cta as _two

            # XCHG_2CTA_NOPRE=1 (measurement): no whole-slice L2 prefetch and no pre-wait prestage.
            pre = _os.environ.get("XCHG_2CTA_NOPRE", "0") != "1"
            _two.run_splitk_2cta_scatter(
                a.contiguous(),
                weight,
                publication.chunk_view(view, 0, rows),
                plan.enable_pdl,
                peer_stride,
                slots,
                pre,
                pre,
            )
            logger.info_once(
                f"[musespark_open] hop128 producer: o_proj of {TEP8_WIDE64_MAX_ROWS + 1}.."
                f"{_wide.DIRECT_MAX_ROWS}-row steps in one 2-CTA split-K 1 scatter launch (B4)",
                key="musespark_open.hop128_producer",
            )
            return view.view(*input_tensor.shape[:-1], int(weight.shape[0]))
        if mode != "chunks":
            if not fb_layout:
                raise RuntimeError(
                    f"[musespark_open] hop128 producer {mode}: needs the fb-layout workspace"
                )
            _wide.run_splitk_scatter_wide(
                a,
                b,
                publication.chunk_view(view, 0, rows),
                plan.enable_pdl,
                _wide.hop128_tactic(mode, stages),
                peer_stride,
                slots,
            )
            logger.info_once(
                f"[musespark_open] hop128 producer: o_proj of {TEP8_WIDE64_MAX_ROWS + 1}.."
                f"{_wide.DIRECT_MAX_ROWS}-row steps in one {mode} launch ({stages} stages)",
                key="musespark_open.hop128_producer",
            )
            return view.view(*input_tensor.shape[:-1], int(weight.shape[0]))
        tactic = _wide.SplitKTactic(*fields)
        for row0 in range(0, rows, TEP8_WIDE64_MAX_ROWS):
            n = min(TEP8_WIDE64_MAX_ROWS, rows - row0)
            _wide.run_splitk_scatter(
                a[row0 : row0 + n],
                b,
                publication.chunk_view(view, row0, n),
                plan.enable_pdl,
                tactic,
                peer_stride,
                fb_layout,
                slots,
            )
        return view.view(*input_tensor.shape[:-1], int(weight.shape[0]))

    def _launches_itself(self, input_tensor, weight, bias) -> bool:
        """Whether this call launches the plan's kernel itself: it fits, and when a transport slot is armed
        the kernel's family stores into it (``_apply_publish``'s direct branch)."""
        if not self.fits(input_tensor, weight, bias):
            return False
        if self.publish is None:
            return True
        policy = self.publish_policy
        k = int(weight.shape[1])
        rows = input_tensor.numel() // max(k, 1)
        kernel = self.site.plan.kernel_for(self.site.role, rows)
        return (
            policy is not None
            and kernel is not None
            and rows == int(self.publish.shape[0])
            and parse_kernel(kernel)[0] in policy.direct
        )

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

    def _apply_publish(
        self,
        module: torch.nn.Module,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        force_active: bool,
        deferred: bool = False,
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
            if deferred:
                plan.join()
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
            out = None
            if TEP8_WIDE64_SITES and not lora:
                out = dispatcher.wide64(input, self.weight, bias)
            if out is None:
                out = Linear.apply_linear(self, input, bias, lora_params, layer_idx)
            if DIRECT_PREFETCH_MIN_ROWS and not lora:
                # After the call's GEMM is on the stream: the prefetch overlaps attention.
                dispatcher.fork_direct_prefetch(input, self.weight)
            return out
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
    size and its projection layout (``default_kernels``). ``wide`` (``wide_dense_gemms``,
    tep4 or tep8 with the sharded projections) adds ``wide_kernels`` for the 9..32-row calls. Every module is checked
    before any is re-classed, so a refusal
    leaves the model as it was. Only the weights' shapes are read, so this runs
    on a model whose weights are still on the meta device.

    Raises:
        ValueError: if the low-M dispatch is off (``TRTLLM_LOW_M_GEMM_BACKEND``), so
            the built-in ``Linear`` never asks a dispatcher; if no ``kernels`` are given
            and the device and TP size have no table (``kernel_steps``); if ``prefetch`` is asked
            without the sharded projections (``sharded_projections``), whose
            pooler GEMM would read the whole replicated weight; if a
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
    wide_table = {}
    if wide:
        if tuple(capability) != CAPABILITY or tp_size not in (4, 8) or not sharded:
            raise ValueError(
                "[musespark_open] wide_dense_gemms is measured for sm107 at TP4 and TP8 with the sharded "
                f"pooler and unpooler; got SM {tuple(capability)} at TP{tp_size}, "
                f"{'sharded' if sharded else 'replicated'} projections"
            )
        wide_table = wide_kernels(tp_size)
        pool8 = POOL_SPLITK8
        if pool8:
            logger.info(
                "[musespark_open] dense_gemms: pool_splitk8: pool wide kernels "
                f"{sorted(set(wide_table.get('pool', ())))} may sum K in 8 slices (not the 8-row cell's bits)"
            )
        for role, wide_cells in wide_table.items():
            family, fields = parse_kernel(kernels[role][MAX_ROWS - 1])
            for kernel in wide_cells:
                wide_family, wide_fields = parse_kernel(kernel)
                if pool8 and role == "pool" and wide_family == SPLITK and wide_fields[2] == 8:
                    continue
                if family != SPLITK or wide_family != SPLITK or wide_fields[2] != fields[2]:
                    raise ValueError(
                        f"[musespark_open] wide_dense_gemms: {role} kernel {kernel} does not sum "
                        f"the K slices of the {MAX_ROWS}-row kernel {kernels[role][MAX_ROWS - 1]}"
                    )
    plan = DenseGemmPlan(
        kernels, prefetch=prefetch, enable_pdl=LAUNCH_PDL, capability=capability, wide=wide_table
    )
    plan.tp_size = tp_size
    if TEP8_WIDE64_SITES and tp_size == 8:
        cells = default_kernels(8, sharded, capability)
        for role in sorted(TEP8_WIDE64_SITES):
            kernel = TEP8_WIDE64_KERNELS[role]
            family, fields = parse_kernel(kernel)
            cell_family, cell_fields = parse_kernel(cells[role][MAX_ROWS - 1])
            if family != SPLITK or cell_family != SPLITK or fields[2] != cell_fields[2]:
                raise ValueError(
                    f"[musespark_open] dense_gemms tep8_wide64: {role} kernel {kernel} does not "
                    f"sum the K slices of the 8-row kernel {cells[role][MAX_ROWS - 1]}"
                )
            shape = {"qkv": QKV_SHAPE, "o_proj": O_PROJ_SHAPE}[role]
            for rows in (TEP8_WIDE64_MIN_ROWS, TEP8_WIDE64_MAX_ROWS):
                error = kernel_error(kernel, rows, *shape)
                if error is not None:
                    raise ValueError(
                        f"[musespark_open] dense_gemms tep8_wide64: {role} kernel {kernel} "
                        f"cannot serve {rows} rows: {error} (ops/gemm/splitk_gemm.py validates "
                        "it with its wide64 setting)"
                    )
            if role in TEP8_WIDE64_WIDE_SITES and TEP8_WIDE64_WIDE_MAX_ROWS > TEP8_WIDE64_MAX_ROWS:
                # MAX_ROWS: resolve (and check) the one-launch kernel here, not at a step.
                if TEP8_WIDE64_WIDE_2CTA:
                    split_k = TEP8_WIDE64_WIDE_SPLIT_K or fields[2]
                    error = _2cta_shape_error(shape[1], shape[0], split_k)
                    if error is not None:
                        raise ValueError(
                            f"[musespark_open] tep8_wide64_splitk WIDE_TILE 2cta: {role} {error}"
                        )
                else:
                    key = (kernel, TEP8_WIDE64_WIDE_TILE, *shape, TEP8_WIDE64_WIDE_SPLIT_K)
                    _WIDE_TILES_KERNELS[key] = _wide_tiles_kernel(*key)
        logger.info(
            f"[musespark_open] dense_gemms tep8_wide64: TP8 calls of {TEP8_WIDE64_MIN_ROWS}.."
            f"{TEP8_WIDE64_MAX_ROWS} rows of "
            + ", ".join(f"{r}={TEP8_WIDE64_KERNELS[r]}" for r in sorted(TEP8_WIDE64_SITES))
            + " (flashinfer split-K, the 8-row cells' split_k); pool/unpool stay with projection_publish"
        )

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

    # The calls no table here takes -- the LM head's, and any declined call above -- run the
    # built-in low-M dispatcher, whose race imports flashinfer's split-K module by path. With a
    # split-K setting on (``splitk_gemm.configure``), that path resolves to
    # ops/gemm/splitk_gemm.py (no flashinfer patch).
    from ...ops.gemm import splitk_gemm as _splitk_gemm

    _splitk_gemm.serve_flashinfer_path()

    if prefetch and DROPPED_PREFETCH_CONSUMERS:
        logger.info(
            "[musespark_open] dense_gemm_prefetch: drop_weight_prefetch drops the weight prefetch "
            f"of {sorted(DROPPED_PREFETCH_CONSUMERS)} wherever that GEMM runs split-K (weight "
            "loads before its wait)"
        )
    logger.info(
        f"[musespark_open] dense_gemms: dense projections of all {len(layers)} layers (SM {capability}, "
        f"TP{tp_size}, "
        f"{'sharded' if sharded else 'replicated'} pooler/unpooler) on calls of at most "
        f"{MAX_ROWS} rows by {describe(kernels)}; "
        f"L2 prefetch of the next weight {'on' if prefetch else 'off'}"
    )
    if wide_table:
        steps = "; ".join(
            f"{site}=" + ", ".join(f"{k}@..{last}" for last, k in cells)
            for site, cells in wide_steps(tp_size).items()
        )
        logger.info(
            f"[musespark_open] wide_dense_gemms: calls of {MAX_ROWS + 1}..{WIDE_MAX_ROWS} rows by "
            f"{steps}; each row gets the bits of its {MAX_ROWS}-row kernel"
        )
    return plan
