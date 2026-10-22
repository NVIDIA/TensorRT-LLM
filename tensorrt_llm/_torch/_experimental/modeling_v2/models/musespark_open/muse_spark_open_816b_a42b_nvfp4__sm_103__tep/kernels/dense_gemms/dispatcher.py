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
unchanged. A fast path may also give the plan a table for 9..32 rows
(``DenseGemmPlan.set_wide_kernels``, ``wide_projection_gathers``): while it arms
the plan (``wide_armed``, one decode forward), a call of those rows at a site the
table names runs that table's kernel. Nothing else in the layer changes, so the other fast paths
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

#: The same table for sm107 (212 SMs) at tep8, chosen the same way: the direct kernel at two or
#: four outputs per CTA for QKV at 1-3 rows and for the unpooler slice at 1-7 rows;
#: split-K 4 for QKV from 4 rows (36 x 4 CTAs is the widest split-K grid at its width);
#: flashinfer's split-K ``(64, 8, 1, 12)`` for the attention output projection from 3 rows,
#: where it is faster than cuBLAS on this arch. The pooler keeps the sm103 cells, which
#: are within the measurement spread of the fastest here.
SM107_KERNELS: Mapping[str, Tuple[Tuple[int, str], ...]] = {
    "qkv": (
        (1, "direct:256,2"),
        (2, "direct:128,4"),
        (3, "direct:64,2"),
        (MAX_ROWS, "splitk:64,8,4,12"),
    ),
    "o_proj": ((2, "direct:64,4"), (MAX_ROWS, "splitk:64,8,1,12")),
    "pool": DEFAULT_KERNELS["pool"],
    "unpool": ((1, "direct:256,2"), (7, "direct:128,2"), (MAX_ROWS, "splitk:64,8,4,12")),
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

#: The measured table of each ``(compute capability, TP size)``, with the pooler and
#: unpooler on the rank's slices (``kernel_steps`` swaps in the replicated tep4 cells).
KERNELS_BY_SM_TP: Mapping[
    Tuple[Tuple[int, int], int], Mapping[str, Tuple[Tuple[int, str], ...]]
] = {
    ((10, 3), 8): DEFAULT_KERNELS,
    ((10, 3), 4): TP4_KERNELS,
    ((10, 7), 8): SM107_KERNELS,
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
        ValueError: if ``KERNELS_BY_SM_TP`` has no table for ``(capability, tp_size)``.
    """
    key = (tuple(capability), int(tp_size))
    table = KERNELS_BY_SM_TP.get(key)
    if table is None:
        raise ValueError(
            f"[musespark_open] dense_gemms: no kernel table for SM {key[0]} at TP {key[1]} "
            f"(supported: {sorted(KERNELS_BY_SM_TP)})"
        )
    if key == ((10, 3), 4) and not sharded:
        return {**table, **TP4_REPLICATED_PROJECTION_KERNELS}
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
    ) -> None:
        self.kernels = dict(kernels)
        #: The 9..32-row table (``set_wide_kernels``) and whether the current forward arms it.
        self.wide_kernels: Dict[str, Tuple[str, ...]] = {}
        self.wide_armed = False
        #: The fast path armed for the current decode forward above 32 rows (``batch_dense_gemms``'s
        #: ``BatchDenseGemms``), whose ``projection`` every re-classed projection's
        #: ``apply_linear`` calls; ``None`` otherwise.
        self.batch = None
        self.prefetch = prefetch
        self.enable_pdl = enable_pdl
        #: The compute capability of the device the plan serves (``install`` reads it once);
        #: what an arch-specific producer policy keys on.
        self.capability = None if capability is None else tuple(capability)
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
        """The kernel of a ``rows``-row call at site ``role``, or None where the plan has none:
        ``kernels`` at 1..8 rows; ``wide_kernels`` at 9..32 rows while ``wide_armed``."""
        if 1 <= rows <= MAX_ROWS:
            return self.kernels[role][rows - 1]
        if self.wide_armed and WIDE_MIN_ROWS <= rows <= WIDE_MAX_ROWS:
            wide = self.wide_kernels.get(role)
            return None if wide is None else wide[rows - WIDE_MIN_ROWS]
        return None

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
    families storing into the slot (cuBLAS on sm103, flashinfer's split-K and direct kernels on
    sm107) and everything else copied.

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
        if self.publish is not None:
            return self._apply_publish(module, input_tensor, weight, bias, force_active)
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
        if (
            plan.prefetch
            and self.site.next_weight is not None
            and rows not in self.site.skip_prefetch_rows
        ):
            plan.fork_prefetch(self.site.next_weight())
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
        kernel = plan.kernels[self.site.role][rows - 1]
        if self.fits(input_tensor, weight, bias) and parse_kernel(kernel)[0] in policy.direct:
            dense_gemm(input_tensor.view(-1, k), weight, kernel, plan.enable_pdl, out=publish)
            if (
                plan.prefetch
                and self.site.next_weight is not None
                and rows not in self.site.skip_prefetch_rows
            ):
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
        if batch is None or (getattr(self, "lora", None) is not None and bool(lora_params)):
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
) -> DenseGemmPlan:
    """Route every layer's four projections through ``dense_gemm``; return the shared plan.

    ``kernels`` is the kernel of each site per row count; when None, the table of the
    device's compute capability (``device_capability``, read once here), the model's TP
    size and its projection layout (``default_kernels``). Every module is checked
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
    plan = DenseGemmPlan(kernels, prefetch=prefetch, enable_pdl=LAUNCH_PDL, capability=capability)

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
    return plan
