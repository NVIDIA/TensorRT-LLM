# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""This rank's 256 / EP columns of the MoE router logits for a decode step of at most 8 rows.

The router is an FP32 ``Linear(8192, 256)`` replicated on every rank and scaled
by the metap multiplier. Under expert parallelism over the TP ranks (``ranks``:
8 over two trays, 4 on one) rank ``r`` runs experts ``[L r, L r + L)`` with
``L = 256 / ranks`` (32 or 64), yet the built-in small-batch kernel
(``_router_projection_kernel`` in ``modeling_musespark_open.py``) computes all
256 logits on every rank, reading the whole 8 MiB weight. This entry computes
only the rank's ``L`` columns -- 1 or 2 MiB of weight -- and leaves the exchange of
the slices to the op that consumes them (``ops/comm/pool_gather_route``).

Each logit is the built-in one, bit for bit. The body is the built-in kernel's
statement for statement: one program per (row, expert tile), the ``[ROWS, 8192]``
weight tile and the row loaded with the built-in expressions, one ``tl.sum``
over their product (with fp-fusion off, so each product is rounded before it is
added) and the multiplier applied to the sum. The summation order of a logit is
fixed by the Triton layout of that tile, which only the warp count decides: the
tile's row count is a register repetition of the same per-row layout. The
caller passes the warp count the built-in launch would use at its row count;
the entry runs one expert per program (two programs per SM at 16 warps fill the
GPU in one wave at 8 rows), which changes no bit. Only the weight rows it reads
and the output's row stride differ from the built-in launch.

Programmatic dependent launch: the weight tile is immutable, so it is loaded
before ``griddepcontrol.wait``; the activation row only after it. With
``early_trigger`` the program calls ``griddepcontrol.launch_dependents`` first,
so the next kernel on the stream may become resident and start its own
pre-wait work (the pool GEMM's weight TMA) while this one runs; that kernel's
own wait still returns only after this grid completed, and this grid completes
only after its own wait, so whatever this kernel's predecessor wrote is visible
to it as well.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: The checkpoint's router: 256 experts over the 8192-wide residual.
EXPERTS = 256
HIDDEN = 8192
#: Expert-parallel rank counts (tep8 over two trays, tep4 on one); rank ``r`` of ``ranks``
#: runs ``local_experts(ranks)`` experts.
RANK_COUNTS = (4, 8)


def local_experts(ranks: int) -> int:
    """Experts each of ``ranks`` expert-parallel ranks runs: 32 at EP8, 64 at EP4."""
    return EXPERTS // ranks


#: Rows a call may carry: one request's last accepted token and up to 7 drafts.
MAX_ROWS = 8
#: Warp counts the built-in launch uses at these row counts (16 with the MoE
#: prepare stream, 4 without it from 3 rows up); each fixes a summation order.
WARP_COUNTS = (4, 16)


@triton.jit
def _router_shard_kernel(
    hidden_states,
    weight,
    output,
    scale,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS: tl.constexpr,
    PDL: tl.constexpr,
    EARLY_TRIGGER: tl.constexpr,
):
    if EARLY_TRIGGER:
        tl.extra.cuda.gdc_launch_dependents()
    token = tl.program_id(0)
    expert = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, HIDDEN)
    # The built-in kernel's tile and row, the immutable weight first.
    w = tl.load(
        weight + expert[:, None] * HIDDEN + offsets[None, :],
        expert[:, None] < EXPERTS,
        other=0.0,
    )
    if PDL:
        tl.extra.cuda.gdc_wait()
    x = tl.load(hidden_states + token * HIDDEN + offsets).to(tl.float32)
    logits = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(output + token * EXPERTS + expert, logits, expert < EXPERTS)


#: ``router_shard_side`` (``router_fork``) launches ``_router_shard_kernel``'s statements
#: with ``PDL`` and ``EARLY_TRIGGER`` false, written out under a symbol of their own, so
#: that every binary of ``_router_shard_kernel`` waits before it exits (the composition
#: tests audit every binary of that symbol).
@triton.jit
def _router_shard_side_kernel(
    hidden_states,
    weight,
    output,
    scale,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, HIDDEN)
    w = tl.load(
        weight + expert[:, None] * HIDDEN + offsets[None, :],
        expert[:, None] < EXPERTS,
        other=0.0,
    )
    x = tl.load(hidden_states + token * HIDDEN + offsets).to(tl.float32)
    logits = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(output + token * EXPERTS + expert, logits, expert < EXPERTS)


#: Experts per program. Not part of the summation order (see the module docstring).
TILE_ROWS = 1


class RouterShard(OpWrapper):
    """``(hidden_states.float() @ weight[L r:L r+L].T) * scale`` in the built-in summation order.

    Args:
        hidden_states: ``[M, 8192]`` BF16, contiguous, ``1 <= M <= 8``.
        weight: the full router weight ``[256, 8192]`` FP32, contiguous.
        rank: the expert-parallel rank ``r``; columns ``[L r, L r + L)`` are computed,
            ``L = local_experts(ranks)``.
        scale: the metap multiplier, a Python float (applied in FP32 after the sum).
        num_warps: the warp count the built-in ``_router_projection_kernel`` launch
            uses for this ``M`` (4 or 16); it fixes the summation order.
        enable_pdl: launch with programmatic dependent launch (weight read before
            ``griddepcontrol.wait``).
        early_trigger: with ``enable_pdl``, release the next kernel at entry.
        ranks: the expert-parallel group size (8 or 4).

    Returns:
        ``[M, L]`` FP32 logits of experts ``L r .. L r + L - 1``, newly allocated.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K=7 verify: 8 rows, the served launch (prepare stream, 16 warps)",
            spec=dict(rows=8, num_warps=16),
        ),
        Cell(why="one row at 16 warps (no speculation)", spec=dict(rows=1, num_warps=16)),
        Cell(
            why="an odd row count the built-in launches at 1-row tiles (3 rows)",
            spec=dict(rows=3, num_warps=16),
        ),
        Cell(
            why="8 rows without the prepare stream: the 4-warp summation order",
            spec=dict(rows=8, num_warps=4),
        ),
        Cell(
            why="tep4 K=7 verify: 8 rows, 64 columns per rank (one tray)",
            spec=dict(rows=8, num_warps=16, ranks=4),
        ),
    )

    note = """
    The claim the test proves is equality with the built-in kernel's columns
    ``L r .. L r + L - 1`` at the same warp count, for every rank, every row count
    1..8 and both warp counts, at EP8 (L = 32) and EP4 (L = 64), and with an
    early trigger; the cells additionally gate the op against the float64
    reference.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        enable_pdl: bool = False,
        early_trigger: bool = False,
        ranks: int = 8,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        local = local_experts(ranks)
        output = torch.empty((tokens, local), device=hidden_states.device, dtype=torch.float32)
        first = rank * local
        _router_shard_kernel[(tokens, local // TILE_ROWS)](
            hidden_states,
            weight[first : first + local],
            output,
            scale,
            EXPERTS=local,
            HIDDEN=HIDDEN,
            ROWS=TILE_ROWS,
            PDL=bool(enable_pdl),
            EARLY_TRIGGER=bool(enable_pdl and early_trigger),
            num_warps=num_warps,
            enable_fp_fusion=False,
            launch_pdl=bool(enable_pdl),
        )
        return output

    def reference(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        enable_pdl: bool = False,
        early_trigger: bool = False,
        ranks: int = 8,
    ) -> torch.Tensor:
        """Float64 product of the rank's rows, the FP32 multiplier applied to the FP32 sum.

        ``num_warps``, ``enable_pdl`` and ``early_trigger`` only change the
        summation order or the launch and are ignored here.
        """
        del num_warps, enable_pdl, early_trigger
        local = local_experts(ranks)
        first = rank * local
        rows = weight[first : first + local].double()
        logits = (hidden_states.double() @ rows.t()).float()
        return logits * torch.tensor(scale, dtype=torch.float32, device=logits.device)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        enable_pdl: bool = False,
        early_trigger: bool = False,
        ranks: int = 8,
    ) -> None:
        """The kernel indexes both operands as dense row-major and reads ``M`` rows of 8192."""
        if hidden_states.dtype != torch.bfloat16 or hidden_states.dim() != 2:
            raise ValueError(f"hidden_states must be 2-D BF16, got {hidden_states.dtype}")
        if not hidden_states.is_contiguous() or hidden_states.shape[1] != HIDDEN:
            raise ValueError(f"hidden_states must be contiguous [M, {HIDDEN}]")
        if not 1 <= hidden_states.shape[0] <= MAX_ROWS:
            raise ValueError(f"1..{MAX_ROWS} rows, got {hidden_states.shape[0]}")
        if weight.dtype != torch.float32 or tuple(weight.shape) != (EXPERTS, HIDDEN):
            raise ValueError(f"weight must be FP32 [{EXPERTS}, {HIDDEN}]")
        if not weight.is_contiguous():
            raise ValueError("weight must be contiguous")
        if ranks not in RANK_COUNTS:
            raise ValueError(f"ranks must be one of {RANK_COUNTS}, got {ranks}")
        if not 0 <= rank < ranks:
            raise ValueError(f"rank must be in [0, {ranks}), got {rank}")
        if num_warps not in WARP_COUNTS:
            raise ValueError(f"num_warps must be one of {WARP_COUNTS}, got {num_warps}")


router_shard = RouterShard()


class RouterShardSide(OpWrapper):
    """``RouterShard``'s logits, by a plain launch into ``out`` on the current stream.

    For a call on a side stream forked from the stream that produced the input
    (``router_fork``): the launch carries no programmatic-dependent-launch
    attribute, so a captured edge into it is a full one -- it starts once its
    predecessors completed -- and the kernel (``_router_shard_side_kernel``)
    reads its input without a ``griddepcontrol.wait``. ``out`` is allocated by
    the caller on the stream that consumes it.

    Args:
        hidden_states: ``[M, 8192]`` BF16, contiguous, ``1 <= M <= 8``.
        weight: the full router weight ``[256, 8192]`` FP32, contiguous.
        rank: the expert-parallel rank ``r``; columns ``[L r, L r + L)`` are computed.
        scale: the metap multiplier, a Python float (applied in FP32 after the sum).
        num_warps: the warp count the built-in ``_router_projection_kernel`` launch
            uses for this ``M`` (4 or 16); it fixes the summation order.
        out: ``[M, L]`` FP32, contiguous, on the input's device; overwritten.
        ranks: the expert-parallel group size (8 or 4); ``L = local_experts(ranks)``.

    Returns:
        ``out``, holding the FP32 logits of experts ``L r .. L r + L - 1``.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K=7 verify: 8 rows, the served launch (prepare stream, 16 warps)",
            spec=dict(rows=8, num_warps=16),
        ),
        Cell(why="one row at 16 warps (no speculation)", spec=dict(rows=1, num_warps=16)),
        Cell(
            why="an odd row count the built-in launches at 1-row tiles (3 rows)",
            spec=dict(rows=3, num_warps=16),
        ),
        Cell(
            why="8 rows without the prepare stream: the 4-warp summation order",
            spec=dict(rows=8, num_warps=4),
        ),
    )

    note = """
    The claim the test proves is equality with ``router_shard`` and with the
    built-in kernel's columns ``L r .. L r + L - 1`` at the same warp count, for
    every rank, every row count 1..8 and both warp counts, at EP8 and EP4; the
    cells additionally gate the op against the float64 reference.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        out: torch.Tensor,
        ranks: int = 8,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        local = local_experts(ranks)
        first = rank * local
        _router_shard_side_kernel[(tokens, local // TILE_ROWS)](
            hidden_states,
            weight[first : first + local],
            out,
            scale,
            EXPERTS=local,
            HIDDEN=HIDDEN,
            ROWS=TILE_ROWS,
            num_warps=num_warps,
            enable_fp_fusion=False,
            launch_pdl=False,
        )
        return out

    def reference(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        out: torch.Tensor,
        ranks: int = 8,
    ) -> torch.Tensor:
        """``RouterShard.reference``; ``num_warps`` and ``out`` do not change the values."""
        del out
        return router_shard.reference(hidden_states, weight, rank, scale, num_warps, ranks=ranks)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        out: torch.Tensor,
        ranks: int = 8,
    ) -> None:
        """``RouterShard.is_valid``, and ``out`` a dense ``[M, L]`` FP32 tensor on the input's device."""
        router_shard.is_valid(hidden_states, weight, rank, scale, num_warps, ranks=ranks)
        local = local_experts(ranks)
        if out.dtype != torch.float32 or tuple(out.shape) != (hidden_states.shape[0], local):
            raise ValueError(f"out must be FP32 [{hidden_states.shape[0]}, {local}]")
        if not out.is_contiguous() or out.device != hidden_states.device:
            raise ValueError("out must be contiguous, on the input's device")


router_shard_side = RouterShardSide()


def builtin_warps(tokens: int, overlaps: bool) -> int:
    """The warp count of the built-in small-batch router launch at ``tokens`` rows (1..8).

    ``overlaps``: the MoE has a prepare stream and no MoE communication strategy
    (the built-in launch then uses 16 warps at every row count up to 11).
    """
    if overlaps or tokens <= 2:
        return 16
    return 4


#: ``router_shard_publish`` (``pool_into_route``) launches ``_router_shard_side_kernel``'s
#: statements under a symbol of its own whose only difference is the store: the logit
#: is written into ``COPIES`` boxes of ``BOX`` words each (one per CTA of the routed
#: pool kernel that reads them) instead of one ``[M, 32]`` output. Placed after every
#: other kernel of this file, so their source lines and line info are unchanged.
@triton.jit
def _router_shard_publish_kernel(
    hidden_states,
    weight,
    boxes,
    scale,
    EXPERTS: tl.constexpr,
    HIDDEN: tl.constexpr,
    ROWS: tl.constexpr,
    COPIES: tl.constexpr,
    BOX: tl.constexpr,
):
    token = tl.program_id(0)
    expert = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, HIDDEN)
    w = tl.load(
        weight + expert[:, None] * HIDDEN + offsets[None, :],
        expert[:, None] < EXPERTS,
        other=0.0,
    )
    x = tl.load(hidden_states + token * HIDDEN + offsets).to(tl.float32)
    logits = tl.sum(w * x[None, :], axis=1) * scale
    for copy in tl.static_range(COPIES):
        tl.store(boxes + copy * BOX + token * EXPERTS + expert, logits, expert < EXPERTS)


def publish_box(ranks: int) -> int:
    """Words of one logit box: an ``[8, L]`` FP32 block (rows past ``M`` are not written)."""
    return MAX_ROWS * local_experts(ranks)


#: The EP8 box, which the EP8-only callers (the routed pool kernel) read.
PUBLISH_BOX = publish_box(8)


def publish_copies(tokens: int) -> int:
    """Boxes a call of ``tokens`` rows writes: one per CTA of the routed pool kernel (2 rows each)."""
    return (tokens + 1) // 2


class RouterShardPublish(OpWrapper):
    """``RouterShardSide``'s logits, stored into the routed pool kernel's direct transport.

    ``pool_into_route``: the router shard on the router fork's side stream (a plain launch, no
    ``griddepcontrol``) writes its ``[M, 32]`` logit block straight into this
    rank's logit boxes of a multicast transport (``ops/comm/pool_direct_route``):
    box ``c`` is read by CTA ``c`` of every rank's routed pool kernel, so each
    box is a full copy, row ``m`` at words ``L m .. L m + L - 1``. One store per
    element per box and call; the stores are the publication. Each logit is
    ``router_shard``'s, bit for bit (the same statements up to the store).

    Args:
        hidden_states: ``[M, 8192]`` BF16, contiguous, ``1 <= M <= 8``.
        weight: the full router weight ``[256, 8192]`` FP32, contiguous.
        rank: the expert-parallel rank ``r``; columns ``[L r, L r + L)`` are computed.
        scale: the metap multiplier, a Python float (applied in FP32 after the sum).
        num_warps: the warp count of the built-in launch at this ``M`` (4 or 16).
        boxes: ``[publish_copies(M), publish_box(ranks)]`` FP32, contiguous (the
            transport's boxes, on its multicast address: write-only).
        ranks: the expert-parallel group size (8 or 4); ``L = local_experts(ranks)``.

    Returns:
        ``boxes`` as ``[publish_copies(M), M, L]`` (the written rows of each box; a view
        of the multicast address, which nothing may read).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="K=7 verify: 8 rows, the served launch (16 warps), four boxes",
            spec=dict(rows=8, num_warps=16),
        ),
        Cell(why="one row at 16 warps: one box", spec=dict(rows=1, num_warps=16)),
        Cell(
            why="an odd row count (3 rows): two boxes, the second CTA one row",
            spec=dict(rows=3, num_warps=16),
        ),
        Cell(
            why="8 rows without the prepare stream: the 4-warp summation order",
            spec=dict(rows=8, num_warps=4),
        ),
    )

    note = """
    The claim the test proves is that every box holds ``router_shard``'s logits
    bit for bit (and the built-in kernel's columns ``32r .. 32r + 31``) for every
    rank, row count 1..8 and both warp counts, and that box words past the rows
    are untouched; the cells additionally gate the op against the float64
    reference.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        boxes: torch.Tensor,
        ranks: int = 8,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        local = local_experts(ranks)
        first = rank * local
        _router_shard_publish_kernel[(tokens, local // TILE_ROWS)](
            hidden_states,
            weight[first : first + local],
            boxes,
            scale,
            EXPERTS=local,
            HIDDEN=HIDDEN,
            ROWS=TILE_ROWS,
            COPIES=publish_copies(tokens),
            BOX=publish_box(ranks),
            num_warps=num_warps,
            enable_fp_fusion=False,
            launch_pdl=False,
        )
        return boxes.view(-1, MAX_ROWS, local)[:, :tokens]

    def reference(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        boxes: torch.Tensor,
        ranks: int = 8,
    ) -> torch.Tensor:
        """``RouterShard.reference`` per box: ``[publish_copies(M), M, L]``."""
        logits = router_shard.reference(hidden_states, weight, rank, scale, num_warps, ranks=ranks)
        return logits.unsqueeze(0).expand(publish_copies(hidden_states.shape[0]), -1, -1)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        rank: int,
        scale: float,
        num_warps: int,
        boxes: torch.Tensor,
        ranks: int = 8,
    ) -> None:
        """``RouterShard.is_valid``, and ``boxes`` a dense ``[copies, publish_box(ranks)]`` FP32 tensor."""
        router_shard.is_valid(hidden_states, weight, rank, scale, num_warps, ranks=ranks)
        copies = publish_copies(hidden_states.shape[0])
        box = publish_box(ranks)
        if boxes.dtype != torch.float32 or tuple(boxes.shape) != (copies, box):
            raise ValueError(f"boxes must be FP32 [{copies}, {box}]")
        if not boxes.is_contiguous() or boxes.device != hidden_states.device:
            raise ValueError("boxes must be contiguous, on the input's device")


router_shard_publish = RouterShardPublish()
