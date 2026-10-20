# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step expert placement for ring-buddy replicated experts (``replicated_experts``).

Under expert parallelism over ``RANKS`` ranks (``RING_SIZES``: 8 ranks of 32-expert
blocks at tep8, 4 ranks of 64-expert blocks at tep4), block ``b`` (global experts
``[BLOCK*b, BLOCK*(b+1))``) lives on rank ``b``. With ring-buddy replication
rank ``b + 1`` (mod ``RANKS``) holds a second copy of block ``b``, so every
selected expert can run on its owner or on the owner's successor. This entry
decides, from the routing of one step, which of the two runs each expert and
rewrites the step's top-k ids into the id space of one rank's stacked expert
tensor ``[own block | predecessor's block]``:

* an expert this rank runs from its own block       -> local index ``i``
* an expert this rank runs for its predecessor      -> ``BLOCK + i``
* anything else                                     -> an id in
  ``[2*BLOCK, EXPERTS)``, i.e. outside the local window of a
  ``local_expert_offset=0, local_num_experts=2*BLOCK`` expert call.

The decision depends only on the set of selected experts, which every rank
computes identically, so every rank derives the same placement and the
``(token, slot)`` pairs are partitioned across ranks: each is executed by
exactly one rank.

Placement: let ``load[b]`` be the number of distinct selected experts in block
``b`` and ``x[b]`` how many of them move to rank ``b + 1``; rank ``r`` then runs
``load[r] - x[r] + x[r-1]`` experts. A target maximum ``T`` is feasible exactly
when every arc of ``n + 1`` consecutive ranks (``n < RANKS``) can absorb the
load of its first ``n`` blocks, ``sum <= (n + 1) * T``, and the whole ring
holds ``D <= RANKS * T`` (the cut conditions of the ring flow). The kernel
takes the smallest such ``T`` and the pointwise smallest flow,
``x[r] = max(0, max_n(sum of the n blocks ending at r) - n * T))``, which every
feasible flow dominates. Within a block the ``x[b]`` present experts with the
highest local index move.

The kernel is one program that sits on the decode step's MoE critical path
when the main stream is the longer branch, so it is kept to ``[slots, RANKS]``
work: the census is one presence word per block, as wide as the block (32 bits
at tep8, 64 bits at tep4), an OR-reduction of ``1 << i`` over the slots of that
block, which also counts a repeated expert once; a block's load is the word's
population count, and an expert's rank within its block is the population count
of the bits above its own.

With ``balance=False`` and ``static_from`` at the block size every expert stays on
its owner. Those ids address a one-window call as well: the rank's own block
alone (``local_num_experts = BLOCK``), the owner-only window
``ring_buddy.install(windows=1)`` gives ``persistent_experts`` and
``bf16_routed_experts`` without replication.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Model geometry: 256 experts over the 8 expert-parallel ranks of tep8 (the
#: defaults), or over the 4 of tep4 (``ranks=4``).
EXPERTS = 256
RANKS = 8
BLOCK = EXPERTS // RANKS
TOP_K = 8
#: The ring sizes the kernel serves: expert-parallel ranks -> experts per block.
RING_SIZES = {8: EXPERTS // 8, 4: EXPERTS // 4}


def block_of(ranks: int) -> int:
    """Experts per block on a ring of ``ranks``.

    Raises:
        ValueError: for a ring size not in ``RING_SIZES``.
    """
    if ranks == 8:
        return RING_SIZES[8]
    elif ranks == 4:
        return RING_SIZES[4]
    raise ValueError(f"ring_buddy_assign serves rings of {sorted(RING_SIZES)} ranks, got {ranks}")


#: Largest ``M * TOP_K`` one program handles (M <= 16 decode rows).
MAX_SLOTS = 128


#: Warps of the one program at every slot tile. The kernel is a chain of small
#: reductions, so more warps shorten each step's per-thread serial work more
#: than they add cross-warp exchanges up to four (GB300, CUDA graph of
#: back-to-back launches, us per launch at 1 / 2 / 4 / 8 warps: 16 slots
#: 1.52 / 1.52 / 1.52 / 1.60, 64 slots 2.21 / 1.86 / 1.65 / 1.72, 128 slots
#: 3.08 / 2.37 / 1.73 / 2.15; a one-kernel BF16 cast of the weights is 0.8).
NUM_WARPS = 4


@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit
def _popc32(v):
    """Set bits of each uint32 lane (SWAR population count), as int32."""
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _popc(v, WIDE: tl.constexpr):
    """Set bits of each presence word (uint32, or uint64 when ``WIDE``), as int32."""
    if WIDE:
        return _popc32(v.to(tl.uint32)) + _popc32((v >> 32).to(tl.uint32))
    else:
        return _popc32(v)


@triton.jit
def _ring_buddy_assign_kernel(
    selected_ptr,  # [num_slots] int32 global expert ids
    weights_ptr,  # [num_slots] float32 routing weights
    local_ids_ptr,  # [num_slots] int32 out
    weights_bf16_ptr,  # [num_slots] bfloat16 out
    stats_ptr,  # [2 * RANKS] int32 out: per-rank loads before, after
    rank,
    num_slots,
    SLOTS: tl.constexpr,
    EXPERTS: tl.constexpr,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    BALANCE: tl.constexpr,
    STATIC_FROM: tl.constexpr,
):
    tl.static_assert(BLOCK == 32 or BLOCK == 64, "one 32- or 64-bit presence word per block")
    WIDE: tl.constexpr = BLOCK == 64
    s = tl.arange(0, SLOTS)
    smask = s < num_slots
    g = tl.load(selected_ptr + s, mask=smask, other=0)
    w = tl.load(weights_ptr + s, mask=smask, other=0.0)
    blk = g // BLOCK
    i = g % BLOCK
    ranks = tl.arange(0, RANKS)

    # Census: one presence word per block, bit i set when local expert i is
    # selected by any slot; a duplicate id sets the same bit.
    if WIDE:
        shift = i.to(tl.uint64)
        bit = tl.full([SLOTS], 1, tl.uint64) << shift
        no_bits = tl.zeros([SLOTS, RANKS], dtype=tl.uint64)
    else:
        shift = i.to(tl.uint32)
        bit = tl.full([SLOTS], 1, tl.uint32) << shift
        no_bits = tl.zeros([SLOTS, RANKS], dtype=tl.uint32)
    in_rank = (blk[:, None] == ranks[None, :]) & smask[:, None]  # [SLOTS, RANKS]
    word = tl.reduce(tl.where(in_rank, bit[:, None], no_bits), 0, _bit_or)  # [RANKS]
    load = _popc(word, WIDE)  # [RANKS]

    if BALANCE:
        # arc[r, n-1]: load of the n blocks ending at r (n = 1..RANKS).
        span = ranks + 1
        back = (ranks[:, None] - ranks[None, :] + RANKS) % RANKS  # [r, k]
        inarc = back[:, None, :] < span[None, :, None]  # [r, n, k]
        arc = tl.sum(tl.where(inarc, load[None, None, :], 0), axis=2)  # [r, n]
        need = tl.where(
            span[None, :] == RANKS,
            (arc + RANKS - 1) // RANKS,
            (arc + span[None, :]) // (span[None, :] + 1),
        )
        t = tl.max(tl.max(need, axis=1), axis=0)
        x = tl.max(tl.maximum(arc - span[None, :] * t, 0), axis=1)  # [RANKS]
    elif STATIC_FROM < BLOCK:
        # Fixed placement: local experts STATIC_FROM.. of every block run on
        # the successor, whatever the step selects.
        x = _popc(word >> STATIC_FROM, WIDE)
    else:
        x = tl.zeros([RANKS], dtype=tl.int32)

    # The x[b] present experts of block b with the highest local index move:
    # a slot's expert moves when fewer than x[b] present experts of its block
    # sit above it.
    word_s = tl.reduce(tl.where(in_rank, word[None, :], no_bits), 1, _bit_or)  # [SLOTS]
    above = _popc((word_s >> shift) >> 1, WIDE)
    x_blk = tl.sum(tl.where(in_rank, x[None, :], 0), axis=1)  # [SLOTS]
    er = (blk + (above < x_blk).to(tl.int32)) % RANKS

    local = tl.where(
        er == rank,
        tl.where(blk == rank, i, BLOCK + i),
        2 * BLOCK + g % (EXPERTS - 2 * BLOCK),
    )
    tl.store(local_ids_ptr + s, local, mask=smask)
    tl.store(weights_bf16_ptr + s, w.to(tl.bfloat16), mask=smask)

    from_prev = ((ranks[:, None] + RANKS - 1) % RANKS) == ranks[None, :]
    x_prev = tl.sum(tl.where(from_prev, x[None, :], 0), axis=1)
    tl.store(stats_ptr + ranks, load)
    tl.store(stats_ptr + RANKS + ranks, load - x + x_prev)


def balance_ring(load: List[int]) -> Tuple[int, List[int]]:
    """The kernel's placement in plain Python: ``(max load, x)``.

    ``x[b]`` experts of block ``b`` move to rank ``b + 1``: the smallest
    feasible maximum and the smallest flow that reaches it.
    """
    ranks = len(load)
    arc = [
        [sum(load[(r - k) % ranks] for k in range(n)) for n in range(1, ranks + 1)]
        for r in range(ranks)
    ]
    t = -(-sum(load) // ranks)
    for r in range(ranks):
        for n in range(1, ranks):
            t = max(t, -(-arc[r][n - 1] // (n + 1)))
    x = [max(0, max(arc[r][n - 1] - n * t for n in range(1, ranks + 1))) for r in range(ranks)]
    return t, x


class RingBuddyAssign(OpWrapper):
    """Place one step's selected experts on owner or successor; rewrite ids for one rank.

    Takes the routing method's ``selected`` ids ``[M, 8]`` (int32, global) and
    its fp32 ``weights``, this rank's index and the ring size ``ranks`` (8, the
    default, or 4; the block is ``256 / ranks`` experts). Returns ``(local_ids
    [M, 8] int32, weights_bf16 [M, 8], stats [2 * ranks] int32)``: the ids in
    this rank's stacked expert space (see the module docstring), the weights
    cast to the BF16 the TRTLLM-gen expert op requires, and the per-rank
    distinct-expert loads before (``stats[:ranks]``) and after
    (``stats[ranks:]``) placement. ``balance=False`` places every expert on its
    owner, which reproduces plain expert parallelism through the stacked layout
    (or the owner-only window); with ``balance=False`` and ``static_from`` below
    the block size, local experts ``static_from..`` of every block run on the
    successor at every step instead (a routing-independent regrouping, the
    static placement). ``static_from`` defaults to the block size.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="a single decode row: 8 experts, the no-speculation shape",
            spec=dict(rows=1, routing="uniform"),
        ),
        Cell(
            why="DFlash with 1 draft token (16 slots, the smallest tile, partly masked)",
            spec=dict(rows=2, routing="keep5"),
        ),
        Cell(
            why="DFlash with 2 draft tokens: 24 of the 32-slot tile live",
            spec=dict(rows=3, routing="keep5"),
        ),
        Cell(
            why="DFlash with 4 draft tokens: 40 of the 64-slot tile live",
            spec=dict(rows=5, routing="keep5"),
        ),
        Cell(
            why="DFlash K=7 verify width with independent rows: ~57 distinct experts",
            spec=dict(rows=8, routing="uniform"),
        ),
        Cell(
            why="DFlash K=7 verify width with correlated rows (keep 5 of the previous "
            "row's 8): ~28 distinct experts, the measured served census",
            spec=dict(rows=8, routing="keep5"),
        ),
        Cell(
            why="every selected expert in one block: the most skewed census, the "
            "carry must propagate across ranks",
            spec=dict(rows=8, routing="one_block"),
        ),
        Cell(
            why="the program's slot limit: 16 rows, 128 slots",
            spec=dict(rows=16, routing="uniform"),
        ),
        Cell(
            why="tep4 (4 ranks of 64-expert blocks, 64-bit presence words) at the DFlash K=7 "
            "verify width with correlated rows",
            spec=dict(rows=8, routing="keep5", ranks=4),
        ),
        Cell(
            why="tep4, a single decode row",
            spec=dict(rows=1, routing="uniform", ranks=4),
        ),
        Cell(
            why="tep4, every selected expert in one 64-expert block, high bits of the word set",
            spec=dict(rows=8, routing="one_block", ranks=4),
        ),
        Cell(
            why="tep4 at the program's slot limit: 16 rows, 128 slots",
            spec=dict(rows=16, routing="uniform", ranks=4),
        ),
    )

    note = """
    Pure function of ``selected`` and ``rank``; one program, so it is meant for
    decode rows (``M * 8 <= 128``). Duplicate ids within a row are not expected
    (top-k selection is distinct) and are counted once in the census.

    The rewritten ids are meaningful only for an expert call with
    ``local_expert_offset=0`` and ``local_num_experts=2*BLOCK`` over a stacked
    ``[own block | predecessor's block]`` weight tensor (64 experts at tep8, 128
    at tep4), or, for the owner-only placement, ``local_num_experts=BLOCK`` over
    the rank's own block; any other window would execute experts this rank was
    not assigned.
    """

    def __call__(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block = block_of(ranks)
        static_from = block if static_from is None else static_from
        num_slots = selected.numel()
        local_ids = torch.empty_like(selected, dtype=torch.int32)
        weights_bf16 = torch.empty(selected.shape, dtype=torch.bfloat16, device=selected.device)
        stats = torch.empty(2 * ranks, dtype=torch.int32, device=selected.device)
        slots = max(16, triton.next_power_of_2(num_slots))
        _ring_buddy_assign_kernel[(1,)](
            selected,
            weights,
            local_ids,
            weights_bf16,
            stats,
            rank,
            num_slots,
            SLOTS=slots,
            EXPERTS=EXPERTS,
            RANKS=ranks,
            BLOCK=block,
            BALANCE=balance,
            STATIC_FROM=static_from,
            num_warps=NUM_WARPS,
        )
        return local_ids, weights_bf16, stats

    def reference(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The placement in plain Python over the host copy of ``selected``."""
        size = block_of(ranks)
        static_from = size if static_from is None else static_from
        flat = [int(v) for v in selected.reshape(-1).tolist()]
        present = sorted(set(flat))
        load = [0] * ranks
        for expert in present:
            load[expert // size] += 1
        if balance:
            x = balance_ring(load)[1]
        else:
            x = [
                sum(1 for e in present if e // size == b and e % size >= static_from)
                for b in range(ranks)
            ]
        moved = set()
        for b in range(ranks):
            in_block = [e for e in present if e // size == b]
            moved.update(sorted(in_block, reverse=True)[: x[b]])
        local = []
        for expert in flat:
            owner, index = divmod(expert, size)
            runs_on = (owner + (expert in moved)) % ranks
            if runs_on != rank:
                local.append(2 * size + expert % (EXPERTS - 2 * size))
            elif owner == rank:
                local.append(index)
            else:
                local.append(size + index)
        after = [load[r] - x[r] + x[(r - 1) % ranks] for r in range(ranks)]
        device = selected.device
        return (
            torch.tensor(local, dtype=torch.int32, device=device).reshape(selected.shape),
            weights.to(torch.bfloat16),
            torch.tensor(load + after, dtype=torch.int32, device=device),
        )

    def is_valid(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        rank: int,
        balance: bool = True,
        static_from: Optional[int] = None,
        ranks: int = RANKS,
    ) -> None:
        """Inputs the kernel would read past or misplace without an error."""
        assert ranks in RING_SIZES, f"ranks must be one of {sorted(RING_SIZES)}, got {ranks}"
        block = block_of(ranks)
        static_from = block if static_from is None else static_from
        assert selected.is_cuda and selected.dtype == torch.int32 and selected.is_contiguous(), (
            "selected must be contiguous int32 on the device; the kernel reads raw words"
        )
        assert selected.ndim == 2 and selected.shape[1] == TOP_K, (
            f"selected must be [M, {TOP_K}], got {tuple(selected.shape)}"
        )
        assert 0 < selected.numel() <= MAX_SLOTS, (
            f"one program covers at most {MAX_SLOTS} slots (M <= 16), got {selected.numel()}"
        )
        assert weights.dtype == torch.float32 and weights.is_contiguous(), (
            "weights must be contiguous float32"
        )
        assert weights.shape == selected.shape, "weights and selected must have one shape"
        assert 0 <= static_from <= block, f"static_from must be in [0, {block}]"
        assert 0 <= rank < ranks, f"rank must be in [0, {ranks}), got {rank}"


ring_buddy_assign = RingBuddyAssign()
