# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step placement of the touched experts over the hot-replica window (``hot_replicas``).

Under tep4 expert parallelism rank ``r`` owns experts ``[64 r, 64 r + 64)``. With
``hot_replicas`` every rank also holds ``R`` copies of other ranks' experts (the
layer's hot experts, chosen offline from a routing census), stacked after its own
block: local ids ``0..63`` are its own experts, ``64 + k`` its ``k``-th copy. Every
hot expert has exactly one copy, so it can run on its owner or on its holder.

The layer's census table (``table``, int32, identical on every rank) holds:

* ``table[0:H]`` -- the ``H = 4 R`` hot experts (global ids, distinct), in priority order;
* ``table[H:2H]`` -- the rank holding each one's copy (never its owner);
* ``table[2H:2H + 4R]`` -- per rank ``r``, the experts of its copies in stack order
  (``table[2H + r R + k]`` is the expert at local id ``64 + k`` on rank ``r``).

The placement of one step is an integer function of the step's selected ids and the
table only, so every rank derives the same placement without communication:

1. the touched set: one 64-bit presence word per owner block, the OR of
   ``1 << (e % 64)`` over the step's slots (a repeated expert counts once);
2. every touched expert on its owner: rank ``r``'s load is the population count of its
   presence word (the loads without copies);
3. the hot experts in table order: a touched one moves to its holder when the holder's
   load so far plus one is below its owner's; the owner's load drops by one and the
   holder's grows by one. A move never raises the larger of the two loads, so no
   step's maximum load exceeds its maximum without copies.

Each selected (token, slot) then runs on exactly one rank, the one its expert was
placed on, and this entry rewrites the step's top-k ids into one rank's stacked id
space: its own experts placed on it -> ``e - 64 r``; a copy placed on it -> ``64 + k``;
anything else -> an id in ``[64 + R, 256)``, outside a ``local_expert_offset=0,
local_num_experts=64 + R`` expert call. The expert call's unfinalized output then
feeds the built-in finalize unchanged: the cross-rank sum still adds every selected
expert row exactly once.

The kernel is one program on the decode step's MoE prologue chain, so its per-slot
work is elementwise and its only cross-slot work is the presence reduction (a
256-bin histogram of the step's ids, whose nonzero bins become the presence bits);
the placement itself is ``H`` scalar steps. The table reaches it as kernel arguments
(``pack_table``: 10-bit (expert, holder) entries and this rank's 8-bit copy list, 3 per
int32 word), so those steps read no memory; it is immutable, so a CUDA graph may bake
the arguments in.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Model geometry at tep4: 256 experts over 4 expert-parallel ranks of 64.
EXPERTS = 256
RANKS = 4
BLOCK = EXPERTS // RANKS
TOP_K = 8
#: Copies per rank a table may hold (``R``); the kernel is compiled per R.
REPLICAS = (2, 4, 8)
#: Decode rows the entry serves (5 to 64 requests of 8 rows).
MIN_ROWS = 33
MAX_ROWS = 512
MAX_SLOTS = MAX_ROWS * TOP_K


def table_words(replicas: int) -> int:
    """int32 words of one layer's census table for ``replicas`` copies per rank."""
    return 3 * RANKS * replicas


#: Kernel arguments of a packed table: entry words (3 entries of 10 bits each, up to
#: 4 * max(REPLICAS) entries), then copy words (3 copies of 8 bits each).
ENTRY_WORDS = -(-RANKS * max(REPLICAS) // 3)
COPY_WORDS = -(-max(REPLICAS) // 3)
PACKED_WORDS = ENTRY_WORDS + COPY_WORDS


def pack_table(table: Sequence[int], rank: int, replicas: int) -> Tuple[int, ...]:
    """The kernel arguments of one layer's table (``build_table``) for ``rank``: host ints.

    Every word is below ``2**30``, so each argument is a 32-bit integer whatever the table.
    """
    h = RANKS * replicas
    if len(table) != table_words(replicas):
        raise ValueError(f"a table of {replicas} copies per rank has {table_words(replicas)} words")
    entries = [int(table[k]) | (int(table[h + k]) << 8) for k in range(h)]
    mine = [int(v) for v in table[2 * h + rank * replicas : 2 * h + (rank + 1) * replicas]]
    words = [0] * PACKED_WORDS
    for k, v in enumerate(entries):
        words[k // 3] |= v << (10 * (k % 3))
    for k, v in enumerate(mine):
        words[ENTRY_WORDS + k // 3] |= v << (8 * (k % 3))
    return tuple(words)


def unpack_table(
    packed: Sequence[int], rank: int, replicas: int
) -> Tuple[List[int], List[int], List[int]]:
    """``(hot experts, holders, this rank's copies)`` of a packed table."""
    h = RANKS * replicas
    entries = [(int(packed[k // 3]) >> (10 * (k % 3))) & 1023 for k in range(h)]
    mine = [(int(packed[ENTRY_WORDS + k // 3]) >> (8 * (k % 3))) & 255 for k in range(replicas)]
    return [v & 255 for v in entries], [v >> 8 for v in entries], mine


def _num_warps(slots: int) -> int:
    return 4 if slots <= 1024 else 8


@triton.jit
def _popc32(v):
    """Set bits of each uint32 lane (SWAR population count), as int32."""
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return ((v * 0x01010101) >> 24).to(tl.int32)


@triton.jit
def _popc64(v):
    return _popc32(v.to(tl.uint32)) + _popc32((v >> 32).to(tl.uint32))


@triton.jit
def _bit_or(a, b):
    return a | b


@triton.jit
def _word(i: tl.constexpr, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10):
    """Packed argument ``i`` (a compile-time index)."""
    if i == 0:
        return a0
    elif i == 1:
        return a1
    elif i == 2:
        return a2
    elif i == 3:
        return a3
    elif i == 4:
        return a4
    elif i == 5:
        return a5
    elif i == 6:
        return a6
    elif i == 7:
        return a7
    elif i == 8:
        return a8
    elif i == 9:
        return a9
    else:
        return a10


@triton.jit
def _pick(k, v0, v1, v2, v3):
    """``v_k`` for a scalar rank ``k`` in 0..3."""
    return tl.where(k == 0, v0, tl.where(k == 1, v1, tl.where(k == 2, v2, v3)))


@triton.jit(
    do_not_specialize=[
        "rank", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10", "c0", "c1", "c2",
    ]
)  # fmt: skip
def _hot_replica_assign_kernel(
    selected_ptr,  # [num_slots] int32 global expert ids
    weights_ptr,  # [num_slots] float32 routing weights
    local_ids_ptr,  # [num_slots] int32 out
    weights_bf16_ptr,  # [num_slots] bfloat16 out
    stats_ptr,  # [2 * RANKS] int32 out: per-rank loads before, after
    rank,
    num_slots,
    # pack_table: 3 (expert | holder << 8) entries of 10 bits per word, in priority order
    e0,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    e9,
    e10,
    # pack_table: this rank's copies in stack order, 3 of 8 bits per word
    c0,
    c1,
    c2,
    SLOTS: tl.constexpr,
    R: tl.constexpr,
    EXPERTS: tl.constexpr,
    RANKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tl.static_assert(BLOCK == 64, "one 64-bit presence word per tep4 block")
    tl.static_assert(RANKS == 4, "the tep4 ring")
    H: tl.constexpr = RANKS * R
    WINDOW: tl.constexpr = BLOCK + R
    s = tl.arange(0, SLOTS)
    smask = s < num_slots
    g = tl.load(selected_ptr + s, mask=smask, other=0)
    w = tl.load(weights_ptr + s, mask=smask, other=0.0)
    blk = g // BLOCK
    i = g % BLOCK
    shift = i.to(tl.uint64)
    one64 = tl.full([], 1, tl.uint64)
    zero64 = tl.zeros([], dtype=tl.uint64)
    ranks = tl.arange(0, RANKS)

    # 1. Presence: one word per owner block, from the per-expert slot counts (the
    # SLOTS - num_slots masked-off slots load expert 0). Distinct bits sum to their OR.
    experts = tl.arange(0, EXPERTS)
    counts = tl.histogram(g, EXPERTS) - tl.where(experts == 0, SLOTS - num_slots, 0)
    bits = tl.where(counts > 0, one64 << (experts % BLOCK).to(tl.uint64), zero64)
    word = tl.sum(tl.reshape(bits, [RANKS, BLOCK]), 1)  # [RANKS]
    w0 = tl.reduce(tl.where(ranks == 0, word, zero64), 0, _bit_or)
    w1 = tl.reduce(tl.where(ranks == 1, word, zero64), 0, _bit_or)
    w2 = tl.reduce(tl.where(ranks == 2, word, zero64), 0, _bit_or)
    w3 = tl.reduce(tl.where(ranks == 3, word, zero64), 0, _bit_or)

    # 2. Every touched expert on its owner.
    before = _popc64(word)  # [RANKS] int32
    l0 = tl.sum(tl.where(ranks == 0, before, 0), 0)
    l1 = tl.sum(tl.where(ranks == 1, before, 0), 0)
    l2 = tl.sum(tl.where(ranks == 2, before, 0), 0)
    l3 = tl.sum(tl.where(ranks == 3, before, 0), 0)

    # 3. The hot experts in table order: a touched one moves to its holder when the
    # holder's load + 1 is below its owner's, so no move raises the larger of the two.
    m0 = zero64
    m1 = zero64
    m2 = zero64
    m3 = zero64
    for k in tl.static_range(H):
        entry = (
            _word(k // 3, e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10) >> (10 * (k % 3))
        ) & 1023
        e = entry & 255
        h = entry >> 8
        o = e // BLOCK
        eb = (e % BLOCK).to(tl.uint64)
        touched = ((_pick(o, w0, w1, w2, w3) >> eb) & one64) != zero64
        move = touched & (_pick(h, l0, l1, l2, l3) + 1 < _pick(o, l0, l1, l2, l3))
        d = move.to(tl.int32)
        l0 += tl.where(h == 0, d, 0) - tl.where(o == 0, d, 0)
        l1 += tl.where(h == 1, d, 0) - tl.where(o == 1, d, 0)
        l2 += tl.where(h == 2, d, 0) - tl.where(o == 2, d, 0)
        l3 += tl.where(h == 3, d, 0) - tl.where(o == 3, d, 0)
        mb = tl.where(move, one64 << eb, zero64)
        m0 = m0 | tl.where(o == 0, mb, zero64)
        m1 = m1 | tl.where(o == 1, mb, zero64)
        m2 = m2 | tl.where(o == 2, mb, zero64)
        m3 = m3 | tl.where(o == 3, mb, zero64)

    # 4. Per slot: on its owner, on this rank's copy, or elsewhere.
    moved_word = tl.where(blk == 0, m0, tl.where(blk == 1, m1, tl.where(blk == 2, m2, m3)))
    moved = ((moved_word >> shift) & 1) != 0
    copy = tl.zeros([SLOTS], dtype=tl.int32)
    for k in tl.static_range(R):
        mine = (_word(k // 3, c0, c1, c2, c0, c0, c0, c0, c0, c0, c0, c0) >> (8 * (k % 3))) & 255
        copy = tl.where(g == mine, k + 1, copy)
    elsewhere = WINDOW + g % (EXPERTS - WINDOW)
    local = tl.where(
        moved,
        tl.where(copy > 0, BLOCK + copy - 1, elsewhere),
        tl.where(blk == rank, i, elsewhere),
    )
    tl.store(local_ids_ptr + s, local, mask=smask)
    tl.store(weights_bf16_ptr + s, w.to(tl.bfloat16), mask=smask)

    after = tl.where(ranks == 0, l0, tl.where(ranks == 1, l1, tl.where(ranks == 2, l2, l3)))
    tl.store(stats_ptr + ranks, before)
    tl.store(stats_ptr + RANKS + ranks, after)


def build_table(entries: Sequence[Tuple[int, int]], replicas: int) -> List[int]:
    """One layer's census table (``table_words(replicas)`` ints) from its (expert, holder) entries.

    ``entries`` are in priority order; every rank holds exactly ``replicas`` copies, every
    expert appears once and its holder is not its owner.

    Raises:
        ValueError: for entries that do not form such a table.
    """
    h = RANKS * replicas
    if len(entries) != h:
        raise ValueError(
            f"a table of {replicas} copies per rank has {h} entries, got {len(entries)}"
        )
    experts = [int(e) for e, _ in entries]
    holders = [int(r) for _, r in entries]
    if len(set(experts)) != h or not all(0 <= e < EXPERTS for e in experts):
        raise ValueError("hot experts must be distinct ids in [0, 256)")
    if any(r == e // BLOCK or not 0 <= r < RANKS for e, r in zip(experts, holders)):
        raise ValueError("every copy is held by a rank other than its owner")
    mine = [[e for e, r in zip(experts, holders) if r == rank] for rank in range(RANKS)]
    if any(len(m) != replicas for m in mine):
        raise ValueError(
            f"every rank holds exactly {replicas} copies, got {[len(m) for m in mine]}"
        )
    return experts + holders + [e for m in mine for e in m]


def place(selected: Sequence[int], table: Sequence[int], replicas: int) -> Tuple[dict, list, list]:
    """The placement rule in plain Python: ``(rank of each touched expert, loads before, after)``."""
    h = RANKS * replicas
    hot = [int(v) for v in table[:h]]
    holder = [int(v) for v in table[h : 2 * h]]
    touched = set(int(v) for v in selected)
    runs_on = {e: e // BLOCK for e in touched}
    before = [sum(1 for e in touched if e // BLOCK == r) for r in range(RANKS)]
    load = list(before)
    for e, r in zip(hot, holder):
        o = e // BLOCK
        if e in touched and load[r] + 1 < load[o]:
            runs_on[e] = r
            load[o] -= 1
            load[r] += 1
    return runs_on, before, load


class HotReplicaAssign(OpWrapper):
    """Place one step's touched experts over owners and hot copies; rewrite ids for one rank.

    Takes the routing method's ``selected`` ids ``[M, 8]`` (int32, global), its fp32
    ``weights``, the layer's census table packed for this rank (``pack_table`` of
    ``build_table``'s words: host ints), this rank's index and the copies per rank
    ``replicas``. Returns ``(local_ids
    [M, 8] int32, weights_bf16 [M, 8], stats [8] int32)``: the ids in this rank's
    stacked id space (``[own block | copies]``, see the module docstring), the weights
    cast to the BF16 the TRTLLM-gen expert op requires, and the per-rank touched-expert
    loads before (``stats[:4]``, every expert on its owner) and after the placement.
    """

    ARCHS = frozenset({Arch.SM_103})

    note = """
    Pure function of ``selected``, ``table`` and ``rank``; one program over at most
    4096 slots, meant for decode rows (33..512). Duplicate ids within a row are not
    expected (top-k selection is distinct) and count once in the presence words.

    The rewritten ids are meaningful only for an expert call with
    ``local_expert_offset=0`` and ``local_num_experts=64 + replicas`` over the stacked
    ``[own block | copies in table order]`` weights of this rank; any other window
    would execute experts this rank was not assigned.
    """

    def __call__(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        packed: Sequence[int],
        rank: int,
        replicas: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_slots = selected.numel()
        local_ids = torch.empty_like(selected, dtype=torch.int32)
        weights_bf16 = torch.empty(selected.shape, dtype=torch.bfloat16, device=selected.device)
        stats = torch.empty(2 * RANKS, dtype=torch.int32, device=selected.device)
        slots = max(16, triton.next_power_of_2(num_slots))
        _hot_replica_assign_kernel[(1,)](
            selected,
            weights,
            local_ids,
            weights_bf16,
            stats,
            rank,
            num_slots,
            *packed,
            SLOTS=slots,
            R=replicas,
            EXPERTS=EXPERTS,
            RANKS=RANKS,
            BLOCK=BLOCK,
            num_warps=_num_warps(slots),
        )
        return local_ids, weights_bf16, stats

    def reference(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        packed: Sequence[int],
        rank: int,
        replicas: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The placement in plain Python over a host copy of ``selected`` and the unpacked table."""
        flat = [int(v) for v in selected.reshape(-1).tolist()]
        hot, holders, mine = unpack_table(packed, rank, replicas)
        runs_on, before, after = place(flat, hot + holders, replicas)
        window = BLOCK + replicas
        local = []
        for e in flat:
            if runs_on[e] != rank:
                local.append(window + e % (EXPERTS - window))
            elif e // BLOCK == rank:
                local.append(e - BLOCK * rank)
            else:
                local.append(BLOCK + mine.index(e))
        device = selected.device
        return (
            torch.tensor(local, dtype=torch.int32, device=device).reshape(selected.shape),
            weights.to(torch.bfloat16),
            torch.tensor(before + after, dtype=torch.int32, device=device),
        )

    def is_valid(
        self,
        selected: torch.Tensor,
        weights: torch.Tensor,
        packed: Sequence[int],
        rank: int,
        replicas: int,
    ) -> None:
        """Inputs the kernel would read past or misplace without an error."""
        assert replicas in REPLICAS, f"replicas must be one of {REPLICAS}, got {replicas}"
        assert selected.is_cuda and selected.dtype == torch.int32 and selected.is_contiguous(), (
            "selected must be contiguous int32 on the device; the kernel reads raw words"
        )
        assert selected.ndim == 2 and selected.shape[1] == TOP_K, (
            f"selected must be [M, {TOP_K}], got {tuple(selected.shape)}"
        )
        assert 0 < selected.numel() <= MAX_SLOTS, (
            f"one program covers at most {MAX_SLOTS} slots (M <= {MAX_ROWS}), "
            f"got {selected.numel()}"
        )
        assert weights.dtype == torch.float32 and weights.is_contiguous(), (
            "weights must be contiguous float32"
        )
        assert weights.shape == selected.shape, "weights and selected must have one shape"
        assert len(packed) == PACKED_WORDS and all(
            isinstance(v, int) and 0 <= v < 1 << 30 for v in packed
        ), f"a packed table is {PACKED_WORDS} host ints in [0, 2**30) words (pack_table)"
        assert 0 <= rank < RANKS, f"rank must be in [0, {RANKS}), got {rank}"


hot_replica_assign = HotReplicaAssign()
