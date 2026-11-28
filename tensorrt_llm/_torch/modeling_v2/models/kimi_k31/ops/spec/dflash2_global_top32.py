# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's global top-32 candidates per block slot: the shard-local top-32 and the
merge of the all-gathered ``(id, value)`` pairs.

``K31DFlashWorker._dflash2_global_top_k`` (the engine worker the K3.1 worker inherits) takes
``torch.topk(gen_logits, 32)`` of a full-vocab drafter head, or, under vocab tensor parallelism,
``torch.topk`` of each rank's shard, all-gathers the interleaved ``(global id, value)`` FP32 pairs
along the last dim, and takes ``torch.topk`` of the gathered values. These two entries are its
two selections; the all-gather between them stays the engine's ``allgather``.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_top32 import TOP_K, merge_top32, top32

#: The Lamport all-gather's arrival sentinel: the 32-bit word of FP32 ``-0.0`` (and of a BF16 ``+0.0``
#: followed by a BF16 ``-0.0``), delivered as ``0``.
_SENTINEL = -(2**31)


def _shard_spec(n, slots, tp, *, dtype=torch.bfloat16, values="randn", lamport=False) -> dict:
    return dict(n=n, slots=slots, tp=tp, dtype=dtype, values=values, lamport=lamport)


def _merge_spec(n, slots, tp, *, values="randn", lamport=False) -> dict:
    return dict(n=n, slots=slots, tp=tp, values=values, lamport=lamport)


def lamport_gathered_logits(logits: torch.Tensor) -> torch.Tensor:
    """BF16 logits as the Lamport vocab gather delivers them: a word ``0x80000000`` (an even ``+0.0``
    and an odd ``-0.0``) as two ``+0.0`` (``LamportVocabGather``'s exception)."""
    words = logits.contiguous().view(torch.int32)
    return torch.where(words == _SENTINEL, torch.zeros_like(words), words).view(logits.dtype)


def decode_lamport_pairs(gathered: torch.Tensor) -> torch.Tensor:
    """Gathered Lamport-mode pairs with every ``(-(id + 1), +0.0)`` read back as ``(id, -0.0)``, the
    layout (and strides) of ``gathered`` kept."""
    ids, values = gathered[..., 0::2], gathered[..., 1::2]
    signed = ids < 0
    out = gathered.clone()
    out[..., 0::2] = torch.where(signed, -1.0 - ids, ids)
    out[..., 1::2] = torch.where(signed, torch.full_like(values, -0.0), values)
    return out


class K31DFlash2ShardTop32(OpWrapper):
    """``torch.topk(gen_logits, 32, dim=-1)`` of the drafter logits, in torch's order.

    Args:
        gen_logits: ``[n, K, width]`` BF16 / FP32 contiguous logits: the full vocabulary
            (``width == vocab``) or this rank's shard of it.
        top_k: 32.
        id_offset: the shard's first global id (``tp_rank * width``), added to the ids.
        pairs: return the interleaved FP32 ``(global id, value)`` pairs ``[n, K, 64]`` of the
            vocab-parallel exchange (``torch.stack([ids.float(), values.float()], -1)
            .flatten(-2)``) instead of ``(values [n, K, 32] in the logits dtype, ids [n, K, 32]
            int64)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="full vocab, one request, K = 4 (4 rows)", spec=_shard_spec(1, 4, 1)),
        Cell(why="full vocab, one request, K = 7", spec=_shard_spec(1, 7, 1)),
        Cell(why="full vocab, 8 requests, K = 7", spec=_shard_spec(8, 7, 1)),
        Cell(why="full vocab, 64 requests, K = 7 (448 rows)", spec=_shard_spec(64, 7, 1)),
        Cell(why="full vocab, FP32 logits", spec=_shard_spec(8, 4, 1, dtype=torch.float32)),
        Cell(
            why="full vocab, FP32 logits (the logits processor's dtype), 64 requests, K = 7",
            spec=_shard_spec(64, 7, 1, dtype=torch.float32),
        ),
        Cell(
            why="full vocab, FP32 logits, tied values",
            spec=_shard_spec(8, 7, 1, dtype=torch.float32, values="ties"),
        ),
        Cell(why="TP8 shard, FP32 logits, pairs", spec=_shard_spec(64, 7, 8, dtype=torch.float32)),
        Cell(
            why="TP2 shard, FP32 logits, pairs, ties",
            spec=_shard_spec(8, 4, 2, dtype=torch.float32, values="ties"),
        ),
        Cell(
            why="full vocab, tied logits: the network's order among equal values",
            spec=_shard_spec(8, 7, 1, values="ties"),
        ),
        Cell(
            why="full vocab, NaN, +-inf and +-0 among the logits",
            spec=_shard_spec(4, 7, 1, values="special"),
        ),
        Cell(why="TP2 shard (81,920), pairs", spec=_shard_spec(8, 7, 2)),
        Cell(why="TP4 shard (40,960), pairs, ties", spec=_shard_spec(16, 4, 4, values="ties")),
        Cell(why="TP8 shard (20,480), pairs", spec=_shard_spec(64, 7, 8)),
        Cell(why="TP16 shard (10,240), pairs, 1 request", spec=_shard_spec(1, 7, 16)),
        Cell(
            why="TP32 shard (5,120): one CTA a row, pairs, ties",
            spec=_shard_spec(32, 7, 32, values="ties"),
        ),
        Cell(
            why="TP32 shard, FP32 logits, specials",
            spec=_shard_spec(4, 4, 32, dtype=torch.float32, values="special"),
        ),
        Cell(
            why="sm_107 TEP8 generation: TP8 shard, FP32 logits, pairs, 1 request, K = 7",
            spec=_shard_spec(1, 7, 8, dtype=torch.float32),
        ),
        Cell(
            why="sm_107 DEP16 generation: full vocab, FP32 logits, 12 requests a rank, K = 7",
            spec=_shard_spec(12, 7, 1, dtype=torch.float32),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: full vocab, FP32 logits, 16 requests, K = 5",
            spec=_shard_spec(16, 5, 1, dtype=torch.float32),
        ),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_top32.py``. One CTA of 256 threads selects among up to 8,192
    elements of a row (radix select of the 32nd largest key, one block scan, the gathered order,
    the 32-element bitonic network of torch's ``sortKeyValueInplace``); a wider row is split
    into 8,192-element chunks whose index-ordered top-32 a second launch merges. One launch for a
    shard of at most 8,192 (TP32), two otherwise. Bitwise: values (bits, NaN payloads included),
    ids and their order equal ``torch.topk``'s on every cell (torch 2.14).
    """

    def __call__(
        self, gen_logits: torch.Tensor, top_k: int, id_offset: int, pairs: bool
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        n, slots, width = gen_logits.shape
        out = top32(gen_logits.view(n * slots, width), id_offset=id_offset, pairs=pairs)
        if pairs:
            return out.view(n, slots, 2 * TOP_K)
        values, ids = out
        return values.view(n, slots, TOP_K), ids.view(n, slots, TOP_K)

    def reference(
        self, gen_logits: torch.Tensor, top_k: int, id_offset: int, pairs: bool
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``K31DFlashWorker._dflash2_global_top_k``'s selections before the all-gather."""
        local_values, local_ids = torch.topk(gen_logits, top_k, dim=-1)
        local_ids = local_ids + id_offset
        if not pairs:
            return local_values, local_ids
        return torch.stack([local_ids.float(), local_values.float()], dim=-1).flatten(-2)

    def is_valid(self, gen_logits: torch.Tensor, top_k: int, id_offset: int, pairs: bool) -> None:
        assert top_k == TOP_K, f"the kernel selects the top {TOP_K}"
        assert gen_logits.dtype in (torch.bfloat16, torch.float32), "BF16 or FP32 logits"
        assert gen_logits.is_contiguous(), "the kernel reads contiguous rows"
        width = gen_logits.shape[-1]
        assert width >= TOP_K and width % 8 == 0, "rows of whole 16-byte vectors, >= 32 values"
        assert id_offset + width < (1 << 24), "pair ids must be exact in FP32"
        n, slots = gen_logits.shape[:2]
        if gen_logits.dtype == torch.float32 and width == 256:
            assert n * slots < 152, "torch sorts this shape stably: not certified"


class K31DFlash2ShardIndexPairs(OpWrapper):
    """This rank's top-32 of its shard of the drafter logits as interleaved FP32 ``(global id,
    value)`` pairs in index order: ``torch.topk(gen_logits, 32)``'s selection (the lowest index
    first among values equal to the 32nd) sorted by id.

    All-gathered in rank order, every rank's pairs are the full row's top-32 candidates in global
    index order, and ``k31_dflash2_merge_top32`` of them is exactly ``k31_dflash2_shard_top32`` of
    the all-gathered FP32 logits (values, ids and their order, ties included): the candidates of
    the gathered-logits path without the vocabulary all-gather.

    ``lamport=True`` is the operand of the Lamport all-gather (``lamport_allgather``), which delivers
    the word ``0x80000000`` (FP32 ``-0.0``) as ``+0.0``: the selection reads BF16 logits as the
    Lamport vocab gather delivers them (a word ``0x80000000``, an even ``+0.0`` and an odd ``-0.0``,
    as two ``+0.0``: ``lamport_gathered_logits``), and a selected ``-0.0`` is sent as ``(-(id + 1),
    +0.0)``; ``k31_dflash2_merge_top32(..., lamport=True)`` reads it back. Merged, the pairs are
    exactly the top-32 of the Lamport-gathered FP32 logits, signed zeros included.

    Args:
        gen_logits: ``[n, K, width]`` BF16 / FP32 contiguous: this rank's shard.
        id_offset: the shard's first global id (``tp_rank * width``).
        lamport: the Lamport exchange mode.
    Returns:
        ``[n, K, 64]`` FP32 pairs.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP32 shard (5,120), one request, K = 7: the ll block", spec=_shard_spec(1, 7, 32)
        ),
        Cell(
            why="TP32 shard, 4 requests, K = 7, ties",
            spec=_shard_spec(4, 7, 32, values="ties"),
        ),
        Cell(
            why="TP16 shard (10,240: two chunks), 2 requests, K = 4: the mid block",
            spec=_shard_spec(2, 4, 16),
        ),
        Cell(why="TP16 shard, 12 requests, K = 4: the M2 block", spec=_shard_spec(12, 4, 16)),
        Cell(
            why="TP16 shard, 24 requests, K = 4, ties",
            spec=_shard_spec(24, 4, 16, values="ties"),
        ),
        Cell(
            why="TP16 shard, NaN, +-inf and +-0",
            spec=_shard_spec(4, 4, 16, values="special"),
        ),
        Cell(
            why="TP32 shard, FP32 logits, specials",
            spec=_shard_spec(2, 7, 32, dtype=torch.float32, values="special"),
        ),
        Cell(why="TP8 shard (20,480: three chunks), K = 7", spec=_shard_spec(8, 7, 8)),
        Cell(
            why="Lamport mode, TP8 shard, 7 requests, K = 4: the sampled TEP8 B7 block",
            spec=_shard_spec(7, 4, 8, lamport=True),
        ),
        Cell(
            why="Lamport mode, TP16 shard, 16 requests, K = 4: the sampled TEP16 B16 block",
            spec=_shard_spec(16, 4, 16, lamport=True),
        ),
        Cell(
            why="Lamport mode, TP8 shard, signed zeros and sentinel words at the 32nd",
            spec=_shard_spec(4, 4, 8, values="zeros", lamport=True),
        ),
        Cell(
            why="Lamport mode, TP32 shard (one launch), signed zeros at the 32nd",
            spec=_shard_spec(4, 7, 32, values="zeros", lamport=True),
        ),
        Cell(
            why="Lamport mode, TP16 shard, NaN, +-inf and +-0",
            spec=_shard_spec(4, 4, 16, values="special", lamport=True),
        ),
        Cell(
            why="Lamport mode, TP8 shard, FP32 logits with signed zeros at the 32nd",
            spec=_shard_spec(2, 4, 8, dtype=torch.float32, values="zeros", lamport=True),
        ),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_top32.py`` (``top32(..., pairs=True, index_order=True)``): the
    selection kernel writing its 32 in index order, unsorted; a shard over 8,192 values takes the
    chunk launch first (index-ordered chunk candidates, then the same selection over them). One
    launch at TP32, two at TP16. Bitwise to the reference; the merge identity is the entry's test.
    """

    def __call__(
        self, gen_logits: torch.Tensor, id_offset: int, lamport: bool = False
    ) -> torch.Tensor:
        n, slots, width = gen_logits.shape
        out = top32(
            gen_logits.view(n * slots, width),
            id_offset=id_offset,
            pairs=True,
            index_order=True,
            lamport=lamport,
        )
        return out.view(n, slots, 2 * TOP_K)

    def reference(
        self, gen_logits: torch.Tensor, id_offset: int, lamport: bool = False
    ) -> torch.Tensor:
        """``torch.topk``'s selection, sorted by index, as ``(id, value)`` FP32 pairs (Lamport mode:
        of the Lamport-gathered BF16 logits, a ``-0.0`` value sent as ``(-(id + 1), +0.0)``)."""
        if lamport and gen_logits.dtype == torch.bfloat16:
            gen_logits = lamport_gathered_logits(gen_logits)
        values, ids = torch.topk(gen_logits, TOP_K, dim=-1)
        order = ids.argsort(dim=-1)
        ids = ids.gather(-1, order) + id_offset
        values = values.gather(-1, order).float()
        if lamport:
            signed = values.view(torch.int32) == _SENTINEL
            ids = torch.where(signed, -1 - ids, ids)
            values = torch.where(signed, torch.zeros_like(values), values)
        return torch.stack([ids.float(), values], dim=-1).flatten(-2)

    def is_valid(self, gen_logits: torch.Tensor, id_offset: int, lamport: bool = False) -> None:
        assert gen_logits.dtype in (torch.bfloat16, torch.float32), "BF16 or FP32 logits"
        assert gen_logits.is_contiguous(), "the kernel reads contiguous rows"
        width = gen_logits.shape[-1]
        assert width >= TOP_K and width % 8 == 0, "rows of whole 16-byte vectors, >= 32 values"
        assert id_offset + width < (1 << 24), "pair ids must be exact in FP32"


class K31DFlash2MergeTop32(OpWrapper):
    """The global top-32 of the all-gathered pairs: ``torch.topk(gathered[..., 1::2], 32)`` and
    the ids at the selected positions.

    Args:
        gathered: ``[n, K, tp * 64]`` FP32, every rank's interleaved ``(global id, value)`` pairs
            in rank order (the engine ``allgather`` along the last dim).
        top_k: 32.
        lamport: the pairs are ``k31_dflash2_shard_index_pairs(..., lamport=True)``'s: a pair
            ``(-(id + 1), +0.0)`` is read as ``(id, -0.0)`` first (``decode_lamport_pairs``).
    Returns:
        ``(unary_logits [n, K, 32] FP32, candidate_ids [n, K, 32] int64)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(why="TP2, one request, K = 4", spec=_merge_spec(1, 4, 2)),
        Cell(why="TP2, 64 requests, K = 7", spec=_merge_spec(64, 7, 2)),
        Cell(why="TP4, 8 requests, K = 7, ties", spec=_merge_spec(8, 7, 4, values="ties")),
        Cell(why="TP8, 1 request, K = 7", spec=_merge_spec(1, 7, 8)),
        Cell(
            why="TP8, 38 requests, K = 4: 152 rows of 256 values, still the network",
            spec=_merge_spec(38, 4, 8, values="ties"),
        ),
        Cell(why="TP8, 64 requests, K = 7", spec=_merge_spec(64, 7, 8)),
        Cell(why="TP16, 17 requests, K = 7", spec=_merge_spec(17, 7, 16)),
        Cell(why="TP32, 1 request, K = 7", spec=_merge_spec(1, 7, 32)),
        Cell(why="TP32, 64 requests, K = 7, ties", spec=_merge_spec(64, 7, 32, values="ties")),
        Cell(why="TP4, NaN, +-inf and +-0", spec=_merge_spec(4, 7, 4, values="special")),
        Cell(
            why="Lamport mode, TP8, 7 requests, K = 4: the sampled TEP8 B7 block",
            spec=_merge_spec(7, 4, 8, lamport=True),
        ),
        Cell(
            why="Lamport mode, TP16, 16 requests, K = 4: the sampled TEP16 B16 block",
            spec=_merge_spec(16, 4, 16, lamport=True),
        ),
        Cell(
            why="Lamport mode, TP8, signed zeros and sentinel words at the 32nd",
            spec=_merge_spec(4, 4, 8, values="zeros", lamport=True),
        ),
        Cell(
            why="Lamport mode, TP32, signed zeros at the 32nd",
            spec=_merge_spec(4, 7, 32, values="zeros", lamport=True),
        ),
    )

    note = """
    Kernel: ``kernels/spec/dflash2_top32.py``, one launch, one CTA per row. The torch path selects
    from the strided view ``gathered[..., 1::2]``, which torch 2.14 sorts by its bitonic network
    for every row count (a contiguous 256-wide FP32 input of 152+ rows would be sorted stably;
    see ``torch_topk_sorts_stably``). Bitwise on every cell; ties at the 32nd value go to the
    lowest gathered position (the lowest rank first), as in torch.
    """

    def __call__(
        self, gathered: torch.Tensor, top_k: int, lamport: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, slots, cols = gathered.shape
        values, ids = merge_top32(gathered.view(n * slots, cols), lamport=lamport)
        return values.view(n, slots, TOP_K), ids.view(n, slots, TOP_K)

    def reference(
        self, gathered: torch.Tensor, top_k: int, lamport: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``K31DFlashWorker._dflash2_global_top_k``'s selection after the all-gather (of the decoded
        pairs in Lamport mode)."""
        if lamport:
            gathered = decode_lamport_pairs(gathered)
        unary_logits, selected = torch.topk(gathered[..., 1::2], top_k, dim=-1)
        candidate_ids = gathered[..., 0::2].gather(-1, selected).long()
        return unary_logits, candidate_ids

    def is_valid(self, gathered: torch.Tensor, top_k: int, lamport: bool = False) -> None:
        assert top_k == TOP_K, f"the kernel selects the top {TOP_K}"
        assert gathered.dtype == torch.float32, "FP32 pairs"
        assert gathered.is_contiguous(), "the kernel reads contiguous rows"
        cols = gathered.shape[-1]
        assert cols % 2 == 0 and TOP_K <= cols // 2 <= 8192, "32..8192 pairs per row"


k31_dflash2_shard_top32 = K31DFlash2ShardTop32()
k31_dflash2_shard_index_pairs = K31DFlash2ShardIndexPairs()
k31_dflash2_merge_top32 = K31DFlash2MergeTop32()

__all__ = [
    "decode_lamport_pairs",
    "lamport_gathered_logits",
    "K31DFlash2MergeTop32",
    "K31DFlash2ShardIndexPairs",
    "K31DFlash2ShardTop32",
    "k31_dflash2_merge_top32",
    "k31_dflash2_shard_index_pairs",
    "k31_dflash2_shard_top32",
]
