# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's greedy-walk tail under TP16 vocabulary parallelism with one Lamport hop:
the shard top-32 published by multicast, the merge, the selector scores and the walk in one
clustered kernel, and the draft agreement as rank 0's drafts polled by the next-token kernel.

Entries (kernels in ``kernels/spec/rubin_ull_dflash2_rank_topk_walk.py``):

* ``k31_dflash2_rank_top32_publish``: this rank's ``(global id, value)`` pairs of its shard's
  top-32 into its columns of every rank's exchange row;
* ``k31_dflash2_selector_walk_mc``: the greedy tokens of one request's block from the exchanged
  pairs (and, with ``publish_drafts``, rank 0's tokens published for the agreement);
* ``k31_draft_agreed_rewind``: rank 0's published tokens as every rank's next drafts and next
  tokens, with the drafter's KV rewind.

``RankTopKWalk`` is the worker's runtime: the exchange (the pair exchange's MNNVL Lamport
workspace), the scratch and the compiled kernels, bound once outside graph capture.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.comm import lamport_ar as _lam
from ...kernels.spec import dflash2_top32 as _t32
from ...kernels.spec import rubin_ull_dflash2_rank_topk_walk as _k
from .dflash2_global_top32 import k31_dflash2_merge_top32
from .dflash2_selector_greedy_tokens import k31_dflash2_selector_greedy_tokens

#: Lamport arrival sentinel (FP32 ``-0.0``) as an int32 word.
SENTINEL = -(2**31)


@dataclasses.dataclass(frozen=True)
class PairExchange:
    """Device addresses of the pair exchange's Lamport workspace: ``uc_local`` this rank's buffers,
    ``mc_base`` their multicast mapping (``multicast``) or, in a single-GPU emulation, the same
    unicast buffers; ``flags`` the rotation state shared with the engine's MNNVL calls."""

    world: int
    rank: int
    uc_local: int
    mc_base: int
    flags: torch.Tensor
    bytes_per_buffer: int
    multicast: bool

    @staticmethod
    def from_mnnvl(workspace: dict, world: int, rank: int) -> "PairExchange":
        view = _lam.LamportWorkspace.from_mnnvl(workspace, world, rank)
        return PairExchange(
            world, rank, view.uc_local, view.mc_base, view.flags, view.bytes_per_buffer, True
        )

    @staticmethod
    def emulated(
        world: int, rank: int, device, bytes_per_buffer: int = 1 << 20
    ) -> Tuple["PairExchange", torch.Tensor]:
        """A one-GPU stand-in (tests): three sentinel-filled buffers and a fresh rotation state;
        the other ranks' rows are written into ``buffer`` by the caller. Returns ``(exchange,
        buffer)`` (int32 words, ``3 * bytes_per_buffer / 4``)."""
        buf = torch.full(
            (_lam.NUM_BUFFERS * bytes_per_buffer // 4,), SENTINEL, dtype=torch.int32, device=device
        )
        flags = torch.tensor(
            [0, 2, bytes_per_buffer, 1, 0, 0, 0, 0, 0], dtype=torch.int32, device=device
        )
        ptr = buf.data_ptr()
        return PairExchange(world, rank, ptr, ptr, flags, bytes_per_buffer, False), buf


def published_pairs(logits: torch.Tensor, id_offset: int) -> torch.Tensor:
    """The int32 words a rank publishes for ``logits [rows, width]``: ``top32(pairs=True)`` with
    ``-0.0`` as ``+0.0`` (what the zero-padded all-reduce of the engine path delivers too)."""
    words = _t32.top32(logits, id_offset=id_offset, pairs=True).view(torch.int32).clone()
    words[words == SENTINEL] = 0
    return words


def _spec(why: str, **spec) -> Cell:
    base = dict(world=16, rows=7, shard=10240, pitch=10496, values="randn", rank=5)
    base.update(spec)
    return Cell(why=why, spec=base)


class K31DFlash2RankTop32Publish(OpWrapper):
    """This rank's top-32 pairs of each row of its BF16 vocabulary shard (``top32(logits,
    id_offset, pairs=True)``, ``-0.0`` sent as ``+0.0``) into its 64 columns of the row in every
    rank's current Lamport buffer of ``exchange``.

    Args:
        exchange: ``PairExchange``.
        logits: ``[rows, width]`` BF16, width over 8,192 (two chunks or more), packed or the
            leading columns of a wider buffer's rows (16-byte pitch), read in place.
        id_offset: the shard's first global id (``rank * width``).
        scratch: ``(keys, indices, bits, counters, dummy)`` from ``rank_top32_scratch``; the
            counters are zero between calls.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        _spec("helix16 b1: one request's 7 draft rows, TP16 shard read from rows of 10,496"),
        _spec("ties in the shard (the lowest index first)", values="ties"),
        _spec("NaN, +-inf and +-0 among the logits", values="special"),
        _spec("rank 0, packed rows", rank=0, pitch=10240),
        _spec("rank 15, one row", rank=15, rows=1),
    )
    note = """
    One launch, grid (chunks, rows): each CTA selects its chunk's top-32 (``dflash2_top32``'s
    chunk pass); the last CTA of a row to arrive selects and sorts the shard's 32 from the chunks'
    candidates (its second pass) and publishes them. Bitwise ``top32``'s pairs. The Lamport state
    is read after ``griddepcontrol.wait``; the call that polls the rows (the walk) advances it.
    """

    def __call__(self, exchange, logits, id_offset, scratch) -> None:
        rows, width = logits.shape
        keys, idx, bits, counters, dummy = scratch
        full = _t32._pitched_rows(logits)
        _k.compiled_rank_top32(
            exchange.world, _k.rank_chunks(width), full.shape[1] // 2, exchange.multicast
        )(
            full.view(torch.int32),
            keys[:rows],
            idx[:rows],
            bits[:rows],
            counters[:rows],
            dummy[:rows],
            width,
            id_offset,
            exchange.rank,
            exchange.mc_base,
            exchange.flags.data_ptr(),
        )

    def reference(self, exchange, logits, id_offset, scratch) -> torch.Tensor:
        """The ``[rows, 64]`` int32 words that land in the rank's columns of each row."""
        return published_pairs(logits, id_offset)

    def is_valid(self, exchange, logits, id_offset, scratch) -> None:
        rows, width = logits.shape
        assert logits.dtype == torch.bfloat16, "BF16 logits"
        assert _k.rank_chunks(width) >= 2, "a shard over one 8,192-element chunk"
        assert width % 8 == 0 and logits.stride(1) == 1, "rows of whole 16-byte vectors"
        assert id_offset + width < (1 << 24), "pair ids must be exact in FP32"
        assert scratch[0].shape[0] >= rows and scratch[0].shape[1] == _k.rank_chunks(width) * 32
        assert _k.call_bytes(rows, 1, exchange.world) <= exchange.bytes_per_buffer


def rank_top32_scratch(max_rows: int, width: int, device) -> tuple:
    """The publish kernel's scratch: chunk candidates (key, index, bits), zeroed row counters."""
    m = _k.rank_chunks(width) * 32
    return (
        torch.empty(max_rows, m, dtype=torch.int32, device=device),
        torch.empty(max_rows, m, dtype=torch.int32, device=device),
        torch.empty(max_rows, m, dtype=torch.int32, device=device),
        torch.zeros(max_rows, dtype=torch.int32, device=device),
        torch.empty(max_rows, 1, dtype=torch.int32, device=device),
    )


def _walk_spec(why: str, **spec) -> Cell:
    base = dict(world=16, slots=7, rank=5, values="randn", block=torch.bfloat16, publish=False)
    base.update(spec)
    return Cell(why=why, spec=base)


class K31DFlash2SelectorWalkMc(OpWrapper):
    """The greedy draft tokens of one request's block (``[1, K]`` int32) from the pairs every rank
    published into the current Lamport buffer of ``exchange``: ``k31_dflash2_merge_top32`` of
    each row's ``world x 64`` words, then ``k31_dflash2_selector_greedy_tokens`` with the given
    gate. Advances the exchange's rotation state and re-arms the previous call's buffer. With
    ``publish_drafts`` rank 0 also writes its tokens after the rows (``k31_draft_agreed_rewind``).

    Args:
        exchange: ``PairExchange``.
        predecessor_codebook, successor_codebook: ``[vocab, 256]`` BF16.
        gate: ``[K, 256]`` BF16, unit column stride, any 16-byte row stride (the head bank's
            selector columns, read in place).
        anchor_token_ids: ``[1]`` int32 / int64.
        block_dtype: the dtype of the block the tokens stand for (BF16 / FP32).
        publish_drafts: rank 0 publishes its tokens for the agreement.
        tf32_round: the scores' FP32 -> TF32 operand rounding (``TF32_ROUNDINGS``).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        _walk_spec("helix16 b1: K 7, 16 ranks, BF16 block (the served greedy walk)"),
        _walk_spec("rank 0 publishing its drafts", rank=0, publish=True),
        _walk_spec(
            "rank 9 with the agreement payload on (rank 0 alone writes it)", rank=9, publish=True
        ),
        _walk_spec("ties among the logits and the scores", values="ties"),
        _walk_spec("FP32 block", block=torch.float32),
        _walk_spec("K 4", slots=4),
        _walk_spec("K 8", slots=8),
    )
    note = """
    One launch: a cluster of K CTAs (CTA e: poll row e, merge it, gather the codebook rows of its
    candidates and, through distributed shared memory, of slot e - 1's, the 32 x 32 TF32 scores by
    mma.sync) whose CTA 0 walks; four more clusters re-arm the previous Lamport call's buffer.
    Candidates and unary logits bitwise ``merge_top32``'s; scores TF32 x TF32 with FP32
    accumulation (bitwise cuBLAS's TF32 einsum where its operand rounding is RNE and its K order
    is 8 a step, e.g. SM90; elsewhere within its rounding); the walk is the engine walk.
    """

    def __call__(
        self,
        exchange,
        predecessor_codebook,
        successor_codebook,
        gate,
        anchor_token_ids,
        block_dtype,
        publish_drafts=False,
        tf32_round="rne",
        debug=None,
    ) -> torch.Tensor:
        slots = gate.shape[0]
        device = gate.device
        tokens = torch.empty((1, slots), dtype=torch.int32, device=device)
        if debug is None:
            cand = unary = torch.empty((slots, 32), dtype=torch.int32, device=device)
            scores = torch.empty((slots, 32, 32), dtype=torch.float32, device=device)
        else:
            cand, unary, scores = debug
        _k.compiled_walk(
            slots,
            exchange.world,
            block_dtype == torch.bfloat16,
            tf32_round,
            publish_drafts,
            exchange.multicast,
            predecessor_codebook.dtype,
            gate.dtype,
            anchor_token_ids.dtype,
            debug is not None,
        )(
            predecessor_codebook,
            successor_codebook,
            gate,
            anchor_token_ids,
            tokens,
            cand,
            unary,
            scores,
            exchange.rank,
            exchange.uc_local,
            exchange.mc_base,
            exchange.flags.data_ptr(),
        )
        return tokens

    def reference(
        self,
        gathered,
        predecessor_codebook,
        successor_codebook,
        gate,
        anchor_token_ids,
        block_dtype,
        publish_drafts=False,
        tf32_round="rne",
        debug=None,
    ) -> torch.Tensor:
        """The engine path on the gathered rows: ``gathered`` ``[K, world * 64]`` FP32 (every
        rank's published words in rank order) in place of the exchange."""
        slots = gate.shape[0]
        unary, ids = k31_dflash2_merge_top32(gathered.view(1, slots, -1).contiguous(), 32)
        return k31_dflash2_selector_greedy_tokens.reference(
            predecessor_codebook,
            successor_codebook,
            torch.empty((gate.shape[1], 1), dtype=gate.dtype, device=gate.device),
            ids,
            unary,
            None,
            anchor_token_ids,
            block_dtype,
            gate=gate.view(1, slots, -1),
        )

    def is_valid(
        self,
        exchange,
        predecessor_codebook,
        successor_codebook,
        gate,
        anchor_token_ids,
        block_dtype,
        publish_drafts=False,
        tf32_round="rne",
        debug=None,
    ) -> None:
        slots, rank = gate.shape
        assert 2 <= slots <= _k.MAX_SLOTS, f"2..{_k.MAX_SLOTS} block rows"
        assert rank == _k.RANK and predecessor_codebook.shape[1] == _k.RANK, "selector rank 256"
        assert predecessor_codebook.dtype == successor_codebook.dtype == torch.bfloat16
        assert predecessor_codebook.is_contiguous() and successor_codebook.is_contiguous()
        assert gate.dtype == torch.bfloat16 and gate.stride(1) == 1
        assert not gate.data_ptr() % 16 and not (gate.stride(0) * 2) % 16, "16-byte gate rows"
        assert anchor_token_ids.numel() == 1, "one request"
        assert block_dtype in (torch.bfloat16, torch.float32)
        assert tf32_round in _k.TF32_ROUNDINGS
        assert exchange.world * 64 <= 4 * _t32.THREADS, "a row is one vector a thread"
        assert _k.call_bytes(slots, 1, exchange.world) <= exchange.bytes_per_buffer


class K31DraftAgreedRewind(OpWrapper):
    """``(next_draft_tokens [batch, K] int32, next_new_tokens [batch, K + 1])`` from rank 0's
    drafts the last walk call published (``publish_drafts``), with the drafter's ``kv_lens``
    rewind: ``draft_next_new_tokens_rewind`` then the engine agreement (``k31_draft_agreed_tokens``
    of the zero-padded broadcast of rank 0's drafts), for ``batch`` generation requests and no
    context rows.

    ``next_new[b] = [accepted[b, num_accepted[b] - 1], *drafts0[b]]``, ``next_draft = drafts0``;
    in place ``kv_lens[g] = max(kv_lens[g] - rewind[g], 0)``. ``next_new`` takes the promoted
    dtype of ``accepted_tokens`` and int32.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(why="served b1: K 7, rows 7, int32 tokens", spec=dict(slots=7, acc=torch.int32)),
        Cell(why="int64 accepted tokens", spec=dict(slots=7, acc=torch.int64)),
        Cell(why="K 4, a rewind past zero", spec=dict(slots=4, acc=torch.int32, clamp=True)),
    )
    note = """
    One CTA of 32 threads after ``griddepcontrol.wait``: the walk grid has advanced the rotation
    state, which names the walk call's buffer as the dirty one (cleared only at the next call on
    the workspace), and rank 0's drafts are polled after its ``rows`` exchange rows. Integer copies
    and arithmetic: the values of the unfused ops.
    """

    def __call__(self, exchange, rows, accepted_tokens, num_accepted_tokens, kv_lens, rewind):
        batch = rewind.shape[0]
        slots = accepted_tokens.shape[1] - 1
        device = accepted_tokens.device
        dtype = torch.promote_types(accepted_tokens.dtype, torch.int32)
        drafts = torch.empty((batch, slots), dtype=torch.int32, device=device)
        new = torch.empty((batch, slots + 1), dtype=dtype, device=device)
        _k.compiled_agreed_rewind(
            slots,
            rows,
            exchange.world,
            accepted_tokens.dtype,
            dtype,
            kv_lens.dtype,
            rewind.dtype,
            num_accepted_tokens.dtype,
        )(
            accepted_tokens[:batch],
            num_accepted_tokens,
            kv_lens,
            rewind,
            drafts,
            new,
            exchange.uc_local,
            exchange.flags.data_ptr(),
        )
        return drafts, new

    def reference(self, drafts0, accepted_tokens, num_accepted_tokens, kv_lens, rewind):
        """``(next_draft, next_new, kv_lens after)`` from rank 0's ``drafts0 [batch, K]``;
        ``kv_lens`` itself is not modified."""
        batch = drafts0.shape[0]
        dtype = torch.promote_types(accepted_tokens.dtype, torch.int32)
        rows = torch.arange(batch, device=drafts0.device)
        last = accepted_tokens[rows, num_accepted_tokens[:batch].long() - 1]
        drafts = drafts0.to(torch.int32)
        new = torch.cat([last[:, None].to(dtype), drafts.to(dtype)], dim=1)
        kv = kv_lens.clone()
        kv[:batch] = (kv[:batch] - rewind.to(kv.dtype)).clamp(min=0)
        return drafts, new, kv

    def is_valid(self, exchange, rows, accepted_tokens, num_accepted_tokens, kv_lens, rewind):
        batch = rewind.shape[0]
        assert batch >= 1 and accepted_tokens.shape[0] >= batch, "one accepted row a request"
        assert accepted_tokens.stride(-1) == 1, "accepted tokens with unit column stride"
        assert accepted_tokens.shape[1] - 1 <= _k.DRAFT_WORDS, "K up to the payload's words"
        assert kv_lens.shape[0] >= batch and kv_lens.is_contiguous() and rewind.is_contiguous()
        assert _k.call_bytes(rows, batch, exchange.world) <= exchange.bytes_per_buffer


k31_dflash2_rank_top32_publish = K31DFlash2RankTop32Publish()
k31_dflash2_selector_walk_mc = K31DFlash2SelectorWalkMc()
k31_draft_agreed_rewind = K31DraftAgreedRewind()


class RankTopKWalk:
    """The worker's N5 runtime for one request's block of ``slots`` rows: the exchange, the
    publish kernel's scratch and the compiled kernels (``prepare``: outside graph capture)."""

    def __init__(self, exchange: PairExchange, slots: int, shard: int, device) -> None:
        self.exchange = exchange
        self.slots = slots
        self.shard = shard
        self.scratch = rank_top32_scratch(_k.MAX_SLOTS, shard, device)
        self.ready: set = set()

    @staticmethod
    def serves(world: int, slots: int, shard: int, bytes_per_buffer: int) -> bool:
        return (
            world * 64 <= 4 * _t32.THREADS
            and 2 <= slots <= _k.MAX_SLOTS
            and _k.rank_chunks(shard) >= 2
            and shard % 8 == 0
            and _k.call_bytes(slots, 1, world) <= bytes_per_buffer
        )

    def prepare(self, pitch: int, block_dtype, publish: bool, cb_dtype, anchor_dtype) -> None:
        """Compile the two kernels (and the agreement's) for this block's layout."""
        ex = self.exchange
        _k.compiled_rank_top32(ex.world, _k.rank_chunks(self.shard), pitch // 2, ex.multicast)
        _k.compiled_walk(
            self.slots,
            ex.world,
            block_dtype == torch.bfloat16,
            "rne",
            publish,
            ex.multicast,
            cb_dtype,
            torch.bfloat16,
            anchor_dtype,
            False,
        )
        self.ready.add((pitch, block_dtype, publish, cb_dtype, anchor_dtype))

    def is_ready(self, pitch: int, block_dtype, publish: bool, cb_dtype, anchor_dtype) -> bool:
        return (pitch, block_dtype, publish, cb_dtype, anchor_dtype) in self.ready

    def tokens(
        self,
        logits: torch.Tensor,
        gate: torch.Tensor,
        predecessor_codebook: torch.Tensor,
        successor_codebook: torch.Tensor,
        anchor: torch.Tensor,
        block_dtype,
        publish: bool,
    ) -> torch.Tensor:
        """``[1, K]`` int32 greedy tokens: publish this rank's pairs, then the walk."""
        ex = self.exchange
        k31_dflash2_rank_top32_publish(ex, logits, ex.rank * self.shard, self.scratch)
        return k31_dflash2_selector_walk_mc(
            ex, predecessor_codebook, successor_codebook, gate, anchor, block_dtype, publish
        )


__all__ = [
    "K31DFlash2RankTop32Publish",
    "K31DFlash2SelectorWalkMc",
    "K31DraftAgreedRewind",
    "PairExchange",
    "RankTopKWalk",
    "SENTINEL",
    "k31_draft_agreed_rewind",
    "k31_dflash2_rank_top32_publish",
    "k31_dflash2_selector_walk_mc",
    "published_pairs",
    "rank_top32_scratch",
]
