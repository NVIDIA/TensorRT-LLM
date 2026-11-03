# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Greedy and temperature-only target sampling from vocabulary-sharded logits, without gathering
them.

A column-parallel LM head leaves each tensor-parallel rank with the logits of one vocabulary
slice. Each rank reduces its slice to one ``(value, global index)`` candidate per row and the
ranks exchange those instead of the logits; the tokens equal ``torch.argmax`` of the gathered
logits (largest value, ties to the smallest index, the first NaN wins). A temperature-only row
takes the argmax of ``logits / T + G`` (Gumbel-max, an exact draw from ``softmax(logits / T)``)
with i.i.d. Gumbel noise ``G`` keyed by the row's Philox seed and RNG offset and by the global
vocabulary index, so every rank perturbs its columns as the full vocabulary would be.

One launch per call: programs ``(row, chunk)`` write the winner of their chunk; the last program
to finish reduces them per row, stores one 8-byte entry per row into this rank's column of every
rank's exchange slot (:class:`~tensorrt_llm._torch.distributed.peer_buffers.PeerBuffers`), waits
for the entries of all ranks and writes the tokens. An entry holds the value bits, the global
index and the launch counter as a tag; every rank launches the same samplers in the same order,
so the tags agree, and a reader takes only entries of the current launch. Three slots rotate and
each launch zeroes (tag 0) the slot of the launch before.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorrt_llm.mapping import Mapping

from ..pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

# Rows per launch; sizes the exchange buffer.
MAX_ROWS = 256
_BLOCK = 2048
_NUM_SLOTS = 3
_NUM_WARPS = 8
_NO_INDEX = 2147483647
# Entry: low word the value bits; high word (launch tag << 18) | global index. Tags run
# 1 .. _TAG_MOD (a multiple of the slot count) and wrap; an index of _INDEX_MASK is no candidate.
_INDEX_BITS = tl.constexpr(18)
_INDEX_MASK = tl.constexpr((1 << 18) - 1)
_TAG_MOD = tl.constexpr((1 << 14) - 1)
_MAX_VOCAB = (1 << 18) - 1


@triton.jit
def _prefer_argmax(a, a_index, b, b_index):
    """``torch.argmax`` order: NaN first, then the larger value, ties to the smaller index."""
    a_nan = a != a
    b_nan = b != b
    prefer_a = (a_nan & ~b_nan) | (a_nan & b_nan & (a_index < b_index))
    prefer_a = prefer_a | (~a_nan & ~b_nan & ((a > b) | ((a == b) & (a_index < b_index))))
    return tl.where(prefer_a, a, b), tl.where(prefer_a, a_index, b_index)


@triton.jit
def _gumbel_scores(x, vocab_ids, row, temp_ptr, seed_ptr, offset_ptr, greedy_threshold):
    """``x / T + G`` for a sampled row, ``G`` keyed by the row's seed and RNG offset and by each
    global vocabulary index; a greedy row keeps ``x``."""
    temperature = tl.load(temp_ptr + row)
    if temperature > greedy_threshold:
        offset = tl.load(offset_ptr + row).to(tl.uint64)
        zero = tl.zeros(vocab_ids.shape, dtype=tl.uint32)
        bits, _, _, _ = tl.philox(
            tl.load(seed_ptr + row),
            vocab_ids.to(tl.uint32),
            zero + (offset & 0xFFFFFFFF).to(tl.uint32),
            zero + (offset >> 32).to(tl.uint32),
            zero,
        )
        # u = (2k + 1) / 2**24 for 23 random bits k lies strictly inside (0, 1) and u - 1 is
        # exact, so -log(u) = -log1p(u - 1) keeps its precision where u nears 1 (the winners).
        u_minus_1 = (2 * (bits >> 9).to(tl.int32) + 1 - (1 << 24)).to(tl.float32) * (
            1.0 / (1 << 24)
        )
        gumbel = -libdevice.log(-libdevice.log1p(u_minus_1))
        # Correctly rounded: a shard scores each entry as the full vocabulary does.
        scores = libdevice.add_rn(libdevice.div_rn(x, temperature), gumbel)
    else:
        scores = x
    return scores


# One binary for every rank of the group.
@triton.jit(do_not_specialize=["rank", "vocab_offset", "local_vocab", "num_chunks"])
def _sharded_argmax_kernel(
    logits_ptr,
    stride_row,
    temp_ptr,
    seed_ptr,
    offset_ptr,
    part_val_ptr,
    part_idx_ptr,
    state_ptr,
    peers_ptr,
    out_ptr,
    rows,
    local_vocab,
    vocab_offset,
    num_chunks,
    rank,
    greedy_threshold,
    NRANKS: tl.constexpr,
    MAX_ROWS: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    CHUNKS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    NO_INDEX: tl.constexpr,
    TEMPERATURE: tl.constexpr,
):
    """``state_ptr``: int32 (arrival ticket, launch tag); ``peers_ptr``: every rank's exchange
    buffer (3 slots of ``NRANKS x MAX_ROWS`` int64 entries). ``TEMPERATURE``: rows above
    ``greedy_threshold`` are scored ``logits / T + G``."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    in_vocab = cols < local_vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * stride_row + cols, mask=in_vocab, other=float("-inf")
    ).to(tl.float32)
    if TEMPERATURE:
        x = _gumbel_scores(
            x, cols + vocab_offset, row, temp_ptr, seed_ptr, offset_ptr, greedy_threshold
        )
    best, idx = tl.reduce(
        (tl.where(in_vocab, x, float("-inf")), tl.where(in_vocab, cols + vocab_offset, NO_INDEX)),
        0,
        _prefer_argmax,
    )
    tl.store(part_val_ptr + row * num_chunks + chunk, best)
    tl.store(part_idx_ptr + row * num_chunks + chunk, idx)
    # The last program to arrive sees every partial (acquire) and runs the exchange.
    ticket = tl.atomic_add(state_ptr, 1, sem="acq_rel", scope="gpu")
    if ticket == rows * num_chunks - 1:
        r = tl.arange(0, ROWS_PAD)
        live = r < rows
        c = tl.arange(0, CHUNKS_PAD)
        part_mask = live[:, None] & (c[None, :] < num_chunks)
        part_off = r[:, None] * num_chunks + c[None, :]
        pv = tl.load(part_val_ptr + part_off, mask=part_mask, other=float("-inf"), volatile=True)
        pi = tl.load(part_idx_ptr + part_off, mask=part_mask, other=NO_INDEX, volatile=True)
        cand_val, cand_idx = tl.reduce((pv, pi), 1, _prefer_argmax)
        index_field = (
            tl.where(cand_idx == NO_INDEX, _INDEX_MASK, cand_idx).to(tl.uint32).to(tl.uint64)
        )
        value_bits = cand_val.to(tl.uint32, bitcast=True).to(tl.uint64)

        tag = tl.load(state_ptr + 1)
        tag_bits = tag.to(tl.uint64)
        entry = (((tag_bits << _INDEX_BITS) | index_field) << 32 | value_bits).to(
            tl.int64, bitcast=True
        )
        slot = (tag % 3) * (NRANKS * MAX_ROWS)
        # One store per (destination rank, row): [NRANKS, ROWS_PAD] holds an element per thread.
        src = tl.arange(0, NRANKS)
        entry_mask = (src[:, None] < NRANKS) & live[None, :]
        dsts = tl.load(peers_ptr + src).to(tl.uint64).to(tl.pointer_type(tl.int64))
        tl.atomic_xchg(
            dsts[:, None] + (slot + rank * MAX_ROWS + r)[None, :],
            tl.broadcast_to(entry[None, :], [NRANKS, ROWS_PAD]),
            mask=entry_mask,
            sem="relaxed",
            scope="sys",
        )
        mine = tl.load(peers_ptr + rank).to(tl.uint64).to(tl.pointer_type(tl.int64))
        previous = ((tag + 2) % 3) * (NRANKS * MAX_ROWS)
        words = tl.arange(0, NRANKS * MAX_ROWS)
        tl.store(mine + previous + words, tl.zeros([NRANKS * MAX_ROWS], dtype=tl.int64))

        entry_off = slot + src[:, None] * MAX_ROWS + r[None, :]
        want = tag.to(tl.int64)
        got = tl.load(mine + entry_off, mask=entry_mask, other=0, volatile=True)
        pending = ((got >> (32 + _INDEX_BITS)) & _TAG_MOD) != want
        waiting = tl.sum((pending & entry_mask).to(tl.int32))
        while waiting > 0:
            got = tl.load(mine + entry_off, mask=entry_mask, other=0, volatile=True)
            pending = ((got >> (32 + _INDEX_BITS)) & _TAG_MOD) != want
            waiting = tl.sum((pending & entry_mask).to(tl.int32))
        # Every awaited entry is final: one more load, so all copies of an entry reduce the
        # same bits.
        tl.debug_barrier()
        got = tl.load(mine + entry_off, mask=entry_mask, other=0, volatile=True)
        vals = tl.where(entry_mask, got.to(tl.int32).to(tl.float32, bitcast=True), float("-inf"))
        ids = tl.where(entry_mask, ((got >> 32) & _INDEX_MASK).to(tl.int32), _INDEX_MASK)
        _, token = tl.reduce((vals, ids), 0, _prefer_argmax)
        tl.store(out_ptr + r, token.to(out_ptr.dtype.element_ty), mask=live)
        tl.store(state_ptr + 1, tag % _TAG_MOD + 1)
        tl.store(state_ptr, 0)


class VocabShardedSampler:
    """Greedy and temperature-only target tokens from a column-parallel LM head's logit shards.

    Owns the exchange buffers of one tensor-parallel group; every rank of the group calls
    :meth:`sample` the same number of times in the same order.
    """

    def __init__(
        self,
        rank: int,
        nranks: int,
        peer_ptrs: Sequence[int],
        local_vocab: int,
        vocab_size: int,
        keepalive: object = None,
    ):
        if nranks & (nranks - 1) or len(peer_ptrs) != nranks or not 0 <= rank < nranks:
            raise ValueError(f"unsupported exchange group: rank {rank} of {nranks}")
        if vocab_size > _MAX_VOCAB:
            raise ValueError(f"vocab_size={vocab_size} exceeds the exchange index field")
        self.rank = rank
        self.nranks = nranks
        self.local_vocab = local_vocab
        self.vocab_offset = rank * local_vocab
        # The last shard of a padded head ends at the real vocabulary.
        self.valid_vocab = min(local_vocab, vocab_size - self.vocab_offset)
        if self.valid_vocab <= 0:
            raise ValueError(f"rank {rank} holds no vocabulary entries")
        self.num_chunks = triton.cdiv(self.valid_vocab, _BLOCK)
        # Sized by the shard width, not its valid part, so every rank runs the same binary.
        self._chunks_pad = triton.next_power_of_2(triton.cdiv(local_vocab, _BLOCK))
        device = torch.device("cuda", torch.cuda.current_device())
        self._peers = torch.tensor(list(peer_ptrs), dtype=torch.int64, device=device)
        self._state = torch.tensor([0, 1], dtype=torch.int32, device=device)
        self._part_val = torch.empty(MAX_ROWS * self.num_chunks, dtype=torch.float32, device=device)
        self._part_idx = torch.empty(MAX_ROWS * self.num_chunks, dtype=torch.int32, device=device)
        # The peer allocation and its mappings live as long as the sampler.
        self._keepalive = keepalive

    @staticmethod
    def buffer_bytes(nranks: int) -> int:
        """Bytes of one rank's exchange buffer."""
        return _NUM_SLOTS * nranks * MAX_ROWS * 8

    @classmethod
    def create(
        cls, mapping: Mapping, local_vocab: int, vocab_size: int
    ) -> Optional["VocabShardedSampler"]:
        """The sampler of ``mapping``'s TP group, or None where the exchange is unsupported.

        Collective over the TP group: every rank calls it at the same point.
        """
        from ..distributed.peer_buffers import PeerBuffers

        if (
            mapping.tp_size not in (2, 4, 8)
            or mapping.enable_attention_dp
            or mapping.has_pp()
            or mapping.has_cp()
            or vocab_size > _MAX_VOCAB
        ):
            return None
        buffers = PeerBuffers.create(mapping, cls.buffer_bytes(mapping.tp_size))
        if buffers is None:
            return None
        return cls(
            mapping.tp_rank,
            mapping.tp_size,
            buffers.peer_ptrs,
            local_vocab,
            vocab_size,
            keepalive=buffers,
        )

    def sample(
        self,
        logits: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
        seeds: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Int64 tokens of every row of the gathered logits, from this rank's ``[rows, >=
        valid_vocab]`` shard; every rank of the group calls it with the same ``rows``.

        Rows take the argmax, or with per-row FP32 ``temperatures`` and int64 Philox ``seeds`` /
        ``offsets``, rows above the greedy threshold draw from ``softmax(logits / T)``.
        """
        rows = logits.shape[0]
        if not 0 < rows <= MAX_ROWS or logits.shape[1] < self.valid_vocab or logits.stride(1) != 1:
            raise ValueError(f"unsupported shard logits {tuple(logits.shape)}")
        sampled = temperatures is not None
        if sampled and (seeds is None or offsets is None):
            raise ValueError("temperature sampling needs each row's seed and offset")
        out = torch.empty((rows,), dtype=torch.int64, device=logits.device)
        _sharded_argmax_kernel[(rows, self.num_chunks)](
            logits,
            logits.stride(0),
            # Unread without temperatures.
            temperatures if sampled else self._state,
            seeds if sampled else self._state,
            offsets if sampled else self._state,
            self._part_val,
            self._part_idx,
            self._state,
            self._peers,
            out,
            rows,
            self.valid_vocab,
            self.vocab_offset,
            self.num_chunks,
            self.rank,
            GREEDY_TEMPERATURE_THRESHOLD,
            NRANKS=self.nranks,
            MAX_ROWS=MAX_ROWS,
            # At least one exchange entry per thread: no thread holds a copy of another's.
            ROWS_PAD=max(triton.next_power_of_2(rows), 32 * _NUM_WARPS // self.nranks),
            CHUNKS_PAD=self._chunks_pad,
            BLOCK=_BLOCK,
            NO_INDEX=_NO_INDEX,
            TEMPERATURE=sampled,
            num_warps=_NUM_WARPS,
        )
        return out


__all__ = ["VocabShardedSampler"]
