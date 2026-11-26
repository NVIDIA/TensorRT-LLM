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
"""Target verify tokens of attention-DP ranks from an LM head sharded over a node-local group of G ranks.

Attention DP replicates the LM head, so for a few hundred verify rows every rank streams the whole BF16
vocabulary. Here each member of a group multiplies the rows of all G members by its 1/G of the vocabulary
(the same logit bits as the replicated head) and the members sample from their slices, exchanging a few words
per row instead of logits:

- ``gather``: every rank copies its rows and their sampling parameters into every member's buffer (SM stores to
  peer memory), raises its counter at each member and waits for all members' counters.
- ``sample``: per row and slice chunk, the argmax (greedy rows) or the max and the sum of ``exp(x / T - max)``
  (temperature rows); the last chunk of a row sends the slice's entry to every member. Every member then reads
  all slices' entries of every row. A greedy row's owner takes the argmax over slices (``torch.argmax`` order,
  the replicated argmax). A temperature row draws ``u`` from the row's Philox seed and offset; every member
  picks the same slice by its share of the softmax mass, and that slice finds the token by inverse CDF over its
  chunk sums and logits, and stores it at the row's owner. That is an exact draw from ``softmax(logits / T)``.

Entries carry the call count as a tag in their upper 32 bits, so no buffer is cleared. Buffers are reused from
call to call: a rank releases its earlier calls when it starts the next one (its kernels for them have finished
by then), and a call's pushes wait until every member has released the previous call.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.mapping import Mapping

from ..distributed.peer_buffers import PeerBuffers, PeerTransport
from ..pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
from .sharded_sampling import prefer_argmax

# Logit columns per chunk of a row's slice, and int32 words per program of the row push.
_BLOCK = 4096
_PUSH_WORDS = 4096
_NO_INDEX = 2147483647
# Keys of a chunk's argmax: column of a NaN, column + _FIRST_MAX of a maximum.
_FIRST_MAX = tl.constexpr(1 << 29)


@triton.jit
def _row_uniform(seed, offset):
    """``(2k + 1) / 2**24`` for 23 bits ``k`` of Philox keyed by ``seed`` at the row's 64-bit ``offset``:
    uniform, exact in FP32 and strictly inside (0, 1)."""
    lo = (offset & 0xFFFFFFFF).to(tl.uint32)
    hi = ((offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    bits, _, _, _ = tl.philox(seed, lo, hi, lo * 0, lo * 0)
    return ((bits >> 9).to(tl.float32) * 2.0 + 1.0) * (1.0 / 16777216.0)


@triton.jit
def _gather_kernel(
    rows_ptr,
    temps_ptr,
    seeds_ptr,
    offsets_ptr,
    peers_ptr,
    ticket_ptr,
    calls_ptr,
    consumed_ptr,
    rows_offset,
    params_offset,
    flags_offset,
    consumed_offset,
    row_words,
    count,
    rank,
    NRANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS_PAD: tl.constexpr,
    SAMPLED: tl.constexpr,
):
    """Once every other member finished sampling the previous call, copy this rank's rows (int32 words) and
    their sampling parameters into every member's buffers; the last program counts the call and raises this
    rank's arrival counter at every member. This rank's earlier calls are done (stream order), so program 0
    first releases them at every member, and no program waits for that release."""
    pid = tl.program_id(0)
    member = tl.arange(0, NRANKS)
    others = member != rank
    done = tl.load(calls_ptr)
    bases = tl.load(peers_ptr + member).to(tl.uint64)
    if pid == 0:
        released = (bases + consumed_offset).to(tl.pointer_type(tl.int64)) + rank
        tl.atomic_max(released, tl.zeros([NRANKS], tl.int64) + done, sem="relaxed", scope="sys")
    got = tl.atomic_add(
        consumed_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
    )
    pending = tl.sum(((got < done) & others).to(tl.int32))
    while pending > 0:
        got = tl.atomic_add(
            consumed_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
        )
        pending = tl.sum(((got < done) & others).to(tl.int32))
    words = pid * BLOCK + tl.arange(0, BLOCK)
    total = count * row_words
    live = words < total
    data = tl.load(rows_ptr + words, mask=live)
    dst = (bases + rows_offset).to(tl.pointer_type(tl.int32))[:, None] + (rank * total + words)[
        None, :
    ]
    tl.store(dst, tl.broadcast_to(data[None, :], [NRANKS, BLOCK]), mask=live[None, :])
    if pid == 0:
        r = tl.arange(0, ROWS_PAD)
        own = r < count
        if SAMPLED:
            temps = tl.load(temps_ptr + r, mask=own, other=0.0)
            seeds = tl.load(seeds_ptr + r, mask=own, other=0)
            offsets = tl.load(offsets_ptr + r, mask=own, other=0)
        else:
            temps = tl.zeros([ROWS_PAD], tl.float32)
            seeds = tl.zeros([ROWS_PAD], tl.int64)
            offsets = tl.zeros([ROWS_PAD], tl.int64)
        temp_bits = temps.to(tl.int32, bitcast=True).to(tl.int64)
        param_base = (bases + params_offset).to(tl.pointer_type(tl.int64))[:, None]
        slot = (rank * count + r) * 3
        mask = own[None, :]
        tl.store(
            param_base + slot[None, :],
            tl.broadcast_to(temp_bits[None, :], [NRANKS, ROWS_PAD]),
            mask=mask,
        )
        tl.store(
            param_base + (slot + 1)[None, :],
            tl.broadcast_to(seeds[None, :], [NRANKS, ROWS_PAD]),
            mask=mask,
        )
        tl.store(
            param_base + (slot + 2)[None, :],
            tl.broadcast_to(offsets[None, :], [NRANKS, ROWS_PAD]),
            mask=mask,
        )
    # Every thread's stores before the program's arrival; the last arrival publishes all programs' stores.
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="sys")
    if ticket == tl.num_programs(0) - 1:
        tl.store(ticket_ptr, 0)
        tl.store(calls_ptr, done + 1)
        flags = (bases + flags_offset).to(tl.pointer_type(tl.int64)) + rank
        tl.atomic_add(flags, tl.full([NRANKS], 1, tl.int64), sem="release", scope="sys")


@triton.jit
def _wait_kernel(flags_ptr, calls_ptr, NRANKS: tl.constexpr):
    """Spin until every member's counter reached this rank's call count."""
    want = tl.load(calls_ptr)
    member = tl.arange(0, NRANKS)
    got = tl.atomic_add(
        flags_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
    )
    pending = tl.sum((got < want).to(tl.int32))
    while pending > 0:
        got = tl.atomic_add(
            flags_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
        )
        pending = tl.sum((got < want).to(tl.int32))


@triton.jit
def _slice_stats_kernel(
    logits_ptr,
    stride_row,
    params_ptr,
    part_a_ptr,
    part_b_ptr,
    tickets_ptr,
    calls_ptr,
    peers_ptr,
    stats_offset,
    max_total_rows,
    local_vocab,
    vocab_offset,
    num_chunks,
    rank,
    greedy_threshold,
    NRANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS_PAD: tl.constexpr,
    NO_INDEX: tl.constexpr,
):
    """Per (row, chunk): the chunk's argmax (greedy row) or max and sum of ``exp(x / T - max)`` (temperature row).
    The last chunk of a row reduces the row's slice entry and stores it, tagged, at every member."""
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    temperature = tl.load(params_ptr + row * 3).to(tl.int32).to(tl.float32, bitcast=True)
    sampled = temperature > greedy_threshold
    cols = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = cols < local_vocab
    x = tl.load(
        logits_ptr + row.to(tl.int64) * stride_row + cols, mask=valid, other=float("-inf")
    ).to(tl.float32)
    if sampled:
        s = tl.where(valid, x * (1.0 / temperature), float("-inf"))
        m = tl.max(s, 0)
        z = tl.sum(tl.where(valid, tl.exp(s - m), 0.0), 0)
        tl.store(part_a_ptr + row * num_chunks + chunk, m)
        tl.store(part_b_ptr + row * num_chunks + chunk, z.to(tl.int32, bitcast=True))
    else:
        # torch.argmax order: the first NaN, else the first maximum (tl.max skips NaN).
        top = tl.max(x, 0)
        key = tl.where(
            x != x, cols, tl.where(valid & (x == top), cols + _FIRST_MAX, 2 * _FIRST_MAX)
        )
        first = tl.min(key, 0)
        best = tl.where(first < _FIRST_MAX, float("nan"), top)
        tl.store(part_a_ptr + row * num_chunks + chunk, best)
        tl.store(part_b_ptr + row * num_chunks + chunk, (first % _FIRST_MAX) + vocab_offset)
    tl.debug_barrier()
    ticket = tl.atomic_add(tickets_ptr + row, 1, sem="acq_rel", scope="gpu")
    if ticket == num_chunks - 1:
        tl.store(tickets_ptr + row, 0)
        c = tl.arange(0, CHUNKS_PAD)
        live = c < num_chunks
        pa = tl.load(
            part_a_ptr + row * num_chunks + c, mask=live, other=float("-inf"), volatile=True
        )
        pb = tl.load(part_b_ptr + row * num_chunks + c, mask=live, other=0, volatile=True)
        if sampled:
            top = tl.max(pa, 0)
            mass = tl.sum(
                tl.where(live, pb.to(tl.float32, bitcast=True) * tl.exp(pa - top), 0.0), 0
            )
            word0 = top.to(tl.int32, bitcast=True)
            word1 = mass.to(tl.int32, bitcast=True)
        else:
            best, idx = tl.reduce((pa, tl.where(live, pb, NO_INDEX)), 0, prefer_argmax)
            word0 = best.to(tl.int32, bitcast=True)
            word1 = idx
        tag = (tl.load(calls_ptr) & 0x7FFFFFFF) << 32
        entry0 = tag | (word0.to(tl.int64) & 0xFFFFFFFF)
        entry1 = tag | (word1.to(tl.int64) & 0xFFFFFFFF)
        member = tl.arange(0, NRANKS)
        bases = (tl.load(peers_ptr + member).to(tl.uint64) + stats_offset).to(
            tl.pointer_type(tl.int64)
        )
        offset = (rank * max_total_rows + row) * 2
        tl.atomic_xchg(
            bases + offset, tl.broadcast_to(entry0, [NRANKS]), sem="relaxed", scope="sys"
        )
        tl.atomic_xchg(
            bases + offset + 1, tl.broadcast_to(entry1, [NRANKS]), sem="relaxed", scope="sys"
        )


@triton.jit
def _select_kernel(
    logits_ptr,
    stride_row,
    params_ptr,
    part_a_ptr,
    part_b_ptr,
    stats_ptr,
    peers_ptr,
    tokens_offset,
    out_ptr,
    calls_ptr,
    count,
    max_total_rows,
    local_vocab,
    vocab_offset,
    num_chunks,
    rank,
    greedy_threshold,
    NRANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNKS_PAD: tl.constexpr,
):
    """Per row of the group: wait for every slice's entry, then the owner takes a greedy row's argmax; for a
    temperature row every member picks the same slice and that slice stores the drawn token at the owner. It
    waits only for entries, which no kernel waits to send, so no member blocks another."""
    row = tl.program_id(0)
    owner = row // count
    own_row = row % count
    call = tl.load(calls_ptr) & 0x7FFFFFFF
    shard = tl.arange(0, NRANKS)
    words = stats_ptr + (shard * max_total_rows + row) * 2
    w0 = tl.load(words, volatile=True)
    w1 = tl.load(words + 1, volatile=True)
    pending = tl.sum((((w0 >> 32) != call) | ((w1 >> 32) != call)).to(tl.int32))
    while pending > 0:
        w0 = tl.load(words, volatile=True)
        w1 = tl.load(words + 1, volatile=True)
        pending = tl.sum((((w0 >> 32) != call) | ((w1 >> 32) != call)).to(tl.int32))
    temperature = tl.load(params_ptr + row * 3).to(tl.int32).to(tl.float32, bitcast=True)
    lo0 = (w0 & 0xFFFFFFFF).to(tl.int32)
    lo1 = (w1 & 0xFFFFFFFF).to(tl.int32)
    if temperature > greedy_threshold:
        tops = lo0.to(tl.float32, bitcast=True)
        masses = lo1.to(tl.float32, bitcast=True)
        top = tl.max(tops, 0)
        weights = masses * tl.exp(tops - top)
        total = tl.sum(weights, 0)
        u = _row_uniform(tl.load(params_ptr + row * 3 + 1), tl.load(params_ptr + row * 3 + 2))
        target = u * total
        cum = tl.cumsum(weights, 0)
        # The first slice whose cumulative mass passes the target (the last positive one at the rounding edge).
        passed = (cum > target) & (weights > 0)
        last_positive = tl.max(tl.where(weights > 0, shard, -1), 0)
        chosen = tl.min(tl.where(passed, shard, NRANKS), 0)
        chosen = tl.where(chosen == NRANKS, last_positive, chosen)
        # Without positive mass (NaN or infinite logits) slice 0 draws, so the owner still gets a token.
        chosen = tl.maximum(chosen, 0)
        before = tl.sum(tl.where(shard < chosen, weights, 0.0), 0)
        chosen_top = tl.sum(tl.where(shard == chosen, tops, 0.0), 0)
        if chosen == rank:
            # Mass inside this slice, in units of exp(x / T - chosen_top).
            residual = (target - before) * tl.exp(top - chosen_top)
            c = tl.arange(0, CHUNKS_PAD)
            live = c < num_chunks
            pa = tl.load(part_a_ptr + row * num_chunks + c, mask=live, other=float("-inf"))
            pb = tl.load(part_b_ptr + row * num_chunks + c, mask=live, other=0).to(
                tl.float32, bitcast=True
            )
            chunk_mass = tl.where(live, pb * tl.exp(pa - chosen_top), 0.0)
            chunk_cum = tl.cumsum(chunk_mass, 0)
            chunk_pass = (chunk_cum > residual) & (chunk_mass > 0)
            last_chunk = tl.max(tl.where(chunk_mass > 0, c, -1), 0)
            pick = tl.min(tl.where(chunk_pass, c, CHUNKS_PAD), 0)
            pick = tl.maximum(tl.where(pick == CHUNKS_PAD, last_chunk, pick), 0)
            inside = residual - tl.sum(tl.where(c < pick, chunk_mass, 0.0), 0)
            cols = pick * BLOCK + tl.arange(0, BLOCK)
            valid = cols < local_vocab
            x = tl.load(
                logits_ptr + row.to(tl.int64) * stride_row + cols, mask=valid, other=float("-inf")
            )
            e = tl.where(valid, tl.exp(x.to(tl.float32) * (1.0 / temperature) - chosen_top), 0.0)
            e_cum = tl.cumsum(e, 0)
            hit = (e_cum > inside) & (e > 0)
            last_col = tl.max(tl.where(e > 0, cols, -1), 0)
            col = tl.min(tl.where(hit, cols, local_vocab), 0)
            col = tl.maximum(tl.where(col == local_vocab, last_col, col), 0)
            entry = (call.to(tl.int64) << 32) | (col + vocab_offset).to(tl.int64)
            dst = (tl.load(peers_ptr + owner).to(tl.uint64) + tokens_offset).to(
                tl.pointer_type(tl.int64)
            )
            tl.atomic_xchg(dst + own_row, entry, sem="relaxed", scope="sys")
    else:
        if owner == rank:
            values = lo0.to(tl.float32, bitcast=True)
            _, token = tl.reduce((values, lo1), 0, prefer_argmax)
            tl.store(out_ptr + own_row, token.to(tl.int64))


@triton.jit
def _collect_kernel(params_ptr, tokens_ptr, out_ptr, calls_ptr, count, rank, greedy_threshold):
    """Per own row: wait for a temperature row's token, which the member holding it stored here."""
    own_row = tl.program_id(0)
    row = rank * count + own_row
    call = tl.load(calls_ptr) & 0x7FFFFFFF
    temperature = tl.load(params_ptr + row * 3).to(tl.int32).to(tl.float32, bitcast=True)
    if temperature > greedy_threshold:
        got = tl.load(tokens_ptr + own_row, volatile=True)
        while (got >> 32) != call:
            got = tl.load(tokens_ptr + own_row, volatile=True)
        tl.store(out_ptr + own_row, got & 0xFFFFFFFF)


class GroupShardedVerify:
    """The sharded verify head and sampler of one node-local group of attention-DP ranks.

    ``logits(rows, ...)`` gathers the group's rows and returns this rank's ``[G * R, vocab / G]`` logit slice;
    ``sample(slice)`` returns this rank's ``R`` tokens. Every member makes the same calls in the same order with
    the same ``R``.
    """

    def __init__(self, group: Mapping, weight: torch.Tensor, max_rows: int, buffers: PeerBuffers):
        self.group_size = group.tp_size
        self.rank = group.tp_rank
        self.first_dp_rank = group.rank - group.tp_rank
        self.vocab, self.hidden = weight.shape
        self.max_rows = max_rows
        self.local_vocab = self.vocab // self.group_size
        self.vocab_offset = self.rank * self.local_vocab
        self.weight = weight[self.vocab_offset : self.vocab_offset + self.local_vocab]
        self.num_chunks = triton.cdiv(self.local_vocab, _BLOCK)
        self.chunks_pad = triton.next_power_of_2(self.num_chunks)
        g, total = self.group_size, self.group_size * max_rows
        self.layout = layout = self.buffer_layout(g, max_rows, self.hidden)
        self.buffers = buffers
        local = buffers.local_ptr
        self.rows = _wrap(local + layout["rows"], total, self.hidden * 2, torch.bfloat16)
        self.params = _wrap(local + layout["params"], total, 24, torch.int64)
        self.stats = _wrap(local + layout["stats"], g * total, 16, torch.int64)
        self.tokens = _wrap(local + layout["tokens"], 1, max_rows * 8, torch.int64)[0]
        self.arrival = _wrap(local + layout["arrival"], 1, g * 8, torch.int64)[0]
        self.consumed = _wrap(local + layout["consumed"], 1, g * 8, torch.int64)[0]
        self.peers = torch.tensor(buffers.peer_ptrs, dtype=torch.int64, device="cuda")
        self.calls = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.push_ticket = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.row_tickets = torch.zeros(total, dtype=torch.int32, device="cuda")
        self.part_a = torch.empty(total * self.num_chunks, dtype=torch.float32, device="cuda")
        self.part_b = torch.empty(total * self.num_chunks, dtype=torch.int32, device="cuda")
        self.slice_logits = torch.empty(
            (total, self.local_vocab), dtype=torch.bfloat16, device="cuda"
        )
        self._count = 0
        self._sampled = False

    @staticmethod
    def buffer_layout(group_size: int, max_rows: int, hidden: int) -> dict:
        """Byte offsets of the peer buffer's regions; ``end`` is its size."""
        total = group_size * max_rows
        layout = {"rows": 0}
        layout["params"] = layout["rows"] + total * hidden * 2
        layout["stats"] = layout["params"] + total * 3 * 8
        layout["tokens"] = layout["stats"] + group_size * total * 2 * 8
        layout["arrival"] = layout["tokens"] + max_rows * 8
        layout["consumed"] = layout["arrival"] + group_size * 8
        layout["end"] = layout["consumed"] + group_size * 8
        return layout

    @classmethod
    def create(
        cls, mapping: Mapping, weight: torch.Tensor, group_size: int, max_rows: int
    ) -> Optional["GroupShardedVerify"]:
        """The sampler of this rank's group of ``group_size`` consecutive attention-DP ranks, or None where the group is
        unsupported (the same answer on every rank). Collective over the attention-DP group."""
        if (
            group_size not in (2, 4, 8)
            or mapping.tp_size % group_size
            or mapping.gpus_per_node % group_size
            or mapping.world_size != mapping.tp_size
            or weight.shape[0] % group_size
            or weight.shape[0] >= _NO_INDEX
            or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()
        ):
            return None
        # Consecutive ranks form a group: the tensor-parallel group of this mapping.
        group = Mapping(
            world_size=mapping.world_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            tp_size=group_size,
            pp_size=mapping.world_size // group_size,
        )
        nbytes = cls.buffer_layout(group_size, max_rows, weight.shape[1])["end"]
        buffers = PeerBuffers.create(group, nbytes, transport=PeerTransport.IPC)
        if buffers is None:
            return None
        return cls(group, weight, max_rows, buffers)

    def supports(self, rows: int, all_rank_rows: Optional[List[int]]) -> bool:
        """Whether every member of this rank's group runs ``rows`` rows (``all_rank_rows``: every attention-DP
        rank's count, which every rank sees alike)."""
        if not 0 < rows <= self.max_rows or all_rank_rows is None:
            return False
        members = all_rank_rows[self.first_dp_rank : self.first_dp_rank + self.group_size]
        return all(count == rows for count in members)

    def logits(
        self,
        rows: torch.Tensor,
        temperatures: Optional[torch.Tensor] = None,
        seeds: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This rank's vocabulary slice of the group's logits, ``[G * R, vocab / G]`` in member order. With
        ``temperatures`` (and the rows' Philox ``seeds`` / ``offsets``) rows above the greedy threshold are drawn by
        :meth:`sample`; without them every row is greedy."""
        count = rows.shape[0]
        rows = rows.contiguous()
        self._count = count
        g = self.group_size
        row_words = self.hidden // 2
        sampled = temperatures is not None
        self._sampled = sampled
        dummy = self.calls
        _gather_kernel[(triton.cdiv(count * row_words, _PUSH_WORDS),)](
            rows.view(torch.int32),
            temperatures if sampled else dummy,
            seeds if sampled else dummy,
            offsets if sampled else dummy,
            self.peers,
            self.push_ticket,
            self.calls,
            self.consumed,
            self.layout["rows"],
            self.layout["params"],
            self.layout["arrival"],
            self.layout["consumed"],
            row_words,
            count,
            self.rank,
            NRANKS=g,
            BLOCK=_PUSH_WORDS,
            ROWS_PAD=triton.next_power_of_2(self.max_rows),
            SAMPLED=sampled,
        )
        _wait_kernel[(1,)](self.arrival, self.calls, NRANKS=g)
        out = self.slice_logits[: g * count]
        torch.mm(self.rows[: g * count], self.weight.t(), out=out)
        return out

    def sample(self, slice_logits: torch.Tensor) -> torch.Tensor:
        """This rank's ``R`` tokens from every member's slice of the call's logits; the rows' sampling
        parameters came with :meth:`logits`."""
        count = self._count
        g = self.group_size
        total = g * count
        params = self.params[:total]
        _slice_stats_kernel[(total, self.num_chunks)](
            slice_logits,
            slice_logits.stride(0),
            params,
            self.part_a,
            self.part_b,
            self.row_tickets,
            self.calls,
            self.peers,
            self.layout["stats"],
            g * self.max_rows,
            self.local_vocab,
            self.vocab_offset,
            self.num_chunks,
            self.rank,
            GREEDY_TEMPERATURE_THRESHOLD,
            NRANKS=g,
            BLOCK=_BLOCK,
            CHUNKS_PAD=self.chunks_pad,
            NO_INDEX=_NO_INDEX,
        )
        out = torch.empty((count,), dtype=torch.int64, device=slice_logits.device)
        _select_kernel[(total,)](
            slice_logits,
            slice_logits.stride(0),
            params,
            self.part_a,
            self.part_b,
            self.stats,
            self.peers,
            self.layout["tokens"],
            out,
            self.calls,
            count,
            g * self.max_rows,
            self.local_vocab,
            self.vocab_offset,
            self.num_chunks,
            self.rank,
            GREEDY_TEMPERATURE_THRESHOLD,
            NRANKS=g,
            BLOCK=_BLOCK,
            CHUNKS_PAD=self.chunks_pad,
        )
        if self._sampled:
            _collect_kernel[(count,)](
                params, self.tokens, out, self.calls, count, self.rank, GREEDY_TEMPERATURE_THRESHOLD
            )
        return out


class _DeviceArray:
    """A ``__cuda_array_interface__`` over raw device memory; torch keeps it alive with every tensor of it."""

    def __init__(self, ptr: int, shape: tuple, typestr: str):
        self.__cuda_array_interface__ = {
            "shape": shape,
            "typestr": typestr,
            "data": (ptr, False),
            "version": 3,
        }


def _wrap(ptr: int, rows: int, row_bytes: int, dtype: torch.dtype) -> torch.Tensor:
    """A ``[rows, row_bytes // itemsize]`` tensor over raw device memory (bfloat16 through int16)."""
    typestr = "<i8" if dtype == torch.int64 else "<i2"
    width = row_bytes // (8 if dtype == torch.int64 else 2)
    return torch.as_tensor(_DeviceArray(ptr, (rows, width), typestr), device="cuda").view(dtype)


__all__ = ["GroupShardedVerify"]
