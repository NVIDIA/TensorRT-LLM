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
"""DSpark draft head split by vocabulary over the attention-DP group sharing the target head. The
MXFP8 GEMM sums K in one pass and candidates reduce in ``torch.argmax`` order: replicated tokens."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from tensorrt_llm.mapping import Mapping

from ..distributed.peer_buffers import PeerBuffers, PeerTransport
from .group_sharded_sampling import _FIRST_MAX, _wait_kernel, _wrap
from .sharded_sampling import prefer_argmax

# Block positions per call (one candidate slot each).
_MAX_BLOCK = 8
# int64 words of rows per push program: many programs spread the peer stores over many SMs.
_PUSH_WORDS = 1024
# Vocabulary rows per slice unit: the swizzled MXFP8 scale layout tiles 128 rows.
_ROW_TILE = 128
# Columns reduced to one candidate (tp_argmax.WORDS_PER_PART).
_CANDIDATE_COLUMNS = 4096


@triton.jit
def _push_rows_kernel(
    rows_ptr,
    bonus_ptr,
    peers_ptr,
    ticket_ptr,
    calls_ptr,
    consumed_ptr,
    rows_dst,
    bonus_offset,
    flags_offset,
    consumed_offset,
    total_words,
    count,
    rank,
    NRANKS: tl.constexpr,
    BLOCK: tl.constexpr,
    COUNT_PAD: tl.constexpr,
):
    """After every member released the previous call (no overwrite mid-read), push this rank's rows
    and bonus tokens to every member; program 0 first releases this rank's finished calls."""
    pid = tl.program_id(0)
    member = tl.arange(0, NRANKS)
    done = tl.load(calls_ptr)
    bases = tl.load(peers_ptr + member).to(tl.uint64)
    if pid == 0:
        released = (bases + consumed_offset).to(tl.pointer_type(tl.int64)) + rank
        tl.atomic_max(released, tl.zeros([NRANKS], tl.int64) + done, sem="relaxed", scope="sys")
    got = tl.atomic_add(
        consumed_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
    )
    pending = tl.sum((got < done).to(tl.int32))
    while pending > 0:
        got = tl.atomic_add(
            consumed_ptr + member, tl.zeros([NRANKS], tl.int64), sem="acquire", scope="sys"
        )
        pending = tl.sum((got < done).to(tl.int32))
    words = pid * BLOCK + tl.arange(0, BLOCK)
    live = words < total_words
    data = tl.load(rows_ptr + words, mask=live)
    for m in tl.static_range(NRANKS):
        # Buffers are 16-byte aligned: the peer stores vectorize.
        base = tl.multiple_of(tl.load(peers_ptr + m), 16)
        dst = (base.to(tl.uint64) + rows_dst).to(tl.pointer_type(tl.int64))
        tl.store(dst + words, data, mask=live)
    if pid == 0:
        r = tl.arange(0, COUNT_PAD)
        own = r < count
        bonus = tl.load(bonus_ptr + r, mask=own, other=0)
        bonus_dst = (bases + bonus_offset).to(tl.pointer_type(tl.int64))[:, None] + (
            rank * count + r
        )[None, :]
        tl.store(bonus_dst, tl.broadcast_to(bonus[None, :], [NRANKS, COUNT_PAD]), mask=own[None, :])
    # Every thread's stores before the program's arrival; the last arrival publishes all programs'.
    tl.debug_barrier()
    ticket = tl.atomic_add(ticket_ptr, 1, sem="acq_rel", scope="sys")
    if ticket == tl.num_programs(0) - 1:
        tl.store(ticket_ptr, 0)
        tl.store(calls_ptr, done + 1)
        flags = (bases + flags_offset).to(tl.pointer_type(tl.int64)) + rank
        tl.atomic_add(flags, tl.full([NRANKS], 1, tl.int64), sem="release", scope="sys")


@triton.jit
def _add_rn(a, b):
    """FP32 ``a + b`` rounded to nearest, subnormals kept (the replicated chain's add.rn.f32)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 x, y, z;
            mov.b32 x, $1;
            mov.b32 y, $2;
            add.rn.f32 z, x, y;
            mov.b32 $0, z;
        }
        """,
        constraints="=r,r,r",
        args=[a.to(tl.uint32, bitcast=True), b.to(tl.uint32, bitcast=True)],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.float32, bitcast=True)


@triton.jit
def _candidates_kernel(
    base_ptr,
    base_row_stride,
    bias_ptr,
    peers_ptr,
    calls_ptr,
    slot_offset,
    rows,
    local_vocab,
    vocab_offset,
    rank,
    NRANKS: tl.constexpr,
    PARTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per (row, ``BLOCK``-column part) of this slice: the ``torch.argmax``-order (value, id) of
    ``base + bias`` in every member's slot; each 8-byte word carries the call tag (no clearing)."""
    row = tl.program_id(0)
    part = tl.program_id(1)
    cols = part * BLOCK + tl.arange(0, BLOCK)
    valid = cols < local_vocab
    base = tl.load(base_ptr + row.to(tl.int64) * base_row_stride + cols, mask=valid, other=0.0)
    bias = tl.load(bias_ptr + row.to(tl.int64) * local_vocab + cols, mask=valid, other=0.0)
    x = tl.where(valid, _add_rn(base.to(tl.float32), bias.to(tl.float32)), float("-inf"))
    # tl.max skips NaN; a NaN column's key is its column, a maximum's its column + _FIRST_MAX.
    top = tl.max(x, 0)
    key = tl.where(x != x, cols, tl.where(valid & (x == top), cols + _FIRST_MAX, 2 * _FIRST_MAX))
    first = tl.min(key, 0)
    best = tl.where(first < _FIRST_MAX, float("nan"), top)
    tag = (tl.load(calls_ptr) & 0x7FFFFFFF) << 32
    word0 = tag | (best.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF)
    word1 = tag | ((first % _FIRST_MAX) + vocab_offset).to(tl.int64)
    member = tl.arange(0, NRANKS)
    dst = (tl.load(peers_ptr + member).to(tl.uint64) + slot_offset).to(tl.pointer_type(tl.int64))
    dst += 2 * ((rank * rows + row) * PARTS + part)
    tl.store(dst, tl.zeros([NRANKS], tl.int64) + word0)
    tl.store(dst + 1, tl.zeros([NRANKS], tl.int64) + word1)


@triton.jit
def _select_kernel(
    peers_ptr,
    calls_ptr,
    slot_offset,
    tokens_ptr,
    w1_ptr,
    latent_ptr,
    rows,
    rank,
    vocab,
    NRANKS: tl.constexpr,
    PARTS: tl.constexpr,
    DIM: tl.constexpr,
    EMBED: tl.constexpr,
):
    """Per row: wait for every member's candidates of the position in this member's slot, take the
    ``torch.argmax``-order winner and (``EMBED``) copy its Markov embedding row to ``latent``."""
    row = tl.program_id(0)
    call = tl.load(calls_ptr) & 0x7FFFFFFF
    member = tl.arange(0, NRANKS)
    part = tl.arange(0, PARTS)
    own = (tl.load(peers_ptr + rank).to(tl.uint64) + slot_offset).to(tl.pointer_type(tl.int64))
    entries = own + 2 * ((member[:, None] * rows + row) * PARTS + part[None, :])
    word0 = tl.load(entries, volatile=True)
    word1 = tl.load(entries + 1, volatile=True)
    stale = ((word0 >> 32) != call) | ((word1 >> 32) != call)
    pending = tl.sum(tl.sum(stale.to(tl.int32), 1), 0)
    while pending > 0:
        word0 = tl.load(entries, volatile=True)
        word1 = tl.load(entries + 1, volatile=True)
        stale = ((word0 >> 32) != call) | ((word1 >> 32) != call)
        pending = tl.sum(tl.sum(stale.to(tl.int32), 1), 0)
    values = (word0 & 0xFFFFFFFF).to(tl.int32).to(tl.float32, bitcast=True)
    values = tl.reshape(values, [NRANKS * PARTS])
    ids = tl.reshape((word1 & 0xFFFFFFFF).to(tl.int32), [NRANKS * PARTS])
    _, token = tl.reduce((values, ids), 0, prefer_argmax)
    tl.store(tokens_ptr + row, token.to(tl.int64))
    if EMBED:
        d = tl.arange(0, DIM)
        ok = (token >= 0) & (token < vocab)
        embedding = tl.load(w1_ptr + token.to(tl.int64) * DIM + d, mask=ok & (d < DIM), other=0.0)
        tl.store(latent_ptr + row * DIM + d, embedding)


class GroupShardedDraftHead:
    """The vocabulary-split DSpark draft head of one node-local group of attention-DP ranks; every
    member makes the same calls with the same rows (:meth:`supports` reads only shared values)."""

    def __init__(
        self, group: Mapping, vocab: int, hidden: int, max_rows: int, buffers: PeerBuffers
    ):
        g = group.tp_size
        self.group_size = g
        self.rank = group.tp_rank
        self.first_dp_rank = group.rank - group.tp_rank
        self.vocab, self.hidden, self.max_rows = vocab, hidden, max_rows
        self.start, self.stop = self.vocab_range(vocab, g, self.rank)
        self.parts = triton.cdiv(self.stop - self.start, _CANDIDATE_COLUMNS)
        self.layout = layout = self.buffer_layout(g, max_rows, hidden, self.parts)
        self.buffers = buffers
        local = buffers.local_ptr
        self.rows = _wrap(local + layout["rows"], g * max_rows, hidden * 2, torch.bfloat16)
        self.bonus = _wrap(local + layout["bonus"], 1, g * max_rows * 8, torch.int64)[0]
        self.arrival = _wrap(local + layout["arrival"], 1, g * 8, torch.int64)[0]
        self.consumed = _wrap(local + layout["consumed"], 1, g * 8, torch.int64)[0]
        self.peers = torch.tensor(buffers.peer_ptrs, dtype=torch.int64, device="cuda")
        self.calls = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.push_ticket = torch.zeros(1, dtype=torch.int32, device="cuda")

    @staticmethod
    def vocab_range(vocab: int, group_size: int, member: int) -> tuple[int, int]:
        """``[start, stop)`` vocabulary rows of ``member``: whole 128-row tiles, split evenly."""
        tiles = triton.cdiv(vocab, _ROW_TILE)
        start = min(vocab, _ROW_TILE * (member * tiles // group_size))
        stop = min(vocab, _ROW_TILE * ((member + 1) * tiles // group_size))
        return start, stop

    @staticmethod
    def buffer_layout(group_size: int, max_rows: int, hidden: int, parts: int) -> dict:
        """Byte offsets of the peer buffer's regions; ``slot`` is one block position's candidates of
        every member for every row, ``end`` the buffer size."""
        total = group_size * max_rows
        layout = {"rows": 0, "slot": group_size * total * parts * 16}
        layout["bonus"] = layout["rows"] + total * hidden * 2
        layout["entries"] = layout["bonus"] + total * 8
        layout["arrival"] = layout["entries"] + _MAX_BLOCK * layout["slot"]
        layout["consumed"] = layout["arrival"] + group_size * 8
        layout["end"] = layout["consumed"] + group_size * 8
        return layout

    @classmethod
    def create(
        cls, mapping: Mapping, vocab: int, hidden: int, max_rows: int
    ) -> Optional["GroupShardedDraftHead"]:
        """This rank's group head (``min(gpus_per_node, tp_size)`` consecutive attention-DP ranks),
        or None where unsupported (alike on every rank); collective over the group."""
        group_size = min(mapping.gpus_per_node, mapping.tp_size)
        if (
            not mapping.enable_attention_dp
            or not getattr(mapping, "enable_lm_head_tp_in_adp", False)
            or group_size not in (2, 4, 8)
            or mapping.tp_size % group_size
            or mapping.gpus_per_node % group_size
            or mapping.world_size != mapping.tp_size
            or vocab % 8
            or vocab >= 1 << 24
            or hidden % 64
        ):
            return None
        slices = [cls.vocab_range(vocab, group_size, m) for m in range(group_size)]
        parts = {triton.cdiv(stop - start, _CANDIDATE_COLUMNS) for start, stop in slices}
        # Every member reduces the same power-of-two number of candidates per row.
        if len(parts) != 1 or any(stop <= start for start, stop in slices):
            return None
        if triton.next_power_of_2(min(parts)) != min(parts):
            return None
        group = Mapping(
            world_size=mapping.world_size,
            rank=mapping.rank,
            gpus_per_node=mapping.gpus_per_node,
            tp_size=group_size,
            pp_size=mapping.world_size // group_size,
        )
        nbytes = cls.buffer_layout(group_size, max_rows, hidden, parts.pop())["end"]
        buffers = PeerBuffers.create(group, nbytes, transport=PeerTransport.IPC)
        if buffers is None:
            return None
        return cls(group, vocab, hidden, max_rows, buffers)

    def supports(self, rows: int, block: int, all_rank_tokens: Optional[Sequence[int]]) -> bool:
        """Whether every member of this rank's group drafts ``rows`` requests of ``block`` positions
        (``all_rank_tokens``: every attention-DP rank's draft tokens, alike on every rank)."""
        if all_rank_tokens is None or not (1 < block <= _MAX_BLOCK and 0 < rows * block):
            return False
        if rows * block > self.max_rows:
            return False
        members = all_rank_tokens[self.first_dp_rank : self.first_dp_rank + self.group_size]
        return len(members) == self.group_size and all(t == rows * block for t in members)

    def propose(
        self,
        normalized: torch.Tensor,
        bonus: torch.Tensor,
        head_values: torch.Tensor,
        head_scales: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> torch.Tensor:
        """This member's ``[B, K]`` draft tokens from BF16 ``normalized [B, K, hidden]`` and int64
        ``bonus [B]``; the whole MXFP8 head and the Markov ``w1`` / ``w2`` are sliced here."""
        from ..attention.kernels.blackwell.dspark import tp_argmax
        from ..custom_ops.torch_custom_ops import mxfp8_gemm_autotuned

        rows, block, hidden = normalized.shape
        g = self.group_size
        count = rows * block
        words = count * hidden // 4
        layout = self.layout
        _push_rows_kernel[(triton.cdiv(words, _PUSH_WORDS),)](
            normalized.view(torch.int64),
            bonus,
            self.peers,
            self.push_ticket,
            self.calls,
            self.consumed,
            layout["rows"] + self.rank * count * hidden * 2,
            layout["bonus"],
            layout["arrival"],
            layout["consumed"],
            words,
            rows,
            self.rank,
            NRANKS=g,
            BLOCK=_PUSH_WORDS,
            COUNT_PAD=triton.next_power_of_2(self.max_rows),
        )
        _wait_kernel[(1,)](self.arrival, self.calls, NRANKS=g)
        act, act_scale = torch.ops.trtllm.mxfp8_quantize(self.rows[: g * count], True)
        start, stop = self.start, self.stop
        scale_columns = head_scales.numel() // triton.cdiv(self.vocab, _ROW_TILE) // _ROW_TILE
        scale_stop = triton.cdiv(stop, _ROW_TILE) * _ROW_TILE
        base = mxfp8_gemm_autotuned(
            act,
            act_scale,
            head_values[start:stop],
            head_scales[start * scale_columns : scale_stop * scale_columns],
            torch.bfloat16,
        ).view(g * rows, block, stop - start)
        latent = torch.empty((g * rows, w1.shape[1]), dtype=torch.bfloat16, device=act.device)
        tp_argmax.masked_embedding(self.bonus[: g * rows], w1, latent)
        w2 = w2[start:stop]
        sampled = []
        for step in range(block):
            bias = F.linear(latent, w2)
            slot = layout["entries"] + step * layout["slot"]
            _candidates_kernel[(g * rows, self.parts)](
                base[:, step],
                base.stride(0),
                bias,
                self.peers,
                self.calls,
                slot,
                g * rows,
                stop - start,
                start,
                self.rank,
                NRANKS=g,
                PARTS=self.parts,
                BLOCK=_CANDIDATE_COLUMNS,
            )
            tokens = torch.empty((g * rows,), dtype=torch.int64, device=act.device)
            # The bias GEMM consumed this step's latent before the next embedding overwrites it.
            _select_kernel[(g * rows,)](
                self.peers,
                self.calls,
                slot,
                tokens,
                w1,
                latent,
                g * rows,
                self.rank,
                self.vocab,
                NRANKS=g,
                PARTS=self.parts,
                DIM=w1.shape[1],
                EMBED=step + 1 < block,
                num_warps=2,
            )
            sampled.append(tokens[self.rank * rows : (self.rank + 1) * rows])
        return torch.stack(sampled, dim=1)


__all__ = ["GroupShardedDraftHead"]
