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
"""Engram lookups under attention DP through peer memory, for decode-sized forwards.

Under attention DP each rank of an Engram exchange group (``DeepseekV41Engram``: the tensor-parallel
group, or the largest rank group whose size divides the head count) holds contiguous heads of each
table and needs all heads of its own tokens. The NCCL path all-gathers every rank's hash rows over
the tensor-parallel group, looks up the group's tokens in the local heads and sends each rank its
tokens' rows (``alltoallv_tokens``).

Here a forward does:

1. :meth:`EngramAdpExchange.exchange_hashes` (model stream, one launch for every Engram layer):
   every rank pushes its row count and hash rows to every rank of its group and gathers theirs.
2. :meth:`EngramAdpExchange.push` (one launch per layer where the lookup runs): every rank looks up
   its heads of every group token, dequantizing exactly like the lookup kernel, and stores each row
   straight into the requesting rank's receive buffer.
3. :meth:`EngramAdpExchange.receive` (model stream, one launch per layer, at the layer): every rank
   waits for its rows and copies them out.

Each head of a token has one owner, so the result is the NCCL path's, bit for bit.

Protocol (NCCL's LL idea): every 4-byte word travels as one 8-byte packet ``(word, epoch)`` and a
reader polls its own buffer until the packets carry the forward's epoch, so data and flag arrive in
one store and nothing needs clearing. The epoch is a device counter the hash exchange advances;
every rank runs the same exchanges in the same order (the decision to use this path reads only state
all ranks share), so it is the same on every rank of a group.

- The hash slots alternate with the epoch's parity: a rank can only push round ``e + 2`` after every
  rank has read round ``e + 1``, which it pushed after reading round ``e``.
- The row buffers need one slot per layer: an owner pushes round ``e + 1`` rows to a rank only after
  it has that rank's round ``e + 1`` hashes, which the rank pushed after its round ``e`` receives.
- Only the exchange and the receive wait on other ranks, both on the model stream. The push never
  waits, so it may run on a side stream; a rank's model stream waits for its own push before its
  receive, so no rank sits in a later collective while its push is still pending.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

from .engram import ShardedFp8MultiHeadEmbedding, _multi_processor_count

# Largest token budget per rank (max_num_tokens) of an engine that takes the exchange (it sizes the
# peer buffers): decode-only engines; others keep the NCCL path.
_MAX_ROWS = 256

_HASH_BLOCK = 512
_RECV_BLOCK = 1024
# Rows per push iteration; the push takes half the SMs, as the lookup it replaces, which shares the
# GPU with the model stream's first layer.
_PUSH_ROWS = 16
_PUSH_SM_FRACTION = 0.5


@triton.jit
def _mul_rn(a, b):
    """FP32 ``a * b`` rounded to nearest, subnormals kept (as the lookup kernel's ``mul.rn.f32``)."""
    return tl.inline_asm_elementwise(
        """
        {
            .reg .f32 x, y, z;
            mov.b32 x, $1;
            mov.b32 y, $2;
            mul.rn.f32 z, x, y;
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
def _e4m3x2_to_f32(pair):
    """Exact FP32 values of the two E4M3 bytes in the low 16 bits (the lookup kernel's conversion)."""
    low, high = tl.inline_asm_elementwise(
        """
        {
            .reg .b16 s, lo, hi;
            .reg .b32 h;
            cvt.u16.u32 s, $2;
            cvt.rn.f16x2.e4m3x2 h, s;
            mov.b32 {lo, hi}, h;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        constraints="=r,=r,r",
        args=[pair],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )
    return low.to(tl.float32, bitcast=True), high.to(tl.float32, bitcast=True)


@triton.jit
def _bf16x2(a, b, scale):
    """BF16 bits of ``a * scale`` (low half) and ``b * scale`` (high half), as the lookup kernel."""
    lo = _mul_rn(a, scale).to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    hi = _mul_rn(b, scale).to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    return lo | (hi << 16)


@triton.jit(do_not_specialize=["stride0", "stride1", "count", "max_count", "rank"])
def _hash_exchange_kernel(
    h0_ptr,
    h1_ptr,
    stride0,
    stride1,
    out_ptr,
    counts_ptr,
    state_ptr,
    peers_ptr,
    count,
    max_count,
    rank,
    NR: tl.constexpr,
    NR_P2: tl.constexpr,
    NL: tl.constexpr,
    C: tl.constexpr,
    H2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``h0_ptr`` / ``h1_ptr``: int32 views of the layers' int64 ``[count, H]`` hash rows (row
    strides in int32 words); ``out_ptr``: int32 view of int64 ``[NR, NL, max_count, H]``;
    ``counts_ptr``: int32 ``[NR]``; ``state_ptr``: int32 ``[2]`` (epoch, arrival ticket);
    ``peers_ptr``: int64 ``[NR]`` packet buffers of the group, hash slots first (two parity slots of
    ``NR`` source-rank slots, each a count packet followed by ``NL * C * H2`` word packets)."""
    SLOT: tl.constexpr = 1 + NL * C * H2
    program = tl.program_id(0)
    grid = tl.num_programs(0)
    epoch = tl.load(state_ptr)
    parity = (epoch & 1) * (NR * SLOT)
    tag = epoch.to(tl.uint32).to(tl.uint64) << 32

    # This rank's count and words into its slot of every group rank's buffer.
    if program == 0:
        header = (tag | count.to(tl.uint32).to(tl.uint64)).to(tl.int64, bitcast=True)
        for peer in tl.static_range(NR):
            dst = tl.load(peers_ptr + peer).to(tl.uint64).to(tl.pointer_type(tl.int64))
            tl.store(dst + parity + rank * SLOT, header)
    layer_words = count * H2
    push_words = NL * layer_words
    for start in range(program * BLOCK, push_words, grid * BLOCK):
        w = start + tl.arange(0, BLOCK)
        ok = w < push_words
        layer = w // layer_words
        rem = w - layer * layer_words
        t = rem // H2
        j = rem - t * H2
        if NL == 2:
            first = layer == 0
            words = tl.where(
                first,
                tl.load(h0_ptr + t * stride0 + j, mask=ok & first, other=0),
                tl.load(h1_ptr + t * stride1 + j, mask=ok & ~first, other=0),
            )
        else:
            words = tl.load(h0_ptr + t * stride0 + j, mask=ok, other=0)
        packet = (tag | words.to(tl.uint32, bitcast=True).to(tl.uint64)).to(tl.int64, bitcast=True)
        offs = parity + rank * SLOT + 1 + (layer * C + t) * H2 + j
        for peer in tl.static_range(NR):
            dst = tl.load(peers_ptr + peer).to(tl.uint64).to(tl.pointer_type(tl.int64))
            tl.store(dst + offs, packet, mask=ok)
    # Keep the pushes above ahead of the polling below.
    tl.debug_barrier()

    # Every group rank's count, then its words (rows beyond its count are not sent).
    mine = tl.load(peers_ptr + rank).to(tl.uint64).to(tl.pointer_type(tl.int64))
    src = tl.arange(0, NR_P2)
    live = src < NR
    head = tl.load(mine + parity + src * SLOT, mask=live, other=0, volatile=True)
    while tl.sum(((head >> 32).to(tl.int32) != epoch) & live) > 0:
        head = tl.load(mine + parity + src * SLOT, mask=live, other=0, volatile=True)
    if program == 0:
        tl.store(counts_ptr + src, head.to(tl.int32), mask=live)
    per_layer = max_count * H2
    per_src = NL * per_layer
    total = NR * per_src
    for start in range(program * BLOCK, total, grid * BLOCK):
        w = start + tl.arange(0, BLOCK)
        ok = w < total
        s = w // per_src
        rem = w - s * per_src
        layer = rem // per_layer
        rem = rem - layer * per_layer
        t = rem // H2
        j = rem - t * H2
        # This program saw every count packet above with the current epoch.
        cnt = tl.load(mine + parity + s * SLOT, mask=ok, other=0, volatile=True).to(tl.int32)
        valid = ok & (t < cnt)
        offs = parity + s * SLOT + 1 + (layer * C + t) * H2 + j
        got = tl.load(mine + offs, mask=valid, other=0, volatile=True)
        while tl.sum((((got >> 32).to(tl.int32)) != epoch) & valid) > 0:
            got = tl.load(mine + offs, mask=valid, other=0, volatile=True)
        tl.store(out_ptr + w, got.to(tl.int32), mask=valid)

    # The last program of the launch advances the epoch (every program read it above).
    ticket = tl.atomic_add(state_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == grid - 1:
        tl.store(state_ptr, epoch + 1)
        tl.store(state_ptr + 1, 0)


@triton.jit(do_not_specialize=["rows_base", "max_count", "num_rows", "grid_size"])
def _owner_push_kernel(
    weight_ptr,
    scale_ptr,
    offsets_ptr,
    gathered_ptr,
    counts_ptr,
    state_ptr,
    peers_ptr,
    rows_base,
    max_count,
    num_rows,
    grid_size,
    LAYER: tl.constexpr,
    NL: tl.constexpr,
    H: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    HEAD_START: tl.constexpr,
    DIM: tl.constexpr,
    QB: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Rows ``(source rank, token, local head)`` in that order; ``weight_ptr``: int32 view of the
    E4M3 rows (four values per word); ``gathered_ptr``: int64 ``[NR, NL, max_count, H]``
    bucket-local indices; ``counts_ptr``: int32 ``[NR]``; a pushed row lands at packet
    ``rows_base + (token * H + head) * DIM / 2`` of its source rank's buffer. The values are the
    lookup kernel's: E4M3 times ``2 ** (code - 127)`` (code 0 is ``2 ** -127``, 255 is NaN) in
    FP32, rounded once to BF16."""
    HALF: tl.constexpr = DIM // 2
    WORDS: tl.constexpr = DIM // 4
    BLOCKS: tl.constexpr = DIM // QB
    epoch = tl.load(state_ptr) - 1
    tag = epoch.to(tl.uint32).to(tl.uint64) << 32
    words = tl.arange(0, WORDS)
    blocks = tl.arange(0, BLOCKS)
    packets = tl.arange(0, HALF)
    per_src = max_count * LOCAL_HEADS
    for base in tl.range(tl.program_id(0) * BLOCK_R, num_rows, grid_size * BLOCK_R):
        rows = base + tl.arange(0, BLOCK_R)
        in_range = rows < num_rows
        src = rows // per_src
        rem = rows - src * per_src
        token = rem // LOCAL_HEADS
        local_head = rem - token * LOCAL_HEADS
        count = tl.load(counts_ptr + src, mask=in_range, other=0)
        valid = in_range & (token < count)
        head = HEAD_START + local_head
        index = tl.load(
            gathered_ptr + ((src.to(tl.int64) * NL + LAYER) * max_count + token) * H + head,
            mask=valid,
            other=0,
        ).to(tl.int64)
        index += tl.load(offsets_ptr + local_head, mask=valid, other=0).to(tl.int64)
        gather = valid[:, None]
        w = tl.load(weight_ptr + index[:, None] * WORDS + words[None, :], mask=gather, other=0)
        w = w.to(tl.uint32, bitcast=True)
        code = tl.load(
            scale_ptr + index[:, None] * BLOCKS + blocks[None, :], mask=gather, other=127
        ).to(tl.int32)
        bits = tl.where(code == 0, 0x00400000, tl.where(code == 255, 0x7FC00000, code << 23))
        # One scale per word: each block's scale repeated over its QB / 4 words.
        scale = tl.reshape(
            tl.broadcast_to(bits[:, :, None], (BLOCK_R, BLOCKS, QB // 4)), (BLOCK_R, WORDS)
        ).to(tl.float32, bitcast=True)
        a0, a1 = _e4m3x2_to_f32(w & 0xFFFF)
        c0, c1 = _e4m3x2_to_f32(w >> 16)
        low = (tag | _bf16x2(a0, a1, scale).to(tl.uint64)).to(tl.int64, bitcast=True)
        high = (tag | _bf16x2(c0, c1, scale).to(tl.uint64)).to(tl.int64, bitcast=True)
        packet = tl.reshape(tl.join(low, high), (BLOCK_R, HALF))
        dst = (
            tl.load(peers_ptr + src, mask=valid, other=0)
            .to(tl.uint64)
            .to(tl.pointer_type(tl.int64))
        )
        offs = rows_base + (token.to(tl.int64) * H + head) * HALF
        tl.store(dst[:, None] + offs[:, None] + packets[None, :], packet, mask=gather)


@triton.jit(do_not_specialize=["rows_base", "num_words", "rank"])
def _receive_kernel(
    out_ptr,
    state_ptr,
    peers_ptr,
    rows_base,
    num_words,
    rank,
    BLOCK: tl.constexpr,
):
    """Wait for this rank's ``num_words`` row packets of the current epoch (written by their
    owners) and copy their words to ``out_ptr`` (int32 view of ``[tokens, H * DIM]``)."""
    epoch = tl.load(state_ptr) - 1
    mine = tl.load(peers_ptr + rank).to(tl.uint64).to(tl.pointer_type(tl.int64))
    for start in range(tl.program_id(0) * BLOCK, num_words, tl.num_programs(0) * BLOCK):
        w = start + tl.arange(0, BLOCK)
        ok = w < num_words
        got = tl.load(mine + rows_base + w, mask=ok, other=0, volatile=True)
        while tl.sum((((got >> 32).to(tl.int32)) != epoch) & ok) > 0:
            got = tl.load(mine + rows_base + w, mask=ok, other=0, volatile=True)
        tl.store(out_ptr + w, got.to(tl.int32), mask=ok)


@dataclass
class ExchangeRound:
    """One forward's exchanged hashes: ``gathered`` int64 ``[NR, NL, max_count, H]`` (rows beyond
    each group rank's count unset), ``counts`` int32 ``[NR]`` on the device, and the hash tensors
    it took."""

    gathered: torch.Tensor
    counts: torch.Tensor
    max_count: int
    local_count: int
    hashes: tuple


class EngramAdpExchange:
    """Peer buffers and launches of the attention-DP Engram exchange of one group of ``nranks``
    ranks for ``num_layers`` layers of ``heads x dim`` lookups and at most ``capacity`` tokens per
    rank. Every rank of the group issues the same sequence of exchanges (one per forward that
    takes the exchange), pushes and receives."""

    def __init__(
        self,
        rank: int,
        nranks: int,
        peer_ptrs: Sequence[int],
        capacity: int,
        heads: int,
        dim: int,
        num_layers: int,
        keepalive: object = None,
    ):
        if not 0 <= rank < nranks or len(peer_ptrs) != nranks:
            raise ValueError(f"unsupported exchange group: rank {rank} of {nranks}")
        if num_layers not in (1, 2) or dim < 4 or dim & (dim - 1):
            raise ValueError(f"unsupported Engram exchange shape: {num_layers} layers, dim {dim}")
        self.rank = rank
        self.nranks = nranks
        self.capacity = capacity
        self.heads = heads
        self.dim = dim
        self.num_layers = num_layers
        device = torch.device("cuda", torch.cuda.current_device())
        self._peers = torch.tensor(list(peer_ptrs), dtype=torch.int64, device=device)
        # Epoch 0 = never sent (zero-filled buffers); the first exchange is epoch 1.
        self._state = torch.tensor([1, 0], dtype=torch.int32, device=device)
        num_sms = _multi_processor_count(device.index)
        self._max_programs = num_sms
        self._push_programs = max(1, int(num_sms * _PUSH_SM_FRACTION))
        self._keepalive = keepalive

    @staticmethod
    def _hash_slot_packets(capacity: int, heads: int, num_layers: int) -> int:
        return 1 + num_layers * capacity * heads * 2

    @classmethod
    def buffer_bytes(cls, nranks: int, capacity: int, heads: int, dim: int, num_layers: int) -> int:
        hash_packets = 2 * nranks * cls._hash_slot_packets(capacity, heads, num_layers)
        row_packets = num_layers * capacity * heads * (dim // 2)
        return 8 * (hash_packets + row_packets)

    def _rows_base(self, layer_index: int) -> int:
        hash_packets = (
            2 * self.nranks * self._hash_slot_packets(self.capacity, self.heads, self.num_layers)
        )
        return hash_packets + layer_index * self.capacity * self.heads * (self.dim // 2)

    @classmethod
    def create(
        cls,
        mapping,
        group_start: int,
        group_size: int,
        capacity: int,
        heads: int,
        dim: int,
        num_layers: int,
    ) -> Optional["EngramAdpExchange"]:
        """The exchange of the group of tensor-parallel ranks ``[group_start, group_start +
        group_size)`` that holds ``mapping``'s rank, or None where the tensor-parallel group has no
        peer memory (every rank alike). Collective over the tensor-parallel group."""
        from ...distributed.peer_buffers import PeerBuffers

        nbytes = cls.buffer_bytes(group_size, capacity, heads, dim, num_layers)
        buffers = PeerBuffers.create(mapping, nbytes)
        if buffers is None:
            return None
        exchange = cls(
            mapping.tp_rank - group_start,
            group_size,
            buffers.peer_ptrs[group_start : group_start + group_size],
            capacity,
            heads,
            dim,
            num_layers,
            keepalive=buffers,
        )
        logger.info(
            f"DeepSeek-V4.1 Engram ADP exchange: groups of {group_size} of {mapping.tp_size} "
            f"ranks, {capacity} rows per rank, {num_layers} layers, {nbytes / 2**20:.1f} MiB "
            f"{buffers.transport.value} peer memory"
        )
        return exchange

    def takes(self, counts: Sequence[int]) -> bool:
        """Whether a forward with these per-rank token counts (every tensor-parallel rank's) uses
        the exchange; every rank answers alike."""
        return 0 < sum(counts) and max(counts) <= self.capacity

    def exchange_hashes(
        self, hashes: Sequence[torch.Tensor], counts: Sequence[int]
    ) -> ExchangeRound:
        """Gather the ``[count, heads]`` int64 hash rows of every group rank (``counts``, in group
        order) for every layer (one launch on the current stream)."""
        if len(hashes) != self.num_layers or len(counts) != self.nranks:
            raise ValueError("one hash tensor per Engram layer and one count per group rank")
        local = counts[self.rank]
        for h in hashes:
            if (
                h.dtype != torch.int64
                or h.dim() != 2
                or h.shape != (local, self.heads)
                or h.stride(1) != 1
                or not h.is_cuda
            ):
                raise ValueError(
                    f"Engram exchange hashes must be int64 [{local}, {self.heads}] with unit "
                    f"column stride, got {h.dtype} {tuple(h.shape)} strides {h.stride()}"
                )
        device = hashes[0].device
        max_count = max(counts)
        gathered = torch.empty(
            (self.nranks, self.num_layers, max_count, self.heads),
            dtype=torch.int64,
            device=device,
        )
        counts_dev = torch.empty(self.nranks, dtype=torch.int32, device=device)
        h2 = 2 * self.heads
        push_blocks = triton.cdiv(self.num_layers * local * h2, _HASH_BLOCK)
        poll_blocks = triton.cdiv(self.nranks * self.num_layers * max_count * h2, _HASH_BLOCK)
        grid = max(1, min(max(push_blocks, poll_blocks), self._max_programs))
        # int64 rows travel as pairs of 32-bit words (an empty tensor views the same way).
        _hash_exchange_kernel[(grid,)](
            hashes[0].view(torch.int32),
            hashes[-1].view(torch.int32),
            2 * hashes[0].stride(0),
            2 * hashes[-1].stride(0),
            gathered.view(torch.int32),
            counts_dev,
            self._state,
            self._peers,
            local,
            max_count,
            self.rank,
            NR=self.nranks,
            NR_P2=triton.next_power_of_2(self.nranks),
            NL=self.num_layers,
            C=self.capacity,
            H2=h2,
            BLOCK=_HASH_BLOCK,
            num_warps=4,
        )
        return ExchangeRound(gathered, counts_dev, max_count, local, tuple(hashes))

    def push(
        self, round_: ExchangeRound, layer_index: int, table: ShardedFp8MultiHeadEmbedding
    ) -> None:
        """Look up this rank's heads of every group rank's tokens for layer ``layer_index`` and
        store them into the requesters' receive buffers (current stream; never waits on others)."""
        num_rows = self.nranks * round_.max_count * table.num_local_heads
        if num_rows == 0:
            return
        weight, scale = table.storage()
        grid = max(1, min(triton.cdiv(num_rows, _PUSH_ROWS), self._push_programs))
        _owner_push_kernel[(grid,)](
            weight.view(torch.uint8).view(torch.int32),
            scale.view(torch.uint8),
            table.offsets,
            round_.gathered,
            round_.counts,
            self._state,
            self._peers,
            self._rows_base(layer_index),
            round_.max_count,
            num_rows,
            grid,
            LAYER=layer_index,
            NL=self.num_layers,
            H=self.heads,
            LOCAL_HEADS=table.num_local_heads,
            HEAD_START=table.head_start,
            DIM=self.dim,
            QB=table.block_size,
            BLOCK_R=_PUSH_ROWS,
            num_warps=4,
        )

    def receive(self, round_: ExchangeRound, layer_index: int) -> torch.Tensor:
        """This rank's BF16 ``[local tokens, heads * dim]`` rows of layer ``layer_index`` (current
        stream; waits until every owner has pushed them)."""
        out = torch.empty(
            (round_.local_count, self.heads * self.dim),
            dtype=torch.bfloat16,
            device=round_.gathered.device,
        )
        num_words = round_.local_count * self.heads * (self.dim // 2)
        if num_words == 0:
            return out
        grid = max(1, min(triton.cdiv(num_words, _RECV_BLOCK), self._max_programs))
        _receive_kernel[(grid,)](
            out.view(torch.int32),
            self._state,
            self._peers,
            self._rows_base(layer_index),
            num_words,
            self.rank,
            BLOCK=_RECV_BLOCK,
            num_warps=4,
        )
        return out


def engine_capacity(max_num_tokens: int) -> Optional[int]:
    """Rows per rank of an engine's exchange: its ``max_num_tokens``, or None above ``_MAX_ROWS``."""
    return max_num_tokens if 0 < max_num_tokens <= _MAX_ROWS else None


__all__ = ["EngramAdpExchange", "ExchangeRound", "engine_capacity"]
