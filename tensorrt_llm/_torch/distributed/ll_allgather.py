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
"""One-launch all-gather of small tensors over a tensor-parallel group (peer memory, no NCCL).

``LowLatencyAllGather.gather(x)`` returns what ``allgather(x, mapping, dim=-1)`` returns for a
contiguous ``[rows, width]`` shard: every rank's shard side by side in rank order, bit for bit.

Protocol (NCCL's LL idea): every 4-byte word of a shard travels as one 8-byte packet
``(word, epoch)`` that each rank stores into every rank's receive slot; a reader polls its own slot
until the packets of all ranks carry the current epoch, so data and flag arrive in one store and no
slot needs clearing. The epoch (a device counter, advanced by the last program of each launch) is
the same on every rank because every rank launches the same gathers in the same order; slots
alternate with its parity, which is enough: a rank can only start launch ``e + 2`` after every
rank has read launch ``e + 1``'s packets, which it wrote after reading launch ``e``'s.

Each program pushes its words of this rank's shard to every rank, then waits for the same words
from every rank and writes them to the output. Programs never wait for other programs of the same
launch, and the grid stays within one wave.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Largest shard per rank a gather takes; larger gathers keep NCCL.
_DEFAULT_MAX_BYTES = 256 * 1024  # 256 KiB
# Words per program: small blocks spread the remote stores and the polling over more SMs.
_BLOCK = 128


@triton.jit(do_not_specialize=["rank"])
def _ll_allgather_kernel(
    src_ptr,
    out_ptr,
    state_ptr,
    peers_ptr,
    shard_words,
    row_words,
    num_blocks,
    rank,
    NRANKS: tl.constexpr,
    SLOT_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """``state_ptr``: int32 ``[2]`` (epoch, arrival ticket); ``peers_ptr``: int64 ``[NRANKS]``
    receive buffers (two slots of ``SLOT_WORDS`` 8-byte packets, source-rank major). Program ``p``
    moves blocks ``p, p + grid, ...`` of ``BLOCK`` words: it pushes all of them before it waits."""
    program = tl.program_id(0)
    stride = tl.num_programs(0)
    epoch = tl.load(state_ptr)
    slot = (epoch & 1) * SLOT_WORDS
    tag = epoch.to(tl.uint32).to(tl.uint64) << 32
    for block in range(program, num_blocks, stride):
        w = block * BLOCK + tl.arange(0, BLOCK)
        ok = w < shard_words
        words = tl.load(src_ptr + w, mask=ok, other=0)
        packet = (tag | words.to(tl.uint32, bitcast=True).to(tl.uint64)).to(tl.int64, bitcast=True)
        for peer in tl.static_range(NRANKS):
            dst = tl.load(peers_ptr + peer).to(tl.uint64).to(tl.pointer_type(tl.int64))
            tl.store(dst + slot + rank * shard_words + w, packet, mask=ok)
    # Keep the pushes above ahead of the polling below.
    tl.debug_barrier()

    mine = tl.load(peers_ptr + rank).to(tl.uint64).to(tl.pointer_type(tl.int64))
    src = tl.arange(0, NRANKS)
    for block in range(program, num_blocks, stride):
        w = block * BLOCK + tl.arange(0, BLOCK)
        ok = w < shard_words
        offs = slot + src[:, None] * shard_words + w[None, :]
        mask = (src[:, None] < NRANKS) & ok[None, :]
        got = tl.load(mine + offs, mask=mask, other=0, volatile=True)
        waiting = tl.sum(((((got >> 32).to(tl.int32)) != epoch) & mask).to(tl.int32))
        while waiting > 0:
            got = tl.load(mine + offs, mask=mask, other=0, volatile=True)
            waiting = tl.sum(((((got >> 32).to(tl.int32)) != epoch) & mask).to(tl.int32))
        # Output row t holds every rank's row t side by side (all-gather along the last dim).
        row = w // row_words
        col = w % row_words
        out_offs = row[None, :] * (NRANKS * row_words) + src[:, None] * row_words + col[None, :]
        tl.store(out_ptr + out_offs, got.to(tl.int32), mask=mask)

    # The last program of the launch advances the epoch (every program read it above).
    ticket = tl.atomic_add(state_ptr + 1, 1, sem="acq_rel", scope="gpu")
    if ticket == stride - 1:
        tl.store(state_ptr, epoch + 1)
        tl.store(state_ptr + 1, 0)


class LowLatencyAllGather:
    """All-gather along the last dimension for small shards of one tensor-parallel group.

    Every rank of the group constructs it at the same point (:meth:`create` maps the peer
    buffers) and must then issue the same sequence of :meth:`gather` calls (same shapes), as
    with any collective.
    """

    def __init__(
        self, rank: int, nranks: int, peer_ptrs, max_shard_bytes: int, keepalive: object = None
    ):
        if nranks & (nranks - 1) or len(peer_ptrs) != nranks or not 0 <= rank < nranks:
            raise ValueError(f"unsupported gather group: rank {rank} of {nranks}")
        self.rank = rank
        self.nranks = nranks
        self.max_shard_words = max_shard_bytes // 4
        device = torch.device("cuda", torch.cuda.current_device())
        self._peers = torch.tensor(list(peer_ptrs), dtype=torch.int64, device=device)
        # Zero-filled buffers carry epoch 0 everywhere, so the first launch's epoch is 1.
        self._state = torch.tensor([1, 0], dtype=torch.int32, device=device)
        self._num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # The peer allocation (and its mappings) lives as long as the gather.
        self._keepalive = keepalive

    @staticmethod
    def buffer_bytes(nranks: int, max_shard_bytes: int) -> int:
        """Bytes of one rank's receive buffer (two slots of 8-byte packets)."""
        return 2 * nranks * (max_shard_bytes // 4) * 8

    @classmethod
    def create(cls, mapping) -> Optional["LowLatencyAllGather"]:
        """The gather of ``mapping``'s TP group, or None where peer memory is unavailable.

        Collective over the TP group; every rank must call it in step.
        """
        from .peer_buffers import PeerBuffers

        if (
            mapping.tp_size not in (2, 4, 8)
            or mapping.enable_attention_dp
            or mapping.has_pp()
            or mapping.cp_size > 1
        ):
            return None
        buffers = PeerBuffers.create(mapping, cls.buffer_bytes(mapping.tp_size, _DEFAULT_MAX_BYTES))
        if buffers is None:
            return None
        return cls(
            mapping.tp_rank,
            mapping.tp_size,
            buffers.peer_ptrs,
            _DEFAULT_MAX_BYTES,
            keepalive=buffers,
        )

    @property
    def transport(self) -> str:
        return self._keepalive.transport.value

    def supports(self, x: torch.Tensor) -> bool:
        """Whether :meth:`gather` takes this shard."""
        if not (x.is_cuda and x.dim() == 2 and x.is_contiguous() and x.numel()):
            return False
        row_bytes = x.shape[1] * x.element_size()
        shard_words = x.numel() * x.element_size() // 4
        return row_bytes % 4 == 0 and shard_words <= self.max_shard_words

    def gather(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """``[rows, nranks * width]``: every rank's ``[rows, width]`` shard in rank order, or None
        when the shard is unsupported (then no rank may launch: the shapes agree across ranks)."""
        if not self.supports(x):
            return None
        rows, width = x.shape
        out = torch.empty((rows, self.nranks * width), dtype=x.dtype, device=x.device)
        shard_words = x.numel() * x.element_size() // 4
        num_blocks = triton.cdiv(shard_words, _BLOCK)
        # At most one wave, so every program of the launch is resident while it waits.
        _ll_allgather_kernel[(min(num_blocks, self._num_sms),)](
            x.view(torch.int32),
            out.view(torch.int32),
            self._state,
            self._peers,
            shard_words,
            width * x.element_size() // 4,
            num_blocks,
            self.rank,
            NRANKS=self.nranks,
            SLOT_WORDS=self.nranks * self.max_shard_words,
            BLOCK=_BLOCK,
            num_warps=4,
        )
        return out


__all__ = ["LowLatencyAllGather"]
