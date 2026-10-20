# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 index decode scoring split across the TP ranks (index_tp_split).

Every rank scores all index heads on rows r, r + W, ... of each index-K block
and pushes the partial block maxima of head h into the receive slots of the
rank that owns h. That rank's reduce waits for all W partials, writes their max
into the block selector's score buffer and empties the slots. A block max is
exact, so the score is bitwise the unsplit one while each rank reads 1/W of
index-K.
"""

import math
import os

import torch
import triton
import triton.language as tl

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.mapping import Mapping

# A NaN no score takes: marks a slot nothing was pushed to.
_EMPTY = tl.constexpr(-1)
_NUM_SETS = 3


# The slot strides are fixed per process; the rest change from step to step.
@triton.jit(do_not_specialize=["dql", "out_head_stride", "out_block_stride"])
def _reduce_kernel(
    recv,
    out,
    seq_lens,
    dql,
    src_stride,
    head_stride,
    block_stride,
    out_head_stride,
    out_block_stride,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One program per (local head, request). No griddepcontrol.wait: under PDL
    # this grid starts only after every scorer CTA passed its own wait.
    head, req = tl.program_id(0), tl.program_id(1)
    n = tl.cdiv(tl.load(seq_lens + req), 128) * dql
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        mask = i < n
        block, token = i // dql, req * dql + i % dql
        acc = tl.full([BLOCK], float("-inf"), tl.float32)
        for src in tl.static_range(W):
            slot = recv + src * src_stride + head * head_stride + block * block_stride + token
            bits = tl.load(slot, mask=mask, other=0, volatile=True)
            while tl.max(((bits == _EMPTY) & mask).to(tl.int32), axis=0) != 0:
                bits = tl.load(slot, mask=mask, other=0, volatile=True)
            acc = tl.maximum(acc, bits.to(tl.float32, bitcast=True))
            tl.store(slot, tl.full([BLOCK], _EMPTY, tl.int32), mask=mask)
        tl.store(out + head * out_head_stride + block * out_block_stride + token, acc, mask=mask)


class IndexTpSplit:
    """Receive slots of a TP group: [sets, W sources, heads / W, blocks, tokens] per rank.

    Consecutive sparse layers, the wrap from the last to the first included,
    use different sets, so a rank refills a set only after its owner's reduce
    of the layer in between has seen that rank's pushes.
    """

    _instances = {}
    # Every split stays allocated: CUDA graphs captured with it address its slots.
    _alive = []

    @classmethod
    def get(cls, mapping: Mapping, layer_ids, num_heads: int, max_blocks: int, max_tokens: int):
        """The process's split of this TP group; created, collectively, on first use or growth."""
        key = (mapping.pp_rank, mapping.tp_rank, mapping.tp_size)
        split = cls._instances.get(key)
        if split is None or split.recv.shape[3] < max_blocks or split.recv.shape[4] < max_tokens:
            split = cls._instances[key] = cls(mapping, layer_ids, num_heads, max_blocks, max_tokens)
            cls._alive.append(split)
        return split

    def __init__(
        self, mapping: Mapping, layer_ids, num_heads: int, max_blocks: int, max_tokens: int
    ):
        if not can_access_peer(mapping):
            raise RuntimeError(
                "MiniMax-M3 index_tp_split needs CUDA peer access between the TP ranks."
            )
        self.rank, self.world = mapping.tp_rank, mapping.tp_size
        heads = num_heads // self.world
        self.heads = slice(self.rank * heads, (self.rank + 1) * heads)
        n = len(layer_ids)
        self.sets = {layer: i % _NUM_SETS or int(i == n - 1) for i, layer in enumerate(layer_ids)}
        shape = (_NUM_SETS, self.world, heads, max_blocks, max_tokens)
        nbytes = math.prod(shape) * 4
        self._ipc = IpcMemory(mapping, nbytes)
        ptrs = self._ipc.peer_ptrs
        self.recv = pack_strided_memory(
            ptrs[self.rank], nbytes, nbytes, 1, torch.float32, torch.cuda.current_device()
        ).view(shape)
        self.bases = torch.tensor(
            [[ptr + s * nbytes // _NUM_SETS for ptr in ptrs] for s in range(_NUM_SETS)],
            dtype=torch.int64,
            device="cuda",
        )
        # After IpcMemory's zeroing (legacy stream) and before any peer pushes.
        torch.cuda.synchronize()
        self.recv.view(torch.int32).fill_(_EMPTY.value)
        torch.cuda.synchronize()
        Distributed.get(mapping).tp_barrier()

    def score(
        self, layer_idx, idx_q, idx_k_cache, block_table, seq_lens, max_score, decode_query_len
    ):
        """Fill max_score [heads / W, blocks, tokens] for the decode tokens of idx_q [tokens, heads, dim].

        Nothing may run between the two kernels: PDL overlaps them.
        """
        recv = self.recv[self.sets[layer_idx]]
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q,
            idx_k_cache[:, 0, self.rank :: self.world],
            block_table,
            seq_lens,
            recv.transpose(2, 3),
            decode_query_len,
            self.bases[self.sets[layer_idx]],
            self.rank,
        )
        _reduce_kernel[(recv.shape[1], seq_lens.shape[0])](
            recv.view(torch.int32),
            max_score,
            seq_lens,
            decode_query_len,
            *recv.stride()[:3],
            *max_score.stride()[:2],
            W=self.world,
            BLOCK=512,
            launch_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1",
        )
