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
from typing import Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._dlpack_utils import pack_strided_memory
from tensorrt_llm._ipc_utils import IpcMemory, can_access_peer
from tensorrt_llm._torch.distributed.communicator import Distributed
from tensorrt_llm.mapping import Mapping

# int32 bits of a NaN no score takes: marks a slot nothing was pushed to.
_EMPTY = tl.constexpr(-1)
_NUM_SETS = 3


# The slot strides are fixed per process; the rest change from step to step.
@triton.jit(do_not_specialize=["seq_lens", "dql", "out_head_stride", "out_token_stride"])
def _reduce_kernel(
    recv,
    out,
    seq_lens,
    dql,
    src_stride,
    head_stride,
    token_stride,
    out_head_stride,
    out_token_stride,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Programs (local head, request, slice). No griddepcontrol.wait: under PDL
    # this grid starts only after every scorer CTA passed its own wait.
    head, req, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    blocks = tl.cdiv(tl.load(seq_lens + req), 128)
    for start in range(part * BLOCK, blocks * dql, tl.num_programs(2) * BLOCK):
        i = start + tl.arange(0, BLOCK)
        mask = i < blocks * dql
        token, block = req * dql + i // blocks, i % blocks
        slot = recv + head * head_stride + token * token_stride + block
        # Every pass reloads all W sources, so the last one is complete.
        acc = tl.full([BLOCK], float("-inf"), tl.float32)
        pending = 1
        while pending != 0:
            acc = tl.full([BLOCK], float("-inf"), tl.float32)
            missing = tl.zeros([BLOCK], tl.int32)
            for src in tl.static_range(W):
                bits = tl.load(slot + src * src_stride, mask=mask, other=0, volatile=True)
                missing += (bits == _EMPTY).to(tl.int32)
                acc = tl.maximum(acc, bits.to(tl.float32, bitcast=True))
            pending = tl.sum(missing, axis=0)
        for src in tl.static_range(W):
            tl.store(slot + src * src_stride, tl.full([BLOCK], _EMPTY, tl.int32), mask=mask)
        tl.store(out + head * out_head_stride + token * out_token_stride + block, acc, mask=mask)


class IndexTpSplit:
    """Receive slots of a TP group: [sets, W sources, heads / W, tokens, blocks] per rank.

    Consecutive sparse layers, the wrap from the last to the first included,
    use different sets. A rank pushes into a set again only after its own
    reduce of the layer in between has received every rank's pushes, and each
    rank made those only after its reduce of the set's previous layer had
    emptied the set.
    """

    _instances: dict[tuple[int, int, int, tuple[int, ...], int], "IndexTpSplit"] = {}
    # Every split stays allocated: CUDA graphs captured with it address its slots.
    _alive: list["IndexTpSplit"] = []

    @classmethod
    def get(
        cls,
        mapping: Mapping,
        layer_ids: Sequence[int],
        num_heads: int,
        max_blocks: int,
        max_tokens: int,
    ) -> "IndexTpSplit":
        """The process's split of this TP group; created, collectively, on first use or growth."""
        key = (mapping.pp_rank, mapping.tp_rank, mapping.tp_size, tuple(layer_ids), num_heads)
        split = cls._instances.get(key)
        if split is None or split.recv.shape[3] < max_tokens or split.recv.shape[4] < max_blocks:
            split = cls._instances[key] = cls(mapping, layer_ids, num_heads, max_blocks, max_tokens)
            cls._alive.append(split)
        return split

    def __init__(
        self,
        mapping: Mapping,
        layer_ids: Sequence[int],
        num_heads: int,
        max_blocks: int,
        max_tokens: int,
    ) -> None:
        if (
            mapping.enable_attention_dp
            or mapping.tp_size not in (2, 4)
            or not can_access_peer(mapping)
        ):
            raise ValueError(
                "MiniMax-M3 index_tp_split needs TP 2 or 4 with CUDA peer access and no attention DP."
            )
        self.rank, self.world = mapping.tp_rank, mapping.tp_size
        heads = num_heads // self.world
        self.heads = slice(self.rank * heads, (self.rank + 1) * heads)
        n = len(layer_ids)
        self.sets = {layer: i % _NUM_SETS or int(i == n - 1) for i, layer in enumerate(layer_ids)}
        # Rows of whole groups: the scorer pushes a group's W blocks as one vector.
        blocks = -(-max_blocks // self.world) * self.world
        shape = (_NUM_SETS, self.world, heads, max_tokens, blocks)
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
        self,
        layer_idx: int,
        idx_q: torch.Tensor,
        idx_k_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        max_score: torch.Tensor,
        decode_query_len: int,
    ) -> torch.Tensor:
        """Score the decode tokens of idx_q [tokens, heads, dim] into max_score [heads / W, blocks, tokens].

        Returns the view of max_score's storage that holds the scores, blocks
        innermost. Nothing may run between the two kernels: PDL overlaps them.
        """
        recv = self.recv[self.sets[layer_idx]]
        torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
            idx_q,
            idx_k_cache[:, 0, self.rank :: self.world],
            block_table,
            seq_lens,
            recv,
            decode_query_len,
            self.bases[self.sets[layer_idx]],
            self.rank,
        )
        # Blocks innermost, like the slots: the selector reads such rows as vectors.
        heads, blocks, tokens = max_score.shape
        out = max_score.view(heads, tokens, blocks)
        _reduce_kernel[(heads, seq_lens.shape[0], 16)](
            recv.view(torch.int32),
            out,
            seq_lens,
            decode_query_len,
            *recv.stride()[:3],
            *out.stride()[:2],
            W=self.world,
            BLOCK=512,
            launch_pdl=os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1",
        )
        return out.transpose(1, 2)
