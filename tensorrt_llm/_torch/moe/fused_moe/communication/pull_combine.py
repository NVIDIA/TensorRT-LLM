# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pull combine for the two-rank NVLink one-sided MoE all-to-all.

With two EP ranks and the expert output in the workspace, a rank's token has at most two partial
outputs: its own experts' row in the local combine payload and the peer's row in the peer's
payload, both at the token's send index. After a handshake that orders both ranks' FC2 before any
read, one kernel per token reads the two rows (the peer's over NVLink) and writes their FP32 sum
in BF16, optionally plus an addend (the shared expert's output) with the rounding of a separate
add. It replaces the native prepare-combine and combine kernels and the add, with their bits:
the native combine also sums a token's partials in FP32 with zeros for absent ones, and the
partials, accumulated from a zeroed buffer, are never -0.0.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm.mapping import Mapping

# Combines whose busiest rank has fewer tokens keep the native combine.
PULL_COMBINE_MIN_TOKENS = 4096
_FLAG_BYTES = 256


@triton.jit
def _handshake_kernel(epoch_ptr, peer_flag_ptr, own_flag_ptr):
    # Both ranks run the same sequence of handshakes; this rank's count is the epoch. The system
    # release publishes this rank's earlier work (its FC2 rows) before the flag; the acquire after
    # the wait orders the following reads of the peer's rows after the peer's FC2.
    epoch = tl.load(epoch_ptr) + 1
    tl.atomic_xchg(peer_flag_ptr, epoch, sem="release", scope="sys")
    seen = tl.load(own_flag_ptr, volatile=True)
    while seen < epoch:
        seen = tl.load(own_flag_ptr, volatile=True)
    tl.atomic_add(own_flag_ptr, 0, sem="acquire", scope="sys")
    tl.store(epoch_ptr, epoch)


@triton.jit
def _pull_reduce_kernel(
    own_ptr,
    peer_ptr,
    ranks_ptr,
    indices_ptr,
    addend_ptr,
    out_ptr,
    hidden,
    ep_rank,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
):
    # A duplicate target rank carries send index -1, so each rank holds at most one valid slot.
    token = tl.program_id(0).to(tl.int64)
    own_row = -1
    peer_row = -1
    for k in tl.static_range(TOP_K):
        rank = tl.load(ranks_ptr + token * TOP_K + k)
        index = tl.load(indices_ptr + token * TOP_K + k)
        own_row = tl.where((index >= 0) & (rank == ep_rank), index, own_row)
        peer_row = tl.where((index >= 0) & (rank != ep_rank), index, peer_row)
    own_base = own_ptr + tl.maximum(own_row, 0).to(tl.int64) * hidden
    peer_base = peer_ptr + tl.maximum(peer_row, 0).to(tl.int64) * hidden
    for start in range(0, hidden, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < hidden
        own = tl.load(own_base + offsets, mask=mask & (own_row >= 0), other=0.0).to(tl.float32)
        peer = tl.load(peer_base + offsets, mask=mask & (peer_row >= 0), other=0.0).to(tl.float32)
        out = (own + peer).to(tl.bfloat16)
        if HAS_ADDEND:
            shared = tl.load(addend_ptr + token * hidden + offsets, mask=mask, other=0.0)
            out = (shared.to(tl.float32) + out.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + token * hidden + offsets, out, mask=mask)


class PullCombineSync:
    """The handshake flags of one EP2 workspace, in their own NVLink memory, and this rank's epoch.

    Construction is collective (an MNNVL allocation and a barrier), at communicator construction.
    """

    def __init__(self, mapping: Mapping, ep_rank: int):
        self._memory = MnnvlMemory(mapping, _FLAG_BYTES)
        # The DLPack-backed tensor must outlive every view of its storage, so views are
        # taken per use rather than kept.
        self._flags = self._memory.as_torch_strided_tensor(torch.uint8)
        self._ep_rank = ep_rank
        self._flag(ep_rank).zero_()
        self._epoch = torch.zeros(1, dtype=torch.int32, device=self._flags.device)
        torch.cuda.synchronize()
        # Both flags are zero before either rank can raise its peer's.
        MnnvlMemory.get_comm(mapping).barrier()

    def _flag(self, rank: int) -> torch.Tensor:
        return self._flags[rank, :4].view(torch.int32)

    def signal_wait(self) -> None:
        """On the current stream: release this rank's earlier work to the peer and wait for the
        peer's (graph capturable: the epoch lives on the device)."""
        _handshake_kernel[(1,)](
            self._epoch, self._flag(1 - self._ep_rank), self._flag(self._ep_rank), num_warps=1
        )


def pull_reduce(
    own_rows: torch.Tensor,
    peer_rows: torch.Tensor,
    target_ranks: torch.Tensor,
    send_indices: torch.Tensor,
    ep_rank: int,
    out: torch.Tensor,
    addend: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``out[t] = bf16(own + peer)`` of token ``t``'s two partials [``+ addend[t]``, rounded once].

    ``own_rows`` / ``peer_rows``: contiguous ``[rows, hidden]`` BF16 payload rows of this rank's
    tokens, indexed by send index; ``target_ranks`` / ``send_indices``: the dispatch's
    ``[tokens, top_k]`` int32 metadata.
    """
    num_tokens, hidden = out.shape
    if num_tokens == 0:
        return out
    if not all(t.is_contiguous() for t in (own_rows, peer_rows, out)):
        raise ValueError("pull_reduce needs contiguous payload rows and output")
    if addend is not None and (addend.shape != out.shape or not addend.is_contiguous()):
        raise ValueError("pull_reduce needs a contiguous addend shaped like the output")
    _pull_reduce_kernel[(num_tokens,)](
        own_rows,
        peer_rows,
        target_ranks,
        send_indices,
        addend if addend is not None else out,
        out,
        hidden,
        ep_rank,
        TOP_K=target_ranks.shape[1],
        BLOCK=1024,
        HAS_ADDEND=addend is not None,
        num_warps=4,
    )
    return out
