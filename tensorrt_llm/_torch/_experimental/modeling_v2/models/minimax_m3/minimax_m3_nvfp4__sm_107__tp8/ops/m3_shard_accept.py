# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused greedy Eagle3 verification for the MiniMax-M3 target head (vocab-sharded TP head, verify rows <= 48).

Replaces the all-gather of the target logits, the argmax and the strict-acceptance ops of a greedy verify step with
one kernel over this rank's BF16 logits shard (the target lm_head's own local GEMM output):

* Every logit becomes a 64-bit key ``ordered(value) << 32 | ~global_index`` (ties pick the lowest index, as
  ``torch.argmax``), max-reduced per row: warp shuffles, then ``atomicMax``.
* The last CTA to finish pushes the rank's per-row key into every rank's exchange slot through a multicast mapping
  (Lamport: 0 = empty, the key is never 0), polls its own slots for all ranks, takes the global argmax, re-arms the
  slots and runs the strict acceptance. Call n uses slot half n % 2.

The argmax is taken over the BF16 logits without arithmetic, so the tokens are those of the in-tree path's argmax
over the gathered logits. The kernel is compiled into TensorRT-LLM (cpp/tensorrt_llm/kernels/minimaxM3/
shardAccept.cu) and called as ``torch.ops.trtllm.minimax_m3_shard_accept``.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.spec.accept_workspace import (
    create_accept_workspace,
)

from . import m3_report

#: Verify rows per call, and exchange rows per call half of the workspace.
SHARD_ROWS = 48  # capacity: fixed: the kernel's built verify rows


@dataclass(eq=False)
class AcceptWorkspace:
    """Per-rank exchange slots ([2 halves][SHARD_ROWS rows][world] uint64) behind one multicast mapping, the local
    key accumulators and the (finished CTAs, call) counters."""

    uc: torch.Tensor
    mc_ptr: int
    keys: torch.Tensor
    counters: torch.Tensor
    rank: int
    world_size: int
    handle: Any = None
    comm: Any = None


_WS: Optional[AcceptWorkspace] = None
_NUM_SMS: Optional[int] = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _NUM_SMS


def create_workspace(mapping, device, slot_rows: int) -> AcceptWorkspace:
    return create_accept_workspace(mapping, device, slot_rows, AcceptWorkspace)


def _workspace(mapping, device) -> Optional[AcceptWorkspace]:
    """The exchange workspace, created collectively outside CUDA-graph capture on first use."""
    global _WS
    if _WS is not None:
        return _WS
    if torch.cuda.is_current_stream_capturing():
        m3_report.capture_missed("m3_shard_accept", None, "the shard-accept workspace")
        return None
    _WS = create_workspace(mapping, device, SHARD_ROWS)
    return _WS


def shard_accept(
    logits: torch.Tensor,
    vocab_offset: int,
    mapping,
    draft_tokens: torch.Tensor,
    batch_idx: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor] = None,
    force_vec: Optional[torch.Tensor] = None,
):
    """Greedy verification from this rank's BF16 logits shard.

    Args:
        logits: [nreq * (draft_len + 1) <= SHARD_ROWS, V / tp] BF16, unit column stride: this rank's columns
            (global ids ``vocab_offset ..``) of the verify rows, request-major.
        draft_tokens: [nreq, draft_len] int32, the drafts being verified.
        batch_idx: [nreq] int32, each request's row in the verify layout.
        force_total / force_vec: forced acceptance (see ``m3_shard_accept``).
        next_new: optional int32 [nreq, >= 1]; column 0 receives the last accepted token.

    Returns:
        (accepted_tokens [nreq, draft_len + 1] int32, num_accepted [nreq] int32, first-pass gather ids [nreq]
        int64), or None when the shape is not served or the workspace is missing during a capture. Collective over
        the TP group.
    """
    nreq, draft_len = draft_tokens.shape
    if (
        logits.dim() != 2
        or logits.shape[0] != nreq * (draft_len + 1)
        or logits.shape[0] > SHARD_ROWS
        or logits.dtype != torch.bfloat16
        or logits.stride(1) != 1
    ):
        return None
    ws = _workspace(mapping, logits.device)
    if ws is None:
        return None
    dev = logits.device
    accepted = torch.empty((nreq, draft_len + 1), dtype=torch.int32, device=dev)
    num_accepted = torch.empty(nreq, dtype=torch.int32, device=dev)
    gather_ids = torch.empty(nreq, dtype=torch.int64, device=dev)
    torch.ops.trtllm.minimax_m3_shard_accept(
        logits,
        vocab_offset,
        ws.keys,
        ws.counters,
        ws.uc,
        ws.mc_ptr,
        ws.rank,
        ws.world_size,
        draft_tokens.contiguous(),
        batch_idx,
        accepted,
        num_accepted,
        gather_ids,
        force_total,
        next_new,
        _num_sms(),
        force_vec,
    )
    return accepted, num_accepted, gather_ids
