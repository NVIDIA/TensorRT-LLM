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

from . import m3_report

#: Verify rows per call, and exchange rows per call half of the workspace.
SHARD_ROWS = 48  # capacity: fixed: the kernel's built verify rows
# A verify step of more than SHARD_ROWS rows (13..48 requests at the built draft length) runs in chunks of whole
# requests, each chunk a call of the certified geometry on its own rows; greedy chunks are bitwise the gathered argmax
# (lowest index on ties). Rubin TP2 W1b 797141. Kill: False (past SHARD_ROWS the gathered logits + argmax run).
WIDE_SPEC_TAIL = True


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
    """Exchange slots ([2 halves][slot_rows rows][world] uint64) behind one multicast mapping over the TP group, the
    local key accumulators [slot_rows] and the (finished CTAs, call) counters. Collective over the TP group; call
    outside CUDA-graph capture, in the same order on every rank."""
    words = 2 * slot_rows * max(mapping.tp_size, 1)
    if mapping.tp_size > 1:
        from tensorrt_llm._torch.distributed.ops import (
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )
        from tensorrt_llm._utils import mpi_comm

        session = mpi_comm()
        session_group = session.Get_group()
        group = session_group.Incl(mapping.tp_group)
        session_group.Free()
        comm = session.Create_group(group)
        group.Free()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            comm.Free()
            raise RuntimeError("exchange workspace creation failed on another rank")
        handle = _make_mnnvl_mcast_buffer(
            comm, max(words * 8, 4096), mapping, mapping.is_multi_node()
        )
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int64, 0)
        mc = handle.get_mc_buffer((words,), torch.int64, 0)
        uc.zero_()
        torch.cuda.synchronize()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("exchange workspace creation failed on another rank")
        mc_ptr, rank = mc.data_ptr(), mapping.tp_rank
    else:
        uc = torch.zeros(words, dtype=torch.int64, device=device)
        mc_ptr, rank, handle, comm = uc.data_ptr(), 0, None, None
    return AcceptWorkspace(
        uc=uc,
        mc_ptr=mc_ptr,
        keys=torch.zeros(slot_rows, dtype=torch.int64, device=uc.device),
        counters=torch.zeros(2, dtype=torch.int32, device=uc.device),
        rank=rank,
        world_size=max(mapping.tp_size, 1),
        handle=handle,
        comm=comm,
    )


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
        or logits.dtype != torch.bfloat16
        or logits.stride(1) != 1
    ):
        return None
    if logits.shape[0] > SHARD_ROWS and not (WIDE_SPEC_TAIL and SHARD_ROWS // (draft_len + 1) >= 1):
        return None
    ws = _workspace(mapping, logits.device)
    if ws is None:
        return None
    dev = logits.device
    accepted = torch.empty((nreq, draft_len + 1), dtype=torch.int32, device=dev)
    num_accepted = torch.empty(nreq, dtype=torch.int32, device=dev)
    gather_ids = torch.empty(nreq, dtype=torch.int64, device=dev)
    if logits.shape[0] > SHARD_ROWS:
        # WIDE_SPEC_TAIL: whole requests per launch (SHARD_ROWS // (D + 1) of them), each chunk a call of the
        # certified geometry on its own rows; batch_idx is the identity layout (Eagle3 metadata), so a chunk passes
        # its local rows and its gather ids are shifted back to the step's verify layout.
        per = SHARD_ROWS // (draft_len + 1)
        local = _local_rows(per, dev)
        for c0 in range(0, nreq, per):
            c1 = min(nreq, c0 + per)
            r0, r1 = c0 * (draft_len + 1), c1 * (draft_len + 1)
            _launch(
                logits[r0:r1], vocab_offset, ws, draft_tokens[c0:c1], local[: c1 - c0], force_total,
                None if next_new is None else next_new[c0:c1],
                None if force_vec is None else force_vec[c0:c1],
                accepted[c0:c1], num_accepted[c0:c1], gather_ids[c0:c1],
            )  # fmt: skip
            gather_ids[c0:c1] += r0
        m3_report.taken(
            "wide_spec_tail", logits.shape[0], f"shard accept in {-(-nreq // per)} chunks"
        )
        return accepted, num_accepted, gather_ids
    _launch(
        logits, vocab_offset, ws, draft_tokens, batch_idx, force_total, next_new, force_vec,
        accepted, num_accepted, gather_ids,
    )  # fmt: skip
    return accepted, num_accepted, gather_ids


_LOCAL_ROWS: dict = {}


def _local_rows(n: int, device) -> torch.Tensor:
    """A cached int32 arange(n): a chunk's batch_idx in its own (identity) verify layout."""
    key = (n, device)
    t = _LOCAL_ROWS.get(key)
    if t is None:
        t = _LOCAL_ROWS[key] = torch.arange(n, dtype=torch.int32, device=device)
    return t


def _launch(
    logits, vocab_offset, ws, draft_tokens, batch_idx, force_total, next_new, force_vec,
    accepted, num_accepted, gather_ids,
) -> None:  # fmt: skip
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
