# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused sampled verification for the MiniMax-M3 Eagle3 target head (TP-sharded vocabulary, decode rows <= 48).

Replaces, per decode step, the full-vocabulary all-gather of the target logits, the fused sampling kernel and the
strict-acceptance ops for rows sampled with temperature and top-p (the AgentX contract). Rows with top-k or min-p
active must stay on the generic path; the caller decides.

Sampling is top-p rejection sampling, distributed over the TP ranks. With w = exp(logit / T - max) over the FULL
vocabulary and Z = sum(w), a token is in the nucleus iff the mass strictly above its weight is at most top_p * Z
(the fused sampling kernel's kept set, ties included). A candidate is drawn proportionally to w and accepted iff it
is in the nucleus; a rejected candidate lies below the nucleus, so the next candidate is drawn over the tokens heavier
than it. The accepted token is distributed as w restricted to the nucleus. The first candidate lands in the nucleus
with probability >= top_p, so most rows decide on their second exchange.

Per verify row, 8 CTAs (rows <= 32) or 4 CTAs (more rows) on every rank each own a slice of the rank's logits shard,
staged in shared memory once. A candidate is drawn in two stages without per-token noise: each CTA draws a local
candidate by inverse CDF over its slice, and the CTA whose log-mass plus Gumbel noise is largest supplies the
candidate (CTA j wins with probability T_j / sum T, its token with w / T_j). Uniforms come from a counter-based Philox
keyed by the request's seed and offset, the row, the attempt and the CTA. Each round every CTA multicasts one 4-word
record (key, candidate logit, and max and mass in round 0 or the mass above the current candidate later) and reads
the records of all CTAs of the row on all ranks. Every CTA takes the same decision from the same records in the same
order, so all ranks agree without any intra-GPU handshake. Words carry the call's tag: a slot needs no clearing and
two call halves keep a rank one call ahead from overwriting a slot still being read.
"""

from typing import Optional

import torch

from . import m3_report

# The kernel is compiled into TensorRT-LLM (cpp/tensorrt_llm/kernels/minimaxM3/sampledAccept.cu) and called as
# torch.ops.trtllm.minimax_m3_sampled_accept.
MAX_ROWS = 48  # capacity: fixed: the kernel's built rows (decode steps)
# The TP2 vocab shard limit of trtllm::minimax_m3_sm107tp2_sampled_accept (kMaxVs): 200064 / 2 = 100032 columns.
MAX_VS = 100352  # capacity: fixed: the vocabulary shard bound at tp2


class SampledAcceptWorkspace:
    """Exchange slots behind one multicast mapping (two call halves), the (finished rows, call) counters, the per-row
    token scratch and an extra-attempt counter. Created collectively, outside CUDA-graph capture."""

    def __init__(self, uc, mc_ptr, rank, world, handle=None, comm=None):
        self.uc, self.mc_ptr, self.rank, self.world = uc, mc_ptr, rank, world
        self.handle, self.comm = handle, comm
        dev = uc.device
        self.counters = torch.zeros(2, dtype=torch.int32, device=dev)
        self.tokens = torch.zeros(MAX_ROWS, dtype=torch.int32, device=dev)
        self.extra_attempts = torch.zeros(1, dtype=torch.int32, device=dev)

    @classmethod
    def create(cls, mapping, device) -> "SampledAcceptWorkspace":
        words = int(
            torch.ops.trtllm.minimax_m3_sm107tp2_sampled_accept_ws_words(max(mapping.tp_size, 1))
        )
        if mapping.tp_size == 1:
            uc = torch.zeros(words, dtype=torch.int64, device=device)
            return cls(uc, uc.data_ptr(), 0, 1)
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
        ok = not torch.cuda.is_current_stream_capturing()
        if not _mnnvl_workspace_all_succeeded(comm, ok):
            comm.Free()
            raise RuntimeError(
                "SampledAcceptWorkspace.create must run outside CUDA-graph capture on every rank"
            )
        handle = _make_mnnvl_mcast_buffer(comm, words * 8, mapping, mapping.is_multi_node())
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int64, 0)
        mc = handle.get_mc_buffer((words,), torch.int64, 0)
        uc.zero_()
        torch.cuda.synchronize()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("SampledAcceptWorkspace.create failed on another rank")
        return cls(uc, mc.data_ptr(), mapping.tp_rank, mapping.tp_size, handle, comm)


_WS: Optional[SampledAcceptWorkspace] = None


def workspace(mapping, device) -> Optional[SampledAcceptWorkspace]:
    global _WS
    if _WS is None:
        if torch.cuda.is_current_stream_capturing():
            m3_report.capture_missed("spec_tail", None, "the sampled-accept workspace")
            return None
        _WS = SampledAcceptWorkspace.create(mapping, device)
    return _WS


def sampled_accept(
    logits: torch.Tensor,
    vocab_offset: int,
    mapping,
    draft_tokens: torch.Tensor,
    batch_idx: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: Optional[torch.Tensor],
    seeds: torch.Tensor,
    offsets: torch.Tensor,
    force_total: int,
    next_new: Optional[torch.Tensor] = None,
    force_vec: Optional[torch.Tensor] = None,
    ws: Optional[SampledAcceptWorkspace] = None,
    use_pdl: bool = True,
):
    """Sampled verification (temperature + top-p) from this rank's bf16 logits shard [nreq * (draft_len + 1), V / tp]:
    outputs as shard_accept (accepted [nreq, D+1], num_accepted [nreq], gather_ids [nreq]); None
    when unavailable. Per-row sampling parameters and Philox (seed, offset) are the spec metadata's, one per row.
    top_ks / min_ps are not applied: rows with top-k or min-p active must be routed to the generic path by the
    caller."""
    del top_ks, min_ps
    nreq, draft_len = draft_tokens.shape
    rows = nreq * (draft_len + 1)
    if logits.dim() != 2 or logits.shape[0] != rows or rows > MAX_ROWS or logits.shape[1] > MAX_VS:
        return None
    if logits.dtype != torch.bfloat16 or logits.stride(1) != 1:
        return None
    ws = ws if ws is not None else workspace(mapping, logits.device)
    if ws is None:
        return None
    dev = logits.device
    accepted = torch.empty((nreq, draft_len + 1), dtype=torch.int32, device=dev)
    num_accepted = torch.empty(nreq, dtype=torch.int32, device=dev)
    gather_ids = torch.empty(nreq, dtype=torch.int64, device=dev)
    temperatures = temperatures[:rows].float().contiguous()
    top_ps = top_ps[:rows].float().contiguous()
    seeds = seeds[:rows].long().contiguous()
    offsets = offsets[:rows].long().contiguous()
    _launch(
        logits, vocab_offset, temperatures, top_ps, seeds, offsets, ws, draft_tokens, batch_idx,
        accepted, num_accepted, gather_ids, force_total, next_new, force_vec, use_pdl, rows,
    )  # fmt: skip
    return accepted, num_accepted, gather_ids


def _launch(
    logits, vocab_offset, temperatures, top_ps, seeds, offsets, ws, draft_tokens, batch_idx,
    accepted, num_accepted, gather_ids, force_total, next_new, force_vec, use_pdl, rows,
) -> None:  # fmt: skip
    torch.ops.trtllm.minimax_m3_sm107tp2_sampled_accept(
        logits,
        int(vocab_offset),
        temperatures,
        top_ps,
        seeds,
        offsets,
        ws.uc,
        ws.mc_ptr,
        ws.counters,
        ws.tokens,
        ws.rank,
        ws.world,
        draft_tokens.int().contiguous(),
        batch_idx.int().contiguous(),
        accepted,
        num_accepted,
        gather_ids,
        int(force_total),
        next_new,
        force_vec,
        ws.extra_attempts,
        bool(use_pdl),
        PREWAIT_PARAMS,
        EARLY_TRIGGER,
        _ilp_walk(rows),
    )


# The sampled verify loads each row's sampling parameters before its PDL dependency wait (overlapping the head GEMM's
# tail; the same values and tokens), and releases its dependents right after that wait (the dependent parks at its
# own wait; the same values).
PREWAIT_PARAMS = True
EARLY_TRIGGER = True
# SA-PW: the owner walk takes its tokens 8 at a time (the same draw, tokens and Philox draws as the one-token walk,
# bit for bit; -2.2..-2.5 us per call on Rubin TP2). Off in the TP2 deployment (frozen deploy value).
ILP_WALK = False
ILP_WALK_PATH = "m3_sampled_accept_ilp_walk"


def _ilp_walk(rows: int) -> bool:
    """ILP_WALK, logged once per row count as taken; off, one 'not taken' line (logged once)."""
    if not ILP_WALK:
        m3_report.uncovered(ILP_WALK_PATH, "the one-token owner walk (ILP_WALK off)")
        return False
    m3_report.taken(
        ILP_WALK_PATH,
        rows,
        "the sampled verify's owner walk takes 8 tokens at a time",
    )
    return True
