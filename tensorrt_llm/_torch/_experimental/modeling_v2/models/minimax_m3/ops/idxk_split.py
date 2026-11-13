# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IDXK_SPLIT (family level): the MSA index-K pages of a TP8 decode step split over the 8 ranks, with an exact
cross-rank top-16 merge (the idea of vLLM's ``indexer_nvfp4_decode_cp.py``, as CUDA C++ kernels of the family library:
``csrc/m31/idxkSplit.cu``).

M3.1 has 4 index heads; at TP8 rank ``r`` holds head ``r // 2`` and scores every index-K page of every request for it.
Here rank ``r`` scores only the blocks ``p`` with ``p % 8 == r``, for all 4 heads, in ONE pass of the target's existing
index scorer (4 heads x 8 query slots fill its 32-column tile; each score is computed exactly as at 1 head):

1. ``plan`` (once per step, plain launch): the compact block table (compact block ``j`` = global block ``j * 8 + r``),
   the length handed to the scorer (``n_local * 128 + qlen``: no causal mask on any owned block) and the step epoch;
2. ``forward`` (per sparse layer): index-q exchange (ranks 0/2/4/6 publish their head), the scorer over the compact
   table into [4, W, rows], this rank's exact top 16 per (row, head) published to every rank, and the merge of the
   8 x 16 candidates of this rank's head into the FSEL table (``m3_fsel_topk``: ascending block ids, -1 tail).

The table equals the single-rank FSEL / long-select table bit for bit: the order (score desc on the fp32 bit order,
block asc) is total, the global top ``need`` (<= 16) lies in the union of the per-rank top 16s, and every candidate
block ends before its query's own block (``local_blocks >= 1``), so the unmasked compact scores equal the masked
single-rank scores on every candidate. The scorer must run WITHOUT its early issue / pre-wait K staging here
(``early=False``): with the compact length every owned block looks older than the step, and a pre-wait K tile of the
block holding this step's first draft tokens would be stale.

The exchange is Lamport-tagged 64-bit words through the TP multicast mapping (no flags; halves alternate by epoch
parity; the merge re-arms each consumed slot with a tag of the other parity, ``init_local``), so the path is CUDA-graph
safe. Targets keep the switch (``TRTLLM_M31_DISABLE_IDXK_SPLIT``), the row /
context scope and the engagement line; this module only adapts the calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops

ITEM_ID = "IDXK_SPLIT"
WORLD = 8
HEADS = 4
HEAD_DIM = 128
PAGE = 128
TOPK = 16
QWORDS = HEAD_DIM // 4
#: The select contract's blocks per row (kFselTopkMaxBlocks) and the compact blocks per rank at that length.
MAX_BLOCKS = 9216
MAX_LOCAL = (MAX_BLOCKS + WORLD - 1) // WORLD


def layer_words(rows_max: int) -> int:
    """64-bit words of one layer's exchange workspace (idxkSplit.h ``layerWords``)."""
    return 2 * HEADS * rows_max * QWORDS + 2 * WORLD * HEADS * rows_max * TOPK


def init_local(local: torch.Tensor, rows_max: int, num_layers: int) -> None:
    """The workspace's start state: Q words 0 (epoch tag 0, never live: the first plan makes the epoch 1) and the C
    words of half h at a tag of the other parity (half 0 -> 1, half 1 -> 0). Half h only ever carries epochs of parity
    h and the merge re-arms every consumed slot the same way, so no slot matches before its producer writes it,
    however long its row stays idle (the C tag is 18 bits)."""
    words = layer_words(rows_max)
    qw = 2 * HEADS * rows_max * QWORDS
    half = WORLD * HEADS * rows_max * TOPK
    view = local[: num_layers * words].view(num_layers, words)
    view.zero_()
    view[:, qw : qw + half] = 1 << 14


def compact_width(num_cols: int) -> int:
    """Compact blocks per request row: every owned block of a ``num_cols``-block row plus the scored invented one."""
    return (min(num_cols, MAX_BLOCKS) + WORLD - 1) // WORLD + 1


def ops_available() -> bool:
    """Whether this build (or a loaded op library) carries the four ``minimax_m3_sm107tp8_idxk_*`` ops."""
    ns = family_ops()
    return (
        hasattr(ns, "minimax_m3_sm107tp8_idxk_compact")
        and hasattr(ns, "minimax_m3_sm107tp8_idxk_qx")
        and hasattr(ns, "minimax_m3_sm107tp8_idxk_topk")
        and hasattr(ns, "minimax_m3_sm107tp8_idxk_merge")
    )


@dataclass(eq=False)
class IdxkWorkspace:
    """One rank's exchange workspace: ``local`` [num_layers * layer_words] int64 (this rank's slots, unicast), the
    multicast base (0 without one) or the 8 ranks' unicast bases, and the step epoch (int32 [1], +1 per ``plan``)."""

    local: torch.Tensor
    rank: int
    rows_max: int
    num_layers: int
    epoch: torch.Tensor
    mc_base: int = 0
    peer_bases: Optional[List[int]] = None
    keep: List[Any] = field(default_factory=list)

    def layer(self, i: int):
        """(local slice, multicast address or 0, [8] CPU int64 unicast addresses) of layer ``i``."""
        words = layer_words(self.rows_max)
        local = self.local[i * words : (i + 1) * words]
        off = i * words * 8
        mc = self.mc_base + off if self.mc_base else 0
        if self.peer_bases is not None:
            dst = torch.tensor([b + off for b in self.peer_bases], dtype=torch.int64)
        else:
            dst = torch.zeros(0, dtype=torch.int64)
        return local, mc, dst


def create_workspace_mnnvl(mapping, rows_max: int, num_layers: int) -> IdxkWorkspace:
    """The TP8 exchange workspace behind one multicast mapping over the TP group (as ``m3_shard_accept``). Collective
    over the TP group: call outside CUDA-graph capture, in the same order on every rank."""
    from tensorrt_llm._torch.distributed.ops import (
        _make_mnnvl_mcast_buffer,
        _mnnvl_workspace_all_succeeded,
    )
    from tensorrt_llm._utils import mpi_comm

    if mapping.tp_size != WORLD:
        raise ValueError(f"IDXK_SPLIT needs tp_size {WORLD}, got {mapping.tp_size}")
    words = num_layers * layer_words(rows_max)
    session = mpi_comm()
    session_group = session.Get_group()
    group = session_group.Incl(mapping.tp_group)
    session_group.Free()
    comm = session.Create_group(group)
    group.Free()
    if not _mnnvl_workspace_all_succeeded(comm, True):
        comm.Free()
        raise RuntimeError("IDXK_SPLIT workspace creation failed on another rank")
    handle = _make_mnnvl_mcast_buffer(comm, max(words * 8, 4096), mapping, mapping.is_multi_node())
    uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int64, 0)
    mc = handle.get_mc_buffer((words,), torch.int64, 0)
    init_local(uc, rows_max, num_layers)
    torch.cuda.synchronize()
    if not _mnnvl_workspace_all_succeeded(comm, True):
        raise RuntimeError("IDXK_SPLIT workspace creation failed on another rank")
    return IdxkWorkspace(
        local=uc,
        rank=mapping.tp_rank,
        rows_max=rows_max,
        num_layers=num_layers,
        epoch=torch.zeros(1, dtype=torch.int32, device=uc.device),
        mc_base=mc.data_ptr(),
        keep=[handle, comm, mc],
    )


def create_workspaces_emulated(rows_max: int, num_layers: int, device) -> List[IdxkWorkspace]:
    """Eight emulated ranks on one GPU (the oracle): each rank's stores go to all eight local buffers (unicast)."""
    words = num_layers * layer_words(rows_max)
    bufs = [torch.empty(words, dtype=torch.int64, device=device) for _ in range(WORLD)]
    for b in bufs:
        init_local(b, rows_max, num_layers)
    bases = [b.data_ptr() for b in bufs]
    return [
        IdxkWorkspace(
            local=bufs[r],
            rank=r,
            rows_max=rows_max,
            num_layers=num_layers,
            epoch=torch.zeros(1, dtype=torch.int32, device=device),
            peer_bases=bases,
            keep=[bufs],
        )
        for r in range(WORLD)
    ]


@dataclass(eq=False)
class IdxkPlan:
    """Per-step buffers of one rank at a fixed (requests, query length, score width)."""

    cbt: torch.Tensor  # [B, W] int32 compact block table
    score_lens: torch.Tensor  # [B] int32 lengths the scorer sees
    q_tile: torch.Tensor  # [rows, 4, 128] uint8 (FP8 bytes) gathered index-q
    scores: torch.Tensor  # [4, W, rows] fp32 (the scorer writes a transposed view)
    requests: int
    qlen: int
    num_cols: int

    @property
    def rows(self) -> int:
        return self.requests * self.qlen


def make_plan(requests: int, qlen: int, num_cols: int, device) -> IdxkPlan:
    """Allocate a plan (outside capture)."""
    width = compact_width(num_cols)
    rows = requests * qlen
    return IdxkPlan(
        cbt=torch.zeros(requests, width, dtype=torch.int32, device=device),
        score_lens=torch.zeros(requests, dtype=torch.int32, device=device),
        q_tile=torch.zeros(rows, HEADS, HEAD_DIM, dtype=torch.uint8, device=device),
        scores=torch.zeros(HEADS, width, rows, dtype=torch.float32, device=device),
        requests=requests,
        qlen=qlen,
        num_cols=num_cols,
    )


def plan_step(
    ws: IdxkWorkspace, plan: IdxkPlan, block_table: torch.Tensor, seq_lens: torch.Tensor
) -> None:
    """Step setup on the current stream (a plain launch, before the step's first sparse layer)."""
    family_ops().minimax_m3_sm107tp8_idxk_compact(
        block_table[: plan.requests],
        seq_lens[: plan.requests],
        plan.cbt,
        plan.score_lens,
        ws.epoch,
        ws.rank,
        plan.qlen,
    )


def exchange_q(
    ws, plan, layer: int, idx_q: torch.Tensor, publish=True, gather=True, use_pdl=False
) -> None:
    local, mc, dst = ws.layer(layer)
    family_ops().minimax_m3_sm107tp8_idxk_qx(
        idx_q.view(torch.uint8) if idx_q.dtype != torch.uint8 else idx_q,
        local,
        mc,
        dst,
        plan.q_tile,
        ws.epoch,
        ws.rank,
        ws.rows_max,
        publish,
        gather,
        use_pdl,
    )


def score(
    plan: IdxkPlan, idx_k_cache: torch.Tensor, scorer: Callable, q_dtype=torch.float8_e4m3fn
) -> None:
    """``scorer(q [rows, 4, 128], index_k, block_table, seq_lens, score [4, rows, W])``: the target's index scorer at
    4 heads with ``early=False`` (module docstring)."""
    scorer(
        plan.q_tile.view(q_dtype),
        idx_k_cache,
        plan.cbt,
        plan.score_lens,
        plan.scores.transpose(1, 2),
    )


def local_topk(
    ws, plan, layer: int, n_valid: torch.Tensor, init_blocks: int, local_blocks: int, use_pdl=False
):
    _, mc, dst = ws.layer(layer)
    family_ops().minimax_m3_sm107tp8_idxk_topk(
        plan.scores.transpose(1, 2)[:, : plan.rows],
        n_valid,
        mc,
        dst,
        ws.epoch,
        ws.rank,
        ws.rows_max,
        plan.num_cols,
        init_blocks,
        local_blocks,
        use_pdl,
    )


def merge(
    ws,
    plan,
    layer: int,
    n_valid,
    out,
    init_blocks: int,
    local_blocks: int,
    use_pdl=False,
    trigger_phase=2,
) -> torch.Tensor:
    """The FSEL table of this rank's head into ``out`` [rows, 1, 16] int32."""
    local, _, _ = ws.layer(layer)
    family_ops().minimax_m3_sm107tp8_idxk_merge(
        local,
        n_valid,
        out,
        ws.epoch,
        ws.rank // 2,
        ws.rows_max,
        plan.num_cols,
        init_blocks,
        local_blocks,
        use_pdl,
        trigger_phase,
    )
    return out


def forward(
    ws: IdxkWorkspace,
    plan: IdxkPlan,
    layer: int,
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    n_valid: torch.Tensor,
    out: torch.Tensor,
    scorer: Callable,
    *,
    init_blocks: int = 0,
    local_blocks: int = 1,
    use_pdl: bool = False,
) -> torch.Tensor:
    """One sparse layer on one real rank (all four steps; collective over the TP group)."""
    exchange_q(ws, plan, layer, idx_q, use_pdl=use_pdl)
    score(
        plan,
        idx_k_cache,
        scorer,
        idx_q.dtype if idx_q.dtype != torch.uint8 else torch.float8_e4m3fn,
    )
    local_topk(ws, plan, layer, n_valid, init_blocks, local_blocks, use_pdl=use_pdl)
    return merge(ws, plan, layer, n_valid, out, init_blocks, local_blocks, use_pdl=use_pdl)


def reference_table(
    scores_kv: torch.Tensor, n_valid: torch.Tensor, init_blocks: int, local_blocks: int
) -> torch.Tensor:
    """The FSEL table contract in torch: scores_kv [1, num_cols, rows] fp32 (one KV head) -> [rows, 1, 16] int32."""
    _, num_cols, rows = scores_kv.shape
    out = torch.full((rows, 1, TOPK), -1, dtype=torch.int32)
    s = scores_kv[0].t().contiguous().cpu()
    bits = s.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    okey = torch.where(bits >= 0x80000000, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    for t, nraw in enumerate(n_valid[:rows].tolist()):
        nvv = min(nraw, num_cols)
        if nvv <= TOPK:
            out[t, 0, :nvv] = torch.arange(nvv, dtype=torch.int32)
            continue
        cap = min(nvv, MAX_BLOCKS)
        fend = nraw - local_blocks
        flo = min(init_blocks, cap)
        fhi = max(0, cap - max(fend, flo))
        need = TOPK - flo - fhi
        if need <= 0:
            out[t, 0] = torch.tensor(
                [k if k < flo else cap - fhi + (k - flo) for k in range(TOPK)], dtype=torch.int32
            )
            continue
        lo, hi = init_blocks, min(cap, fend)
        p = torch.arange(lo, hi)
        key = (okey[t, lo:hi] << 16) | (0xFFFF - p)
        pick = p[torch.topk(key, need).indices].sort().values
        row = list(range(flo)) + pick.tolist() + list(range(cap - fhi, cap))
        out[t, 0] = torch.tensor(row, dtype=torch.int32)
    return out
