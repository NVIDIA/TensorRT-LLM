# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-latency TP all-reduce + residual + RMSNorm for decode (M <= 8 tokens) over a multicast mapping.

    partial = bf16(x + x2)                           (optional second input, e.g. shared + routed experts)
    updated = bf16(residual + sum_r partial_r)      (ranks summed in FP32, rank order)
    normed  = bf16(updated * rsqrt(mean(updated^2) + eps) * w)

One cluster of CLUSTER CTAs per token; CTA c owns columns [c * H / CLUSTER, ...), one 16-byte chunk (8 BF16)
per thread. Each CTA pushes its slice of this rank's partial through the multicast mapping (one store lands in
every rank's buffer), releases dependents, polls its own buffer until all ranks' slices arrived, sums them, adds
the residual and reduces the RMS statistic across the cluster through DSMEM.

Exchange protocol as in the K3 sandwich: int32 words, 0x80000000 = empty (pushed -0.0 halves become +0.0), each
CTA keeps a call counter whose parity selects the buffer half, readers re-arm the words they read. All ranks of
the TP group must issue the same calls in the same stream order.

The workspace holds two exchange sets. With prepushed 3 two producers pushed this rank's x (set A) and x2 (set B),
e.g. the routed experts' finalize and the shared expert's down projection, and the kernel forms bf16(x + x2) per
rank from the two slots, exactly as the unpushed path does.

Producers that push (the o_proj / shared-expert GEMV ``ar`` tuple, the routed-expert engine's ``arp`` descriptor) are
pointed at a bucket's exchange by its ``mc`` address, ``flags_for(8)`` counters, ``rank``, ``world_size`` and
``max_tokens``; ``get_workspace`` and ``_bucket`` select it.

The kernel is ``torch.ops.trtllm.minimax_m3_mc_allreduce_norm``, reached through the ``ops/comm`` entries
``m3_mc_allreduce_norm`` and ``m3_mc_finalize_allreduce_norm``; this module owns the workspaces, the bucket and
epilogue-option choice and the MXFP8 publication.
"""

from dataclasses import dataclass
from typing import Any

import torch

from ..ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm as _allreduce_norm_op
from ..ops.comm.m3_mc_finalize_allreduce_norm import (
    m3_mc_finalize_allreduce_norm as _finalize_allreduce_norm_op,
)

MAX_TOKENS = 48  # capacity: fixed: the decode kernels' row bound (decode steps)
BUCKETS = (8, 16, 48)  # token capacity of each workspace (grid = bucket * cluster CTAs)
EMPTY_WORD = -2147483648

# Attention / MoE boundary epilogue options (each bit-identical): 1 residual + norm weight loaded at entry (before
# the poll); 2 cluster RMS exchange by st.async + mbarrier complete_tx instead of cluster.sync; 4 call parity rotated
# right after the poll (no trailing __syncthreads); 8 max-smem carveout so the next kernel's CTAs co-reside early.
# The 8-token bucket, by token count 1..8 (GB300 TP4 tuning).
AR_OPTS_BY_M = (7, 7, 15, 15, 15, 15, 7, 7)
# The 16- and 48-token buckets.
AR_OPTS_BY_BUCKET = {16: 15, 48: 15}
# The MoE-finalize all-reduce, per bucket.
AR_FIN_OPTS_BY_BUCKET = {8: 7, 16: 15, 48: 15}


# Option 2 is also the safe ordering of the cluster RMS exchange: its stores into peer CTAs' shared memory follow a
# cluster barrier arrive / wait pair. Without it they run before any cluster barrier, when a peer need not have
# started, so every call keeps it.
_OPT_CLUSTER_BARRIER = 2


def _ar_opts(m: int, bucket: int, finalize: bool = False) -> int:
    """Epilogue options for an m-row call on the given bucket; every option is bit-identical. A bucket missing from
    the tables gets 0, and every result keeps _OPT_CLUSTER_BARRIER."""
    if finalize:
        opts = AR_FIN_OPTS_BY_BUCKET.get(bucket, 0)
    elif bucket == 8:
        opts = AR_OPTS_BY_M[m - 1]
    else:
        opts = AR_OPTS_BY_BUCKET.get(bucket, 0)
    return opts | _OPT_CLUSTER_BARRIER


@dataclass(eq=False)
class McAllReduceWorkspace:
    """Two halves of [max_tokens][world][H] BF16 (int32 words) per rank behind one multicast mapping, plus one call
    counter per CTA per cluster size. Create collectively and eagerly on every TP rank."""

    uc: torch.Tensor
    mc: torch.Tensor
    flags: dict
    rank: int
    world_size: int
    hidden: int
    handle: Any
    comm: Any
    max_tokens: int = 8

    @classmethod
    def create(cls, mapping, hidden: int, max_tokens: int = 8) -> "McAllReduceWorkspace":
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _make_mnnvl_mcast_buffer,
            _mnnvl_workspace_all_succeeded,
        )
        from tensorrt_llm._utils import mpi_disabled

        comm = _get_mnnvl_workspace_comm(mapping)
        words = 4 * max_tokens * mapping.tp_size * (hidden // 2)  # two exchange sets of two halves
        ok = not torch.cuda.is_current_stream_capturing()
        if not _mnnvl_workspace_all_succeeded(comm, ok):
            if not mpi_disabled():
                comm.Free()
            raise RuntimeError(
                "McAllReduceWorkspace.create must run outside CUDA-graph capture on every rank"
            )
        handle = _make_mnnvl_mcast_buffer(comm, words * 4, mapping, mapping.is_multi_node())
        uc = handle.get_uc_buffer(mapping.tp_rank, (words,), torch.int32, 0)
        mc = handle.get_mc_buffer((words,), torch.int32, 0)
        uc.fill_(EMPTY_WORD)
        torch.cuda.synchronize()
        if not _mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("McAllReduceWorkspace.create failed on another rank")
        return cls(
            uc=uc,
            mc=mc,
            flags={},
            rank=mapping.tp_rank,
            world_size=mapping.tp_size,
            hidden=hidden,
            handle=handle,
            comm=comm,
            max_tokens=max_tokens,
        )

    def flags_for(self, cluster: int):
        """This cluster size's per-CTA call counters, or None when they do not exist yet during a capture.
        get_workspace creates the cluster-8 counters with the workspace, so for cluster 8 (the only size built)
        that None never happens and cannot differ between ranks."""
        flags = self.flags.get(cluster)
        if flags is None:
            if torch.cuda.is_current_stream_capturing():
                return None
            flags = torch.zeros(self.max_tokens * cluster, dtype=torch.int32, device=self.uc.device)
            self.flags[cluster] = flags
        return flags


class McBuckets:
    """One workspace per token bucket (8 / 16 / 48): calls of different sizes never share exchange slots."""

    def __init__(self, workspaces):
        self.workspaces = workspaces
        self.hidden = workspaces[0].hidden

    def for_tokens(self, m: int):
        for ws in self.workspaces:
            if m <= ws.max_tokens:
                return ws
        return None


def _bucket(workspace, m):
    return workspace.for_tokens(m) if isinstance(workspace, McBuckets) else workspace


def decline_reason(x, residual, workspace, cluster=8, x2=None) -> str:
    """Why ``m3_mc_allreduce_norm`` returned None for these operands (for the dispatch's miss message)."""
    if workspace is None:
        return "no multicast workspace: it is created collectively on the first eager decode step"
    bucket = _bucket(workspace, x.shape[0])
    if bucket is None:
        return f"no workspace bucket holds {x.shape[0]} tokens (buckets {BUCKETS})"
    if bucket.flags.get(cluster) is None:
        return f"no {cluster}-CTA call counters for the {bucket.max_tokens}-token bucket"
    return (
        f"operands outside the kernel: x {tuple(x.shape)} {x.dtype} contiguous={x.is_contiguous()}, "
        f"residual contiguous={residual.is_contiguous()}, hidden {bucket.hidden}"
        + ("" if x2 is None else f", x2 {tuple(x2.shape)} contiguous={x2.is_contiguous()}")
    )


def m3_mc_allreduce_norm(
    x,
    residual,
    norm_weight,
    eps,
    workspace,
    use_pdl=False,
    cluster=8,
    x2=None,
    quant=False,
    prepushed=0,
):
    """Return (normed, updated_residual) or None when unsupported. With ``quant`` the normed row is also published,
    MXFP8-quantized, to m3_prequant.SLOT for the next MXFP8 GEMM.

    A None makes the caller take another collective, so it must be the same on every TP rank. It is: it depends only
    on the shapes, dtypes and layouts of the operands (the same code builds them on every rank) and on whether the
    workspace's flags exist, which ``get_workspace`` creates collectively, before any capture, for cluster 8."""
    workspace = _bucket(workspace, x.shape[0])
    if workspace is None:
        return None
    if (
        x.dim() != 2
        or x.shape[0] > MAX_TOKENS
        or x.dtype != torch.bfloat16
        or x.shape[1] != workspace.hidden
        or not x.is_contiguous()
        or not residual.is_contiguous()
        or (x2 is not None and (not x2.is_contiguous() or x2.shape != x.shape))
    ):
        return None
    flags = workspace.flags_for(cluster)
    if flags is None:
        return None
    outs = _allreduce_norm_op(
        x,
        x2,
        residual,
        norm_weight,
        float(eps),
        workspace.uc,
        workspace.mc.data_ptr(),
        flags,
        workspace.rank,
        workspace.world_size,
        cluster,
        use_pdl,
        workspace.max_tokens,
        bool(quant and x.shape[1] % 128 == 0),
        int(prepushed),
        _ar_opts(x.shape[0], workspace.max_tokens),
    )
    if len(outs) == 4:
        from .m3_prequant import SLOT

        SLOT.publish(outs[0], outs[2], outs[3])
    return outs[0], outs[1]


def m3_mc_finalize_allreduce_norm(
    fc2,
    perm,
    scales,
    shared,
    residual,
    norm_weight,
    eps,
    workspace,
    use_pdl=False,
    cluster=8,
    quant=False,
):
    """MoE finalize (sum_k scale * fc2[perm] + shared) + all-reduce + residual + RMSNorm in one kernel.
    Returns (normed, updated_residual) or None when unsupported. With ``quant`` the normed row is also published,
    MXFP8-quantized, to m3_prequant.SLOT for the next MXFP8 GEMM."""
    m = residual.shape[0]
    workspace = _bucket(workspace, m)
    if workspace is None:
        return None
    if (
        residual.dim() != 2
        or m > MAX_TOKENS
        or residual.shape[1] != workspace.hidden
        or fc2.dtype != torch.bfloat16
        or not fc2.is_contiguous()
        or fc2.dim() != 2
        or perm.dtype != torch.int32
        or perm.dim() != 2
        or perm.shape[0] != m
        or not perm.is_contiguous()
        or scales.dtype not in (torch.float32, torch.bfloat16)
        or not scales.is_contiguous()
        or (shared is not None and (shared.shape != residual.shape or not shared.is_contiguous()))
    ):
        return None
    flags = workspace.flags_for(cluster)
    if flags is None:
        return None
    outs = _finalize_allreduce_norm_op(
        fc2,
        perm,
        scales,
        shared,
        residual.contiguous(),
        norm_weight,
        float(eps),
        workspace.uc,
        workspace.mc.data_ptr(),
        flags,
        workspace.rank,
        workspace.world_size,
        cluster,
        use_pdl,
        workspace.max_tokens,
        bool(quant and residual.shape[1] % 128 == 0),
        _ar_opts(m, workspace.max_tokens, finalize=True),
    )
    if len(outs) == 4:
        from .m3_prequant import SLOT

        SLOT.publish(outs[0], outs[2], outs[3])
    return outs[0], outs[1]


_WORKSPACE: dict = {}
_AGREE_COMMS: dict = {}


def _tp_comm(mapping):
    """A communicator over mapping's TP group, created once (collective over that group only)."""
    key = tuple(mapping.tp_group)
    comm = _AGREE_COMMS.get(key)
    if comm is None:
        from tensorrt_llm._utils import mpi_comm, mpi_disabled

        if mpi_disabled():
            comm = mapping.tp_group_pg
        else:
            session = mpi_comm()
            session_group = session.Get_group()
            group = session_group.Incl(mapping.tp_group)
            session_group.Free()
            comm = session.Create_group(group)
            group.Free()
        _AGREE_COMMS[key] = comm
    return comm


def tp_agree(mapping, ok: bool) -> bool:
    """Whether ``ok`` holds on every rank of mapping's TP group (one blocking host collective).

    For a fast path whose availability is decided from rank-local state (a kernel build that can fail on one rank):
    every rank must take the same path, so that they issue the same collectives with the same exchange protocol and
    the same kernels. Call it once per decision, at build time, at the same point on every rank, outside CUDA-graph
    capture; never per step. With one rank, or attention DP (whose ranks run different batches), it returns ``ok``.
    """
    if mapping.tp_size <= 1 or mapping.enable_attention_dp:
        return ok
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("tp_agree must run outside CUDA-graph capture")
    from tensorrt_llm._torch.distributed.ops import _mnnvl_workspace_all_succeeded

    return _mnnvl_workspace_all_succeeded(_tp_comm(mapping), bool(ok))


def tp_agree_built(mapping, built, what: str):
    """``built`` when it is not None on every TP rank, else None on every rank (see ``tp_agree``)."""
    if tp_agree(mapping, built is not None) or built is None:
        return built
    from tensorrt_llm.logger import logger

    logger.warning(f"{what}: built here but not on another TP rank; every rank falls back")
    return None


def get_workspace(mapping, hidden: int):
    """Process-wide workspace (created collectively on first eager use); None while capturing before creation.

    The first eager call for a hidden size (the warmup forward, before any CUDA-graph capture) creates one
    McAllReduceWorkspace per bucket in BUCKETS on every TP rank (MPI collective over the TP group) plus each
    bucket's 8-CTA-cluster call counters; later calls, captured or not, return the same McBuckets. The workspaces
    live until process exit.
    """
    ws = _WORKSPACE.get(hidden)
    if ws is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        wss = [McAllReduceWorkspace.create(mapping, hidden, b) for b in BUCKETS]
        for w in wss:
            w.flags_for(8)
        ws = McBuckets(wss)
        _WORKSPACE[hidden] = ws
    return ws
