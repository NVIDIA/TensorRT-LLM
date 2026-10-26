# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused DSpark Markov block chain for a vocabulary-sharded draft head (one launch per block).

``markov_chain_fused`` runs ``DSparkMarkovChainKernel`` (``cute_dsl_kernels/blackwell/dspark``):
the ``K`` sequential Markov positions of every row, each a ``W2`` shard GEMV, a local argmax and
a cross-rank candidate exchange, in one persistent launch. The exchange runs on a one-shot
Lamport workspace of its own (``MarkovWorkspace``), so it never shares the triple-buffer phase of
the model's all-reduces; within one node its buffers are IPC-mapped, across the nodes of a
multi-node NVLink domain fabric-mapped.

The exchange is two-level: per position every CTA stores its candidate as one launch-tagged 8-byte
entry in a local slot, CTA 0 reduces those and stores the rank's winner into every rank's buffer,
and every CTA reduces the ranks' winners (NaN first, larger value, smaller id). The tags (the
kernel's launch epoch + 1) make an entry of an earlier launch never pass for a current one.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div, pad_up

from ..cute_dsl_utils import graph_safe_from_dlpack

# The exchange's entries hold an 18-bit vocabulary id (the vocabulary size marks "no
# candidate").
_MAX_VOCAB = (1 << 18) - 1
MAX_ROWS = 8
MAX_POSITIONS = 8
# One launch writes one 16-byte candidate per (position, row, source rank, CTA) into every rank's
# slot; CTAs = SM count.
_SLOT_VECTOR_BYTES = 16


class MarkovWorkspace:
    """A dedicated one-shot Lamport workspace of a TP group (``[buffers..., flag, layout]``).

    Each of its three slots holds the largest launch: ``MAX_POSITIONS x MAX_ROWS x nranks x SMs``
    candidates. ``slot_bytes`` is read back from the workspace's layout word. The Lamport buffers
    are IPC-mapped device memory (a TP group within one node), or with ``fabric`` fabric-mapped
    memory of a multi-node NVLink domain (``_fabric_lamport_table``); the table and the exchange
    protocol are the same.
    """

    def __init__(self, mapping, fabric: bool = False) -> None:
        nranks = mapping.tp_size
        sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        need = MAX_POSITIONS * MAX_ROWS * nranks * sms * _SLOT_VECTOR_BYTES
        if fabric:
            self._buffers, self.table, self.slot_bytes = _fabric_lamport_table(mapping, need)
        else:
            from ..distributed.allreduce_helper import CustomAllReduceHelper

            self._buffers, workspace = CustomAllReduceHelper.allocate_allreduce_fusion_workspace(
                mapping, ceil_div(need, nranks)
            )
            # layout words: [clear size of the next launch, bytes per slot (the triple-buffer
            # stride)]
            self.slot_bytes = int(self._buffers[4][1].item())
            self.table = workspace[2 * nranks :]
            # Entries: zero (tag 0) everywhere. The -0.0 Lamport fill would read as tag 0x2000 in
            # every entry.
            import cuda.bindings.driver as cuda

            lamport = self._buffers[2]
            (err,) = cuda.cuMemsetD8(lamport.local_ptr, 0, 3 * self.slot_bytes)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"DSpark Markov workspace clear failed: {err}")
            torch.cuda.synchronize()
        if self.slot_bytes < need:
            raise RuntimeError(
                f"DSpark Markov workspace slot holds {self.slot_bytes} bytes, {need} are needed"
            )
        self.rank = mapping.tp_rank
        self.nranks = nranks


def _fabric_lamport_table(mapping, slot_bytes: int):
    """Triple-buffered Lamport buffers of ``slot_bytes`` per slot for every rank of ``mapping``'s
    TP group in fabric-mapped (MNNVL) memory, as an all-reduce fusion workspace's
    ``[buffers..., flag, layout]`` table.

    Collective over the TP group (fabric handle exchange); every rank must call it at the same
    point. Each rank zeroes its own buffer (launch-tagged entries) before the closing collective,
    so no rank writes into a peer's buffer before the peer has cleared it. Returns
    ``(keepalive, table, slot_bytes)``; raises on every rank if any rank failed.
    """
    from ..distributed.ops import (
        _get_mnnvl_workspace_comm,
        _make_mnnvl_mcast_buffer,
        _mnnvl_device_index,
        _mnnvl_workspace_all_succeeded,
    )

    slot = pad_up(slot_bytes, 128)
    words = 3 * slot // 4
    comm = _get_mnnvl_workspace_comm(mapping)
    keepalive, table, error = None, None, None
    try:
        handle = _make_mnnvl_mcast_buffer(comm, 3 * slot, mapping, True)
        buffers = [
            handle.get_uc_buffer(r, (words,), torch.float32, 0) for r in range(mapping.tp_size)
        ]
        device = torch.device("cuda", _mnnvl_device_index(mapping))
        buffers[mapping.tp_rank].zero_()
        flag = torch.zeros(3, dtype=torch.int32, device=device)
        layout = torch.tensor([0, slot], dtype=torch.int64, device=device)
        table = torch.tensor(
            [b.data_ptr() for b in buffers] + [flag.data_ptr(), layout.data_ptr()],
            dtype=torch.int64,
            device=device,
        )
        torch.cuda.synchronize()
        keepalive = (comm, handle, buffers, flag, layout)
    except Exception as e:  # converged below: every rank must agree before any rank goes on
        error = e
    if not _mnnvl_workspace_all_succeeded(comm, error is None):
        raise RuntimeError("DSpark Markov fabric workspace failed on a TP rank") from error
    return keepalive, table, slot


def allocate_markov_workspace(mapping) -> Optional[MarkovWorkspace]:
    """Workspace for ``markov_chain_fused`` on ``mapping``'s TP group, or None if unsupported.

    Collective over the TP group (IPC or fabric handle exchange): every rank must call it at the
    same point.
    """
    from tensorrt_llm._ipc_utils import can_access_peer

    if not _device_supported():
        return None
    if mapping.enable_attention_dp or mapping.tp_size not in (2, 4, 8):
        return None
    if can_access_peer(mapping):
        return MarkovWorkspace(mapping)
    # A TP group across nodes: fabric-mapped buffers where the NVLink domain spans the nodes.
    from ..distributed.ops import MNNVLAllReduce

    if not MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True):
        return None
    try:
        workspace = MarkovWorkspace(mapping, fabric=True)
    except RuntimeError as e:  # every rank of the group raised (see _fabric_lamport_table)
        logger.warning(f"DSpark fused Markov chain: no fabric workspace ({e}); unfused chain")
        return None
    logger.info(f"DSpark fused Markov chain: fabric-memory exchange across TP{mapping.tp_size}")
    return workspace


@functools.lru_cache(maxsize=None)
def _device_supported() -> bool:
    from ..cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    return (
        IS_CUTLASS_DSL_AVAILABLE
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
    )


@functools.lru_cache(maxsize=None)
def _num_ctas(local_vocab: int, device_index: int) -> int:
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    return sms if ceil_div(local_vocab, sms) <= 256 else 0


@functools.lru_cache(maxsize=None)
def _compiled(
    batch: int,
    block: int,
    dim: int,
    local_vocab: int,
    vocab: int,
    nranks: int,
    num_ctas: int,
):
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute

    from ..cute_dsl_kernels.blackwell.dspark.markov_chain import DSparkMarkovChainKernel

    kernel = DSparkMarkovChainKernel(
        batch=batch,
        block=block,
        dim=dim,
        local_vocab=local_vocab,
        vocab=vocab,
        nranks=nranks,
        num_ctas=num_ctas,
    )
    dev = torch.device("cuda")
    base = torch.empty((batch * block, local_vocab), dtype=torch.bfloat16, device=dev)
    w2 = torch.empty((local_vocab, dim), dtype=torch.bfloat16, device=dev)
    w1 = torch.empty((vocab, dim), dtype=torch.bfloat16, device=dev)
    bonus = torch.empty((batch,), dtype=torch.int64, device=dev)
    out = torch.empty((batch, block), dtype=torch.int32, device=dev)
    ws = torch.empty((nranks + 2,), dtype=torch.int64, device=dev)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(
        kernel,
        graph_safe_from_dlpack(base),
        graph_safe_from_dlpack(w2),
        graph_safe_from_dlpack(w1),
        graph_safe_from_dlpack(bonus, 8),
        graph_safe_from_dlpack(out, 4),
        graph_safe_from_dlpack(ws),
        cutlass.Int32(0),
        cutlass.Int32(0),
        stream,
    )


def markov_chain_supported(
    base: torch.Tensor, w2: torch.Tensor, w1: torch.Tensor, workspace: Optional[MarkovWorkspace]
) -> bool:
    """Whether ``markov_chain_fused`` serves these operands (shapes, dtypes, layout, workspace)."""
    if workspace is None or base.dim() != 3 or not 0 < base.shape[0] <= MAX_ROWS:
        return False
    if base.dtype != torch.bfloat16:
        return False
    batch, block, local_vocab = base.shape
    num_ctas = _num_ctas(local_vocab, base.device.index)
    return (
        w2.shape == (local_vocab, w1.shape[1])
        and w1.shape[1] == 256
        and num_ctas > 0
        and block * batch * workspace.nranks * num_ctas * _SLOT_VECTOR_BYTES <= workspace.slot_bytes
        and w1.shape[0] <= _MAX_VOCAB
    )


def markov_chain_fused(
    base: torch.Tensor,
    w2: torch.Tensor,
    w1: torch.Tensor,
    bonus: torch.Tensor,
    offset: int,
    vocab: int,
    workspace: MarkovWorkspace,
) -> torch.Tensor:
    """Draft tokens ``[B, K]`` int32 of the Markov chain over this rank's vocabulary shard.

    ``base`` ``[B, K, local_vocab]`` BF16 draft-head logits of this rank's shard (global ids
    ``offset ..``), ``w2`` ``[local_vocab, D]`` the matching Markov projection rows, ``w1``
    ``[vocab, D]`` the Markov embedding, ``bonus`` ``[B]`` the tokens preceding position 0. Every
    rank of the workspace's TP group must call it with the same ``B`` and ``K``.
    """
    import cuda.bindings.driver as cuda
    import cutlass

    batch, block, local_vocab = base.shape
    out = torch.empty((batch, block), dtype=torch.int32, device=base.device)
    num_ctas = _num_ctas(local_vocab, base.device.index)
    fn = _compiled(
        batch,
        block,
        w2.shape[1],
        local_vocab,
        vocab,
        workspace.nranks,
        num_ctas,
    )
    bonus = bonus.to(torch.int64).contiguous()
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        graph_safe_from_dlpack(base.view(batch * block, local_vocab)),
        graph_safe_from_dlpack(w2),
        graph_safe_from_dlpack(w1),
        graph_safe_from_dlpack(bonus, 8),
        graph_safe_from_dlpack(out, 4),
        graph_safe_from_dlpack(workspace.table),
        cutlass.Int32(workspace.rank),
        cutlass.Int32(offset),
        stream,
    )
    return out
