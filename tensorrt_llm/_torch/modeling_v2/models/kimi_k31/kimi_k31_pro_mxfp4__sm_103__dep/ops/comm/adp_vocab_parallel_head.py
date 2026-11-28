# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 LM head under attention DP, vocab-parallel: the ranks' rows all-gathered, one vocab shard per rank, the
results returned to the rows' owners.

Under attention DP every rank holds the full replicated LM head ``[vocab, hidden]`` and multiplies its own rows by
all of it (2.35 GB of BF16 weights read per rank per head per step). Here the rows of every rank, padded to the
group's largest count, are all-gathered, each rank multiplies all of them by its ``vocab / world`` rows of the
weight, and the owners get back either their logits rows (all-to-all of the shard columns, rebuilt in vocab order and
widened to FP32) or their rows' top-32 candidates (all-to-all of each shard's index-ordered (id, value) pairs,
merged).

The exchanges run on the group's NVLink peer mappings
(``kimi_k31_pro_mxfp4__sm_103__dep/kernels/comm/adp_rows_exchange``: one push kernel each, the FP32 widening inside
the return) once ``reserve_exchange`` has set up the group's symmetric workspace, for up to its ``max_rows`` rows per
rank; otherwise they are the engine's NCCL all-gather and send / recv all-to-all with the padding, rebuild and widening
as torch copies. Both move the same bits.

The shard GEMM over the gathered rows is the same cuBLAS ``F.linear`` as the replicated head; its outputs are the
replicated GEMM's columns bit for bit at the K3.1 DEP16 shapes (hidden 7168, vocab 163840, world 16) for padded row
counts 2..320 (the entry's test sweeps them), so the logits are the replicated head's ``.float()`` and the candidates
its top-32. A one-row GEMM takes another cuBLAS kernel (not the same bits): no rank may hold fewer than 2 rows. Other
worlds and shapes compute the same products but are not certified bitwise (smaller gathered GEMMs pick other kernels).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import cuda.bindings.driver as cuda
import pynvml
import torch
import torch.nn.functional as F
from cutlass import Int32, Int64

from tensorrt_llm._torch.distributed import ops as _dist_ops
from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce, allgather, alltoall_helix
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.spec.dflash2_top32 import (
    TOP_K,
    merge_top32,
    top32,
)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ...kernels.comm import adp_rows_exchange as _x

#: Kimi K3.1 vocabulary and hidden size.
_VOCAB = 163840
_HIDDEN = 7168


#: Rows per rank the exchange workspace is reserved for by default: 32 requests of 5 rows (DEP16 c512 holds 14 per
#: rank); 89 MB per rank at the K3.1 shapes over 16 ranks.
DEFAULT_MAX_ROWS = 160


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


class AdpExchangeWorkspace:
    """One attention-DP group's symmetric workspace (a ``McastGPUBuffer``) for the heads' row exchanges: the flags,
    the gather region ``[world * max_rows, hidden]`` BF16 and the return region of ``max_rows`` rows of ``vocab`` BF16
    (it also holds the top-32 pairs, ``world * 64`` FP32 per row). Allocated collectively by ``reserve_exchange``."""

    def __init__(self, mapping: Mapping, max_rows: int, hidden: int, vocab: int):
        world = mapping.tp_size
        if not 2 <= world <= _x.MAX_WORLD:
            raise ValueError(f"the exchange kernels serve 2..{_x.MAX_WORLD} ranks, got {world}")
        self.world, self.rank = world, mapping.tp_rank
        self.max_rows, self.hidden, self.vocab = max_rows, hidden, vocab
        self.goff = _x.FLAGS_BYTES
        self.roff = self.goff + _round_up(world * max_rows * hidden * 2, 1 << 16)
        ret_row = max(vocab * 2, world * 2 * TOP_K * 4)
        self.total_bytes = self.roff + _round_up(max_rows * ret_row, 1 << 16)
        comm = _dist_ops._get_mnnvl_workspace_comm(mapping)
        self.handle = _dist_ops._make_mnnvl_mcast_buffer(
            comm, self.total_bytes, mapping, mapping.is_multi_node()
        )
        device = torch.device("cuda", _dist_ops._mnnvl_device_index(mapping))
        self.local = self.handle.get_uc_buffer(self.rank, (self.total_bytes,), torch.uint8, 0)
        self.base = self.local.data_ptr()
        peers = [
            self.handle.get_uc_buffer(r, (1,), torch.uint8, 0).data_ptr() for r in range(world)
        ]
        self.peers = torch.tensor(peers, dtype=torch.int64, device=device)
        self.local[: _x.FLAGS_BYTES].zero_()
        torch.cuda.synchronize()
        if not _dist_ops._mnnvl_workspace_all_succeeded(comm, True):
            raise RuntimeError("vocab-parallel head exchange workspace setup failed on a rank")
        self._comm = comm
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    def serves(self, padded: int, hidden: int, vocab: int) -> bool:
        """Whether a call of ``padded`` rows per rank of this hidden size and vocabulary fits (the same answer on
        every rank: it depends on the group's counts and the model only)."""
        return padded <= self.max_rows and hidden == self.hidden and vocab == self.vocab

    @staticmethod
    def _stream() -> cuda.CUstream:
        return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    def gather(self, rows: torch.Tensor, padded: int) -> torch.Tensor:
        """Every rank's ``rows`` ``[n, hidden]`` (BF16) as ``[world * padded, hidden]`` (a view of the gather region;
        rank ``j``'s rows at ``j * padded``, the padding rows unspecified)."""
        n = rows.shape[0]
        if not rows.is_contiguous() or rows.data_ptr() % 16:
            rows = rows.clone(memory_format=torch.contiguous_format)
        row_bytes = self.hidden * 2
        ctas = max(1, min(self.num_sms, -(-n * (row_bytes // 16) // _x.THREADS)))
        _x.compiled_gather(self.world, row_bytes)(
            Int64(rows.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.base),
            Int64(self.goff),
            Int32(n),
            Int32(padded),
            Int32(self.rank),
            Int32(ctas),
            self._stream(),
        )
        size = self.world * padded * row_bytes
        return self.local[self.goff : self.goff + size].view(torch.bfloat16).view(-1, self.hidden)

    def give_back(self, local: torch.Tensor, padded: int, rows: int, widen: bool) -> torch.Tensor:
        """Rank ``j``'s rows of every rank's ``local`` ``[world * padded, cols]`` shard results back to rank ``j`` in
        rank order along the columns: ``[rows, world * cols]``, a new FP32 tensor with ``widen`` (BF16 ``local``),
        else a view of the return region in ``local``'s dtype (valid until the group's next exchange)."""
        cols = local.shape[1]
        col_bytes = cols * local.element_size()
        if widen:
            assert local.dtype == torch.bfloat16
            out = torch.empty((rows, self.world * cols), dtype=torch.float32, device=local.device)
        work = self.world * padded * (col_bytes // 16)
        ctas = max(1, min(self.num_sms, -(-work // _x.THREADS)))
        _x.compiled_return(self.world, col_bytes, widen)(
            Int64(local.data_ptr()),
            Int64(self.peers.data_ptr()),
            Int64(self.base),
            Int64(self.roff),
            Int64(out.data_ptr() if widen else 0),
            Int32(rows),
            Int32(padded),
            Int32(self.rank),
            Int32(ctas),
            self._stream(),
        )
        if widen:
            return out
        size = rows * self.world * col_bytes
        return self.local[self.roff : self.roff + size].view(local.dtype).view(rows, -1)


_EXCHANGES: Dict[tuple, AdpExchangeWorkspace] = {}


def _key(mapping: Mapping) -> tuple:
    return (tuple(mapping.tp_group), mapping.tp_rank)


def exchange_supported(mapping: Mapping) -> bool:
    """Whether the exchange kernels can run over this group: 2..32 ranks of SM100 / SM103 GPUs on one NVLink
    domain whose memory the ranks can map (the engine's MNNVL condition, single node included)."""
    try:
        local = (
            2 <= mapping.tp_size <= _x.MAX_WORLD
            and torch.cuda.get_device_capability() in ((10, 0), (10, 3))
            and MNNVLAllReduce.is_mnnvl(mapping, torch.bfloat16, explicitly_requested=True)
        )
    except pynvml.NVMLError:
        local = False
    return all(MnnvlMemory.get_comm(mapping).allgather(local))


def reserve_exchange(
    mapping: Mapping, max_rows: int, hidden: int = _HIDDEN, vocab: int = _VOCAB
) -> AdpExchangeWorkspace:
    """Set up (collectively: every rank of the group calls it at the same point, outside graph capture) the group's
    exchange workspace for up to ``max_rows`` rows per rank, and compile its kernels; an existing workspace that is
    large enough is kept."""
    ws = _EXCHANGES.get(_key(mapping))
    if ws is None or ws.max_rows < max_rows or (ws.hidden, ws.vocab) != (hidden, vocab):
        ws = AdpExchangeWorkspace(mapping, max_rows, hidden, vocab)
        _EXCHANGES[_key(mapping)] = ws
        logger.info(
            f"Vocab-parallel LM head exchange workspace: {ws.total_bytes >> 20} MiB per rank for {max_rows} rows "
            f"per rank over {ws.world} ranks."
        )
    _x.compiled_gather(ws.world, hidden * 2)
    _x.compiled_return(ws.world, vocab // ws.world * 2, True)
    _x.compiled_return(ws.world, 2 * TOP_K * 4, False)
    return ws


def _exchange(
    mapping: Mapping, padded: int, hidden: int, vocab: int
) -> Optional[AdpExchangeWorkspace]:
    """The group's exchange workspace where it serves this call (a decision every rank makes alike), else None."""
    ws = _EXCHANGES.get(_key(mapping))
    return ws if ws is not None and ws.serves(padded, hidden, vocab) else None


def _shard_rows(
    hidden: torch.Tensor, weight: torch.Tensor, mapping: Mapping, rows_per_rank: Sequence[int]
) -> tuple[torch.Tensor, int, int, Optional[AdpExchangeWorkspace]]:
    """``(local, R, shard, ws)``: this rank's vocab shard of every rank's rows ``[world * R, shard]`` (rank ``j``'s
    rows at ``[j R, j R + rows_per_rank[j])``, the rest padding) and the exchange workspace the call runs on (None:
    NCCL)."""
    world, rank = mapping.tp_size, mapping.tp_rank
    rows = int(hidden.shape[0])
    if rows != rows_per_rank[rank]:
        raise RuntimeError(
            f"vocab-parallel LM head: rank {rank} holds {rows} rows, the group's counts say {rows_per_rank[rank]}"
        )
    padded = max(rows_per_rank)
    ws = _exchange(mapping, padded, hidden.shape[1], weight.shape[0])
    if ws is not None:
        logger.info_once(
            "Vocab-parallel LM head under attention DP: row exchanges on NVLink peer mappings "
            "(kimi_k31_pro_mxfp4__sm_103__dep/kernels/comm/adp_rows_exchange).",
            key="k31_adp_vocab_exchange",
        )
        gathered = ws.gather(hidden, padded)
    else:
        x = hidden
        if rows != padded or not x.is_contiguous():
            x = hidden.new_zeros((padded, hidden.shape[1]))
            x[:rows].copy_(hidden)
        gathered = allgather(x, mapping, dim=0)
    shard = weight.shape[0] // world
    local = F.linear(gathered, weight[rank * shard : (rank + 1) * shard])
    return local, padded, shard, ws


def _cells(kind: str) -> tuple[Cell, ...]:
    return tuple(
        Cell(why=f"{kind}, {path}: {why}", spec=dict(world=4, rows=rows, exchange=path))
        for path in ("mnnvl", "nccl")
        for rows, why in (
            ((40, 40, 40, 40), "DEP16 c256 rows per rank (8 requests x 5), ranks padded alike"),
            ((70, 70, 70, 70), "DEP16 c512 rows per rank (14 requests x 5)"),
            ((65, 70, 5, 70), "uneven ranks (eager steps): padded to the largest"),
        )
    ) + (
        Cell(
            why=f"{kind}, mnnvl: 200 rows per rank, past the workspace's {DEFAULT_MAX_ROWS}: NCCL",
            spec=dict(world=4, rows=(200, 200, 200, 200), exchange="mnnvl"),
        ),
    )


class K31AdpVocabLogits(OpWrapper):
    """``F.linear(hidden, weight).float()``: the replicated LM head's FP32 logits of this rank's rows ``[rows, vocab]``,
    computed vocab-parallel over the attention-DP group (see the module docstring).

    Args:
        hidden: ``[rows, hidden]`` BF16, this rank's rows. weight: ``[vocab, hidden]`` BF16, the replicated head
            (``vocab`` divisible by the group size). mapping: the attention-DP group (``tp_group``, ``tp_rank``).
        rows_per_rank: every rank's row count, identical on every rank (the step's all-rank counts).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells("logits")
    note = """
    With the group's exchange workspace (``reserve_exchange``): the gather kernel, the shard GEMM (cuBLAS, as the
    replicated head) over the gathered rows in the workspace, the return kernel (shard columns pushed to the owners
    in vocab order, widened to FP32 in the same kernel). Otherwise NCCL all-gather of the padded rows, the shard GEMM,
    NCCL all-to-all of the shard columns (send / recv group) and one copy rebuilding the rows in vocab order as FP32.
    Bitwise the replicated head's ``.float()`` where the shard GEMM's columns are the full GEMM's (the test sweeps
    the row counts). Every rank of the group must make the call in the same step with the same ``rows_per_rank``.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> torch.Tensor:
        world = mapping.tp_size
        local, padded, shard, ws = _shard_rows(hidden, weight, mapping, rows_per_rank)
        rows = hidden.shape[0]
        if ws is not None:
            return ws.give_back(local, padded, rows, widen=True)
        recv = alltoall_helix(list(local.view(world, padded, shard).unbind(0)), mapping.tp_group)[0]
        out = torch.empty((rows, world * shard), dtype=torch.float32, device=hidden.device)
        out.view(rows, world, shard).copy_(recv[:, :rows].transpose(0, 1))
        return out

    def reference(
        self,
        hidden_in_rank_order: Sequence[torch.Tensor],
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> torch.Tensor:
        """Collective: every rank's rows in rank order; ``mapping.tp_rank``'s replicated-head logits."""
        return F.linear(hidden_in_rank_order[mapping.tp_rank], weight).float()

    def is_valid(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> None:
        assert hidden.dtype == weight.dtype == torch.bfloat16, "BF16 rows and head"
        assert weight.shape[0] % mapping.tp_size == 0, "the vocabulary splits evenly over the group"
        assert (
            len(rows_per_rank) == mapping.tp_size
            and hidden.shape[0] == rows_per_rank[mapping.tp_rank]
        )
        assert min(rows_per_rank) >= 2, (
            "a one-row replicated GEMM is another kernel: not the same bits"
        )
        assert weight.is_contiguous()


class K31AdpVocabTop32(OpWrapper):
    """``k31_dflash2_shard_top32(F.linear(hidden, weight).float(), 32, 0, False)``: the top-32 candidates (FP32
    values, int64 ids, torch.topk's order) of the replicated LM head's logits of this rank's rows, computed
    vocab-parallel over the attention-DP group: every rank's shard top-32 of every row as index-ordered pairs, the
    pairs all-to-all'd to the rows' owners and merged (``ops/spec/dflash2_global_top32``).

    Args: as ``K31AdpVocabLogits``. Returns ``(unary [rows, 32] FP32, ids [rows, 32] int64)``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells("top32")
    note = """
    The gather (kernel or NCCL, as K31AdpVocabLogits), the shard GEMM, the index-ordered shard top-32 pairs of
    every row, their all-to-all (return kernel into the workspace, or NCCL; 256 B per row and peer), the merge.
    Bitwise the full-vocab top-32 of the replicated head's FP32 logits where the shard GEMM's columns are the full
    GEMM's. Collective rules as K31AdpVocabLogits.
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        world, rank = mapping.tp_size, mapping.tp_rank
        local, padded, shard, ws = _shard_rows(hidden, weight, mapping, rows_per_rank)
        pairs = top32(local, id_offset=rank * shard, pairs=True, index_order=True)
        rows = hidden.shape[0]
        if ws is not None:
            return merge_top32(ws.give_back(pairs, padded, rows, widen=False))
        recv = alltoall_helix(
            list(pairs.view(world, padded, 2 * TOP_K).unbind(0)), mapping.tp_group
        )[0]
        gathered = recv[:, :rows].transpose(0, 1).reshape(rows, world * 2 * TOP_K)
        return merge_top32(gathered)

    def reference(
        self,
        hidden_in_rank_order: Sequence[torch.Tensor],
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collective: every rank's rows in rank order; ``mapping.tp_rank``'s candidates of the replicated head."""
        logits = F.linear(hidden_in_rank_order[mapping.tp_rank], weight).float()
        return top32(logits)

    def is_valid(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        mapping: Mapping,
        rows_per_rank: Sequence[int],
    ) -> None:
        K31AdpVocabLogits.is_valid(self, hidden, weight, mapping, rows_per_rank)
        shard = weight.shape[0] // mapping.tp_size
        assert shard % 8 == 0 and weight.shape[0] < (1 << 24), "pair ids exact in FP32"


def build_top32_kernels(shard: int, dtype: torch.dtype, world: int, device: torch.device) -> None:
    """Compile the CuTe DSL kernels ``K31AdpVocabTop32`` launches for ``world`` vocab shards of ``shard`` columns of
    ``dtype`` logits (the shard top-32 index pairs and their merge) by running them once on two zero rows, with no
    collective. Their first call in a process costs seconds of host time (tracing, even with a warm compile cache);
    a caller runs this eagerly before the first step that may take the op."""
    pairs = top32(torch.zeros((2, shard), dtype=dtype, device=device), pairs=True, index_order=True)
    merge_top32(pairs.repeat(1, world))


k31_adp_vocab_logits = K31AdpVocabLogits()
k31_adp_vocab_top32 = K31AdpVocabTop32()

__all__ = [
    "AdpExchangeWorkspace",
    "K31AdpVocabLogits",
    "K31AdpVocabTop32",
    "build_top32_kernels",
    "exchange_supported",
    "k31_adp_vocab_logits",
    "k31_adp_vocab_top32",
    "reserve_exchange",
]
