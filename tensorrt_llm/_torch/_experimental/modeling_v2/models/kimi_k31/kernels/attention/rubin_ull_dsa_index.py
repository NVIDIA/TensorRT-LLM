# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL kernels of the K3.1 flat DSA decode indexer (Helix key-sharded Top-K).

``key_shard_local_select`` runs the key-shard decode's local part as a PDL chain of two kernels:
the paged logits (``logits.paged_logits_pdl``) and one select/pack kernel
(``select_pack.KeyShardSelectPack``) that replaces the local Top-L and the pack, and merges:

- where the runner supports it: the same kernel pushes its candidates as LL pairs (data and epoch
  in each 8 bytes, no flag or system fence) and merges them, re-selecting rows with short ranks
  from those ranks' pushed local Top-K (``exchange.py``; on 16 GB300 ranks, 8 rows, 3.7-4.5 us
  less per layer from logits to merge end than the Triton merge at 1k-512k context);
- else: the select/pack publishes through the flags and the Triton merge
  (``KeyShardTopKMerge.merge_owned``) follows.

Both produce the same rows.

With ``kv_rows`` (``KvRows``: the MLA block table, request of each row, the layer's page stride and
offset) the exchange writes the winners as KV pool rows (every variant: one kernel, the cluster
and tops select, the split merge grid), bitwise what the remap
(``convertReqIndexToGlobal``) makes of its local token indices, so no remap follows (``ROWS``).

The logits come from the key-balanced persistent scorer (``score.score_logits_pdl``, after vLLM's
``IndexerScore`` with an FP32 epilogue that writes the same bits as the tree kernel) where it supports
the step's shape, else from the tree kernel.
With the scorer and the LL exchange, the scorer also writes each 8 keys' largest logit and rows longer
than one register chunk select from those maxima (``local_topk.py``, vLLM's ``IndexerSelect`` bound),
in a select / push grid followed by a merge grid
launched early by PDL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

#: ``key_shard_local_select`` results: nothing launched, local select/pack only, merged too
#: (local token indices), merged as KV pool rows.
NOT_LAUNCHED, PACKED, MERGED, ROWS = 0, 1, 2, 3

#: Tokens per page of the indexer and MLA caches the kernels' index arithmetic assumes.
PAGE_TOKENS = 64


@dataclass(frozen=True)
class KvRows:
    """The remap of a step's generation rows, ``convertReqIndexToGlobal``'s arguments: KV pool
    row of request-local token ``t`` of row ``r`` =
    ``block_table[req_idx[r]][t // 64] * stride_rows + row_offset + t % 64`` (int32), ``-1``
    past the table's columns or on a negative entry.

    ``stride_rows`` = page index scale x tokens per block, ``row_offset`` = layer offset x tokens
    per block (``get_primary_pool_page_index_params``). Both are kernel arguments: one compiled
    kernel serves every layer, and a captured graph keeps each layer's."""

    block_table: torch.Tensor  # int32 [requests, pages], unit column stride
    req_idx: torch.Tensor  # int32 [>= rows], contiguous
    stride_rows: int
    row_offset: int
    tokens_per_block: int = PAGE_TOKENS

    def __post_init__(self):
        assert self.tokens_per_block == PAGE_TOKENS, self.tokens_per_block
        assert 0 <= self.stride_rows < 2**31 and 0 <= self.row_offset < 2**31
        table, req = self.block_table, self.req_idx
        assert table.dtype == torch.int32 and table.dim() == 2 and table.stride(1) == 1
        assert req.dtype == torch.int32 and req.dim() == 1 and req.is_contiguous()
        assert table.device == req.device

    def table_args(self, rows: int) -> tuple[int, int, int]:
        """(address, row pitch, columns) of the block table, for ``rows`` rows."""
        table = self.block_table
        assert self.req_idx.shape[0] >= rows and table.stride(0) < 2**31 and table.shape[1] < 2**31
        return table.data_ptr(), table.stride(0), table.shape[1]


_SELECT_PACK: dict = {}


def select_pack_for(merge):
    """The compiled select/pack of a merge workspace, or None where its shape is unsupported."""
    from .rubin_ull_dsa_index_select_pack import KeyShardSelectPack

    key = id(merge)
    entry = _SELECT_PACK.get(key)
    if entry is None or entry[0] is not merge:
        runner = KeyShardSelectPack(merge) if KeyShardSelectPack.supported(merge) else None
        entry = _SELECT_PACK[key] = (merge, runner)
    return entry[1]


def key_shard_local_select(
    merge,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    ctx_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule: torch.Tensor,
    local_len: int,
    row_lens: torch.Tensor,
    rows: int,
    owned=None,
    kv_rows: KvRows | None = None,
) -> int:
    """Local logits into the merge workspace, then the exact local Top-L pushed to every rank.

    ``owned`` = (out, counts, bounds, mask_out), ``merge_owned``'s arguments: the same kernel
    also merges into them (``MERGED``; through the LL exchange where enabled and supported, else
    through the flags where enabled); else the caller runs ``merge_owned`` (``PACKED``).
    ``NOT_LAUNCHED`` when the workspace shape has no compiled select/pack. With ``kv_rows``
    the LL exchange writes ``out`` as KV pool rows (``ROWS``); the other merges ignore it.
    """
    runner = select_pack_for(merge)
    if runner is None:
        return NOT_LAUNCHED
    scorer = "tree"
    from ...ops.attention.rubin_ull_dsa_indexer_score import dsa_indexer_score
    from ...ops.attention.rubin_ull_dsa_key_shard_select import dsa_key_shard_select
    from . import rubin_ull_dsa_index_score as score

    logits = merge.local_logits
    if score.unsupported(q, k_cache, weights, row_lens, block_table, logits) is None:
        # With the LL exchange the scorer also writes the vector maxima its select reads.
        tops = runner.tops if owned is not None and _merge_kind(runner) == "exchange" else None
        dsa_indexer_score(
            q, k_cache, weights, row_lens, block_table, logits, tops=tops, tops_from=runner.tops_from
        )
        scorer = "vllm"
    if scorer == "tree":
        from .rubin_ull_dsa_index_logits import paged_logits_pdl

        paged_logits_pdl(
            q, k_cache, weights, ctx_lens, block_table, schedule, local_len, merge.local_logits
        )
    _LAST_SCORE[(rows, merge.world)] = scorer
    kind = _merge_kind(runner) if owned is not None else "triton"
    _LAST_MERGE[(rows, merge.world)] = kind
    if kind == "exchange":
        out, counts, bounds, mask_out = owned
        dsa_key_shard_select(
            runner,
            row_lens,
            rows,
            out,
            counts,
            bounds,
            mask_out,
            exchange=True,
            tops=scorer == "vllm",
            kv_rows=kv_rows,
        )
        return MERGED if kv_rows is None else ROWS
    dsa_key_shard_select(runner, row_lens, rows)
    return PACKED


def _merge_kind(runner) -> str:
    """How the select/pack kernel of ``runner`` merges: "exchange" or "triton" (a separate
    ``merge_owned``)."""
    kind = getattr(runner, "merge_kind", None)
    if kind is None:
        kind = "exchange" if runner.supports_exchange() else "triton"
        runner.merge_kind = kind
    return kind


#: The merge of the last ``key_shard_local_select`` per (rows, world), for the path log.
_LAST_MERGE: dict = {}

#: The logits producer of the last ``key_shard_local_select`` per (rows, world): "vllm" (the
#: key-balanced scorer, ``score.py``) or "tree" (``FP8MQALogitsKernel``), for the path log.
_LAST_SCORE: dict = {}


_LOGGED_PATHS: dict = {}


def log_indexer_paths(
    prologue: str, select_pack: str, merge: str, rows: int, width: int, kv: str = "remap"
) -> None:
    """One INFO line per (rows, width) shape class on first use and whenever its paths change:
    ``K31_DSA_INDEXER prologue=cute|triton|chain select_pack=cute|fallback
    merge=exchange|cute|triton rows=<rows> width=<world> score=vllm|tree kv=direct|remap``. A
    kernel merge ("cute") is reported as the one ``key_shard_local_select`` launched; ``score`` is
    its logits producer (the tree kernel where the select/pack fell back); ``kv``: the merge wrote
    the KV pool rows (``ROWS``) or a remap follows."""
    if merge == "cute":
        merge = _LAST_MERGE.get((rows, width), merge)
    score = _LAST_SCORE.get((rows, width), "tree") if select_pack == "cute" else "tree"
    paths = (prologue, select_pack, merge, score, kv)
    if _LOGGED_PATHS.get((rows, width)) == paths:
        return
    _LOGGED_PATHS[(rows, width)] = paths
    from tensorrt_llm.logger import logger

    logger.info(
        f"K31_DSA_INDEXER prologue={prologue} select_pack={select_pack} merge={merge} "
        f"rows={rows} width={width} score={score} kv={kv}"
    )


def indexer_prologue(device: torch.device):
    """The CuTe DSL indexer prologue (keys into the cache, FP8 queries, scaled head weights)."""
    from .rubin_ull_dsa_index_prologue import indexer_prologue as runner

    return runner(device)
