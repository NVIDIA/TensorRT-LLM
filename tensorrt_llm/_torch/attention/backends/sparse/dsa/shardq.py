# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-sharded sparse MLA prefill under TP without attention DP ("ShardQ").

Head-sharded TP runs the sparse MLA prefill with ``num_heads / tp`` heads per rank on every context token (12 heads
for Kimi K3.1 at TP8), a head group that no prefill kernel runs efficiently, and every rank scores every token in the
DSA indexer. ShardQ splits the context query tokens over the TP ranks like the indexer's query split instead: every
rank scores only its own rows in the indexer (its top-k all-gather is skipped) and runs FlashMLA sparse prefill
(``flash_mla_prefill``) for its rows with all heads. The projected queries are exchanged from head shards to token
shards before the absorb (``[T, H, qk_nope + qk_rope]``, not the absorbed ``[T, H, kv_lora + qk_rope]``), and the
V-projected outputs back to head shards before the output gate and ``o_proj`` (NCCL all-to-alls). Each rank keeps an
all-head copy of the absorb / V weights for this, gathered once per layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import torch

from tensorrt_llm._torch.distributed import allgather, alltoall_helix
from tensorrt_llm._torch.utils import is_torch_compiling
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.logger import logger

from .flash_mla_prefill import (
    FLASH_MLA_HEADS,
    flash_mla_fp8_prefill_supported,
    flash_mla_sparse_prefill_rows,
    gather_context_kv_bf16,
    kv_scale_is_unity,
    rebase_topk_to_compact_kv,
)
from .metadata import DSAtrtllmAttentionMetadata
from .params import disagg_role, is_kimi_k31_dsa
from .prims_ts_prefill import prims_ts_prefill_supported, prims_ts_sparse_prefill_rows

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.mla import MLA

# Fewest context tokens of a step for the token-sharded prefill.
_SHARDQ_MIN_TOKENS = 1024


def use_shardq(mla: "MLA", attn_metadata: DSAtrtllmAttentionMetadata) -> bool:
    """Whether this context step runs the token-sharded sparse MLA prefill (see module docstring): Kimi K3.1 on a
    disaggregated context server under TP without attention DP."""
    mapping = mla.mapping
    if not (
        is_kimi_k31_dsa()
        and disagg_role() == "context"
        and mapping.tp_size > 1
        and not mapping.enable_attention_dp
        and attn_metadata.num_contexts > 0
        and attn_metadata.num_ctx_tokens >= _SHARDQ_MIN_TOKENS
        and mla.num_heads_tp * mapping.tp_size <= FLASH_MLA_HEADS
        and not attn_metadata.is_cuda_graph
        and not is_torch_compiling()
        and flash_mla_fp8_prefill_supported(mla)
        # Shared-indexer layers (and MTP index sharing) read every row's top-k, and a GVR decode prior takes each
        # request's last prefill row; ShardQ only computes this rank's rows.
        and attn_metadata.shared_topk_indices is None
        and not mla.mqa.indexer.top_k.needs_gvr_prior
    ):
        return False
    if not kv_scale_is_unity(mla):
        logger.warning_once(
            "DSA ShardQ prefill needs a unit FP8 KV scale; keeping the head-sharded prefill",
            key="dsa_shardq_kv_scale",
        )
        return False
    return True


def _shardq_bounds(count: int, tp_size: int) -> Tuple[List[int], List[int]]:
    """Token split of ``count`` context rows over ``tp_size`` ranks; the indexer's TP query split
    (``_split_prefill_queries``) uses the same bounds."""
    begins = [count * r // tp_size for r in range(tp_size)]
    ends = [count * (r + 1) // tp_size for r in range(tp_size)]
    return begins, ends


def _compact_rows_of_tokens(
    attn_metadata: DSAtrtllmAttentionMetadata, begin: int, end: int
) -> Tuple[int, int]:
    """Compact KV rows ``[lo, hi)`` of the context requests owning context tokens ``[begin, end)`` (host math)."""
    if end <= begin:
        return 0, 0
    num_contexts = attn_metadata.num_contexts
    q_lens = attn_metadata.seq_lens[:num_contexts].tolist()
    kv_indptr = attn_metadata.host_ctx_kv_indptr[: num_contexts + 1].tolist()
    first, last, token = None, num_contexts - 1, 0
    for i, n in enumerate(q_lens):
        if first is None and begin < token + n:
            first = i
        if end <= token + n:
            last = i
            break
        token += n
    return int(kv_indptr[first or 0]), int(kv_indptr[last + 1])


def _all_head_weights(mla: "MLA") -> Tuple[torch.Tensor, torch.Tensor]:
    """All-head copies of the absorb (``k_b_proj_trans``) and V (``v_b_proj``) weights, gathered over TP once."""
    w = getattr(mla, "_shardq_all_head_weights", None)
    if w is None:
        w_uk = allgather(mla.k_b_proj_trans.data.contiguous(), mla.mapping, dim=0)
        w_uv = allgather(mla.v_b_proj.data.contiguous(), mla.mapping, dim=0)
        w = (w_uk.contiguous(), w_uv.contiguous())
        mla._shardq_all_head_weights = w
    return w


def _exchange_rows(pieces: List[torch.Tensor], rows: int, group: List[int]) -> torch.Tensor:
    """All-to-all of per-peer ``[n_j, ...]`` pieces padded to ``rows``; returns ``[tp, rows, ...]``."""
    send = []
    for p in pieces:
        if p.shape[0] != rows or not p.is_contiguous():
            buf = p.new_empty((rows,) + tuple(p.shape[1:]))
            buf[: p.shape[0]].copy_(p)
            p = buf
        send.append(p)
    return alltoall_helix(send, group)[0]


@nvtx_range("forward_sparse_mla_shardq")
def forward_context_sparse_attn_shardq(
    mla: "MLA",
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
) -> torch.Tensor:
    """Token-sharded sparse MLA prefill for the context rows ``q`` ``[T, H * (qk_nope + qk_rope)]`` (this rank's
    head shard); writes ``output`` ``[T, H * v_head]`` (this rank's head shard)."""
    mapping = mla.mapping
    tp, rank = mapping.tp_size, mapping.tp_rank
    num_tokens = q.shape[0]
    heads = mla.num_heads_tp
    qk_dim = mla.qk_nope_head_dim + mla.qk_rope_head_dim
    v_dim = mla.v_head_dim

    # Every rank appends the whole chunk's latent KV (and applies the identity / real RoPE to its q_pe).
    mla.mqa.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata, is_generation=False)

    begins, ends = _shardq_bounds(num_tokens, tp)
    begin, end = begins[rank], ends[rank]
    rows = max(e - b for b, e in zip(begins, ends))

    # Indexer top-k for this rank's rows only: under the indexer's TP query split the gather is skipped.
    attn_metadata.indexer_skip_q_split_gather = True
    try:
        topk = mla.mqa.indexer.forward_from_projected(
            attn_metadata, q, indexer_intermediates, is_generation=False
        )
    finally:
        attn_metadata.indexer_skip_q_split_gather = False

    q_heads = q.view(num_tokens, heads, qk_dim)
    recv = _exchange_rows([q_heads[b:e] for b, e in zip(begins, ends)], rows, mapping.tp_group)
    n = end - begin
    q_all = recv[:, :n].transpose(0, 1).reshape(n, tp * heads, qk_dim)

    w_uk, w_uv = _all_head_weights(mla)
    out_rows = q.new_empty(n, tp * heads, v_dim)
    attn_metadata._ensure_pool_view_cached()
    if prims_ts_prefill_supported(mla, tp * heads):
        # PrimTS attends the FP8 pool rows of this rank's slice in place (no compact BF16 copy).
        prims_ts_sparse_prefill_rows(
            mla,
            q_all,
            w_uk,
            w_uv,
            topk[begin:end],
            attn_metadata._cached_req_idx_ctx[begin:end],
            attn_metadata,
            out_rows,
        )
    else:
        # Only the KV of the requests this rank's rows belong to.
        row_lo, row_hi = _compact_rows_of_tokens(attn_metadata, begin, end)
        kv = gather_context_kv_bf16(mla, attn_metadata, row_lo, row_hi)
        indices = rebase_topk_to_compact_kv(
            topk[begin:end], attn_metadata._cached_req_idx_ctx[begin:end], attn_metadata, row_lo
        )
        if n > 0:
            flash_mla_sparse_prefill_rows(mla, q_all, w_uk, w_uv, kv, indices, out_rows)

    out_heads = out_rows.view(n, tp, heads, v_dim).transpose(0, 1)
    back = _exchange_rows([out_heads[j] for j in range(tp)], rows, mapping.tp_group)
    out_view = output.view(num_tokens, heads, v_dim)
    if all(e - b == rows for b, e in zip(begins, ends)):
        out_view.copy_(back.view(tp * rows, heads, v_dim))
    else:
        for j, (b, e) in enumerate(zip(begins, ends)):
            out_view[b:e].copy_(back[j, : e - b])
    return output
