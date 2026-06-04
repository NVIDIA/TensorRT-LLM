# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are translated verbatim from SGLang's MiniMax-M3
# naive reference implementation. Upstream source:
#
#   /home/scratch.fredw_sw/workspace/hidden_trail/minimax-m3-sglang-triton_vv1/
#     python/sglang/srt/layers/attention/minimax_sparse_ops/naive/topk_sparse.py
#     python/sglang/srt/layers/attention/minimax_sparse_ops/naive/flash_with_topk_idx.py
#
# Both upstream files carry "Copyright 2025 XunhaoLai. All rights reserved."
# and are vendored here as a parity reference for TensorRT-LLM's MiniMax-M3
# sparse attention. The vendored functions are pure PyTorch (no SGLang
# runtime, no GPU server dependency); they therefore run under the standard
# CUDA-only test process and can be cross-checked against the in-file
# hand-written reference in test_minimax_m3_sparse_attention.py.
"""SGLang-naive sparse-attention reference for MiniMax-M3 parity tests.

The functions here are translations of SGLang's
``minimax_sparse_ops/naive/*.py`` semantics — index-attention block scoring
(``naive_flash_decode_with_topk_idx``) and sparse-GQA attention over
selected blocks (``naive_flash_decode_with_gqa_share_sparse``) — adapted
from SGLang's contiguous ``[max_slots, 2, max_len, num_heads, head_dim]``
KV-cache layout to the paged ``[num_slots, num_kv_heads, head_dim]``
layout TensorRT-LLM's :class:`KVCacheManagerV2` produces.

The adaptation is deliberately minimal:

  * Index scoring uses SGLang's exact einsum form with explicit `block_size`
    max-pooling and the same ``init_blocks`` / ``local_blocks`` priority
    constants (``1e30`` / ``1e29``).
  * Sparse GQA uses SGLang's per-(kv_head, batch) block-selection /
    contiguous slice / softmax / matmul sequence, again driven by the
    paged ``req_to_token`` mapping rather than the contiguous original.
  * Per-head top-k selection followed by per-kv-head union (idx_group_size
    Q heads share one kv head and therefore one selected-block set) matches
    SGLang's contract.

The whole point of this module is *independence from the in-file
hand-written reference*: a bug in either the SUT or the in-file reference
that they share will fail this cross-check. The module is import-only
during test collection and never executed at runtime outside the tests
that explicitly call it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

# Priority constants from SGLang's
# ``naive/flash_with_topk_idx.py``.
_SGLANG_INIT_SCORE: float = 1e30
_SGLANG_LOCAL_SCORE: float = 1e29


@dataclass(frozen=True)
class SGLangNaiveSparseConfig:
    """The subset of MiniMax-M3 sparse-attention geometry the naive ref needs.

    Mirrors the relevant fields of
    :class:`tensorrt_llm._torch.attention_backend.sparse.minimax_m3.MiniMaxM3SparseConfig`
    so callers can pass either object via duck typing.
    """

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    num_index_heads: int
    sparse_index_dim: int
    block_size: int
    topk: int
    init_blocks: int = 0
    local_blocks: int = 0
    score_type: str = "max"

    @classmethod
    def from_minimax_config(cls, cfg) -> "SGLangNaiveSparseConfig":
        return cls(
            num_q_heads=int(cfg.num_q_heads),
            num_kv_heads=int(cfg.num_kv_heads),
            head_dim=int(cfg.head_dim),
            num_index_heads=int(cfg.num_index_heads),
            sparse_index_dim=int(cfg.sparse_index_dim),
            block_size=int(cfg.block_size),
            topk=int(cfg.topk),
            init_blocks=int(getattr(cfg, "init_blocks", 0)),
            local_blocks=int(getattr(cfg, "local_blocks", 0)),
            score_type=str(getattr(cfg, "score_type", "max")),
        )


def _gather_paged_sequence(
    *,
    cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_row: int,
    seq_len: int,
) -> torch.Tensor:
    """Gather a contiguous per-sequence slice out of the paged cache.

    Supports two cache layouts, mirroring the production
    ``_gather_paged_batched`` helper:

      * **3-D flat-slot** ``[num_slots, *channel_dims]``: the legacy
        plain per-slot allocation kept by older fixtures that still
        allocate the cache as a contiguous flat-slot tensor. Slot ids
        in ``req_to_token`` are direct dim-0 indices.
      * **4-D paged** ``[num_pages, tokens_per_block, *channel_dims]``:
        the V2 paged layout the index-K cache exposes after the Stage 14
        rewrite (and the main K/V views ``kv_pool[:, 0]`` / ``kv_pool[:,
        1]`` when not reshaped). The slot id ``s`` is decomposed as
        ``page = s // tokens_per_block, within = s % tokens_per_block``,
        and multi-dim fancy indexing reads the per-token slice directly
        through the pool view.

    Returns ``[seq_len, *channel_dims]`` for both layouts.
    """
    slots = req_to_token[slot_row, :seq_len].to(torch.long)
    if cache.ndim >= 4:
        tokens_per_block = int(cache.shape[1])
        page = slots // tokens_per_block
        within = slots % tokens_per_block
        return cache[page, within]
    return cache.index_select(0, slots)


def sglang_naive_topk_select(
    *,
    idx_q: torch.Tensor,  # [num_index_heads, sparse_index_dim]
    idx_k_seq: torch.Tensor,  # [seq_len, sparse_index_dim]
    config: SGLangNaiveSparseConfig,
    idx_sm_scale: float,
    causal_pos: int,
) -> torch.Tensor:
    """Compute the per-(kv_head) selected blocks for one Q token.

    Returns a ``[num_kv_heads, idx_group_size * topk]`` int64 tensor,
    -1-padded, holding the union of per-index-head top-k block indices
    inside the GQA group. This matches SGLang's contract for the
    block-selection stage: each kv-head consumes one merged set of block
    ids and runs sparse-GQA against the main K/V of those positions.

    Replicates SGLang's exact priority + scoring path:

      * Score per (head, block) = max over ``block_size`` positions of
        ``(idx_q @ idx_k_pos) * idx_sm_scale`` for positions ``<=
        causal_pos``; positions past ``causal_pos`` are masked with
        ``-inf`` (causal) and the block max-pool sees no contribution
        from them.
      * Init priority: ``score[:, :init_blocks] = INIT_SCORE`` forces the
        first ``init_blocks`` blocks into the top-k regardless of value.
      * Local priority: the last ``local_blocks`` valid blocks get
        ``LOCAL_SCORE`` so the tail window is always selected.
      * Per-head top-k over the ``n_valid`` blocks (clamped to the
        causal position). Padded blocks past ``n_valid`` are masked with
        ``-inf`` to exclude them from selection.
      * Union per kv-head: the GQA group's ``idx_group_size`` index heads
        share one kv head, so the selected block ids are unioned across
        the group.
    """
    num_idx_heads = int(config.num_index_heads)
    num_kv_heads = int(config.num_kv_heads)
    block_size = int(config.block_size)
    topk = int(config.topk)
    init_blocks = int(config.init_blocks)
    local_blocks = int(config.local_blocks)
    idx_group_size = num_idx_heads // num_kv_heads
    seq_len = int(idx_k_seq.shape[0])

    # Translate SGLang's einsum step but for a single Q token.
    qk = (idx_q.float() @ idx_k_seq.float().T) * idx_sm_scale  # [H, L]
    arange = torch.arange(seq_len, device=qk.device)
    causal_mask = arange.unsqueeze(0) <= causal_pos
    qk = qk.masked_fill(~causal_mask, float("-inf"))

    # Max-pool over block_size tokens, then add SGLang's priority constants.
    pad = (block_size - (seq_len % block_size)) % block_size
    if pad:
        qk = torch.cat([qk, qk.new_full((num_idx_heads, pad), float("-inf"))], dim=-1)
    n_blocks = qk.shape[-1] // block_size
    scores = qk.view(num_idx_heads, n_blocks, block_size).amax(dim=-1)

    # Causal n_valid clamp.
    eff = min(seq_len, causal_pos + 1) if causal_pos >= 0 else 0
    n_valid = (eff + block_size - 1) // block_size

    if init_blocks > 0 and n_valid > 0:
        scores[:, : min(init_blocks, n_valid)] = _SGLANG_INIT_SCORE
    if local_blocks > 0 and n_valid > 0:
        start = max(0, n_valid - local_blocks)
        if start < n_valid:
            scores[:, start:n_valid] = _SGLANG_LOCAL_SCORE
    if n_valid < n_blocks:
        scores[:, n_valid:] = float("-inf")

    per_head = torch.full((num_idx_heads, topk), fill_value=-1, device=qk.device, dtype=torch.int64)
    take = min(topk, n_valid) if n_valid > 0 else 0
    if take > 0:
        _, idx = scores[:, :n_valid].topk(k=take, dim=-1)
        per_head[:, :take] = idx

    per_kv: List[torch.Tensor] = []
    for kh in range(num_kv_heads):
        chunk = per_head[kh * idx_group_size : (kh + 1) * idx_group_size]
        uniq = sorted({int(x) for x in chunk.flatten().tolist() if int(x) >= 0})
        out = torch.full((idx_group_size * topk,), -1, device=qk.device, dtype=torch.int64)
        for j, v in enumerate(uniq):
            out[j] = v
        per_kv.append(out)
    return torch.stack(per_kv, dim=0)


def sglang_naive_sparse_gqa_attention(
    *,
    q: torch.Tensor,  # [num_q_heads, head_dim]
    k_seq: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    v_seq: torch.Tensor,  # [seq_len, num_kv_heads, head_dim]
    topk_per_kv: torch.Tensor,  # [num_kv_heads, topk_union]
    block_size: int,
    sm_scale: float,
    causal_pos: int,
) -> torch.Tensor:
    """SGLang's sparse-GQA attention for one Q token.

    Translates ``naive_flash_decode_with_gqa_share_sparse`` from the
    upstream file but driven by a single contiguous ``k_seq`` /``v_seq``
    rather than the upstream slotted cache view; the calling code does
    the paged gather first.
    """
    num_q_heads = int(q.shape[0])
    num_kv_heads = int(k_seq.shape[1])
    head_dim = int(q.shape[1])
    gqa_group_size = num_q_heads // num_kv_heads
    out = torch.zeros(num_q_heads, head_dim, device=q.device, dtype=q.dtype)

    for kh in range(num_kv_heads):
        # Build a position list from the per-kv-head topk_union, respecting
        # block size and causal trimming. This is the same per-block loop
        # SGLang's ``naive_flash_decode_with_gqa_share_sparse`` uses.
        positions: List[int] = []
        for bi in topk_per_kv[kh].tolist():
            if bi < 0:
                continue
            start = bi * block_size
            end = min(start + block_size, causal_pos + 1)
            if start >= causal_pos + 1:
                continue
            positions.extend(range(start, end))
        if not positions:
            continue
        sel = torch.tensor(positions, device=q.device, dtype=torch.long)
        ksel = k_seq.index_select(0, sel)[:, kh, :].float()
        vsel = v_seq.index_select(0, sel)[:, kh, :].float()
        for gi in range(gqa_group_size):
            qh = kh * gqa_group_size + gi
            q_vec = q[qh].float()
            scores = (q_vec @ ksel.T) * sm_scale
            attn = scores.softmax(dim=-1)
            out[qh] = (attn @ vsel).to(q.dtype)
    return out


def sglang_naive_sparse_decode(
    *,
    q: torch.Tensor,  # [batch, num_q_heads, head_dim]
    idx_q: torch.Tensor,  # [batch, num_index_heads, sparse_index_dim]
    k_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim]
    v_cache: torch.Tensor,  # [num_slots, num_kv_heads, head_dim]
    idx_k_cache: torch.Tensor,  # [num_slots, 1, sparse_index_dim]
    req_to_token: torch.Tensor,  # [num_reqs, max_tokens]
    slot_ids: torch.Tensor,  # [batch]
    seq_lens: torch.Tensor,  # [batch]
    config: SGLangNaiveSparseConfig,
    sm_scale: float,
    idx_sm_scale: float,
) -> torch.Tensor:
    """Decode-side reference: SGLang-derived block selection + sparse GQA.

    Returns ``[batch, num_q_heads * head_dim]`` (matches the SUT's flat
    output convention).
    """
    batch = int(q.shape[0])
    outs: List[torch.Tensor] = []
    for b in range(batch):
        sl = int(seq_lens[b].item())
        slot_row = int(slot_ids[b].item())
        idx_k_seq = _gather_paged_sequence(
            cache=idx_k_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        ).squeeze(1)  # [seq_len, sparse_index_dim]
        k_seq = _gather_paged_sequence(
            cache=k_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        )
        v_seq = _gather_paged_sequence(
            cache=v_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        )
        topk_per_kv = sglang_naive_topk_select(
            idx_q=idx_q[b],
            idx_k_seq=idx_k_seq,
            config=config,
            idx_sm_scale=idx_sm_scale,
            causal_pos=sl - 1,
        )
        o = sglang_naive_sparse_gqa_attention(
            q=q[b],
            k_seq=k_seq,
            v_seq=v_seq,
            topk_per_kv=topk_per_kv,
            block_size=config.block_size,
            sm_scale=sm_scale,
            causal_pos=sl - 1,
        )
        outs.append(o.reshape(config.num_q_heads * config.head_dim))
    return torch.stack(outs, dim=0)


def sglang_naive_sparse_prefill(
    *,
    q: torch.Tensor,  # [total_q, num_q_heads, head_dim]
    idx_q: torch.Tensor,  # [total_q, num_index_heads, sparse_index_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    config: SGLangNaiveSparseConfig,
    sm_scale: float,
    idx_sm_scale: float,
) -> torch.Tensor:
    """Prefill-side reference: per-Q-token block selection + sparse GQA.

    Each Q token in a chunk uses the same per-sequence paged K/V/idx_K
    but a different causal position. Returns
    ``[total_q, num_q_heads * head_dim]``.
    """
    total_q = int(q.shape[0])
    out = torch.zeros(
        total_q,
        config.num_q_heads * config.head_dim,
        device=q.device,
        dtype=q.dtype,
    )
    cu = cu_seqlens_q.tolist()
    batch = int(slot_ids.shape[0])
    for b in range(batch):
        start, end = int(cu[b]), int(cu[b + 1])
        sl = int(seq_lens[b].item())
        slot_row = int(slot_ids[b].item())
        idx_k_seq = _gather_paged_sequence(
            cache=idx_k_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        ).squeeze(1)
        k_seq = _gather_paged_sequence(
            cache=k_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        )
        v_seq = _gather_paged_sequence(
            cache=v_cache, req_to_token=req_to_token, slot_row=slot_row, seq_len=sl
        )
        pref = int(prefix_lens[b].item())
        for qi in range(start, end):
            causal_pos = pref + (qi - start)
            topk_per_kv = sglang_naive_topk_select(
                idx_q=idx_q[qi],
                idx_k_seq=idx_k_seq,
                config=config,
                idx_sm_scale=idx_sm_scale,
                causal_pos=causal_pos,
            )
            o = sglang_naive_sparse_gqa_attention(
                q=q[qi],
                k_seq=k_seq,
                v_seq=v_seq,
                topk_per_kv=topk_per_kv,
                block_size=config.block_size,
                sm_scale=sm_scale,
                causal_pos=causal_pos,
            )
            out[qi] = o.reshape(config.num_q_heads * config.head_dim)
    return out
