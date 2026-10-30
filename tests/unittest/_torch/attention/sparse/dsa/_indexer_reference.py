# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference adapters for DSA indexer kernel tests."""

import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.vanilla_backend import DSAVanillaIndexer


def mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Reference DeepGEMM ragged-prefill logits with ``-inf`` padding."""
    seq_len_kv = kv.shape[0]
    positions = torch.arange(seq_len_kv, device=kv.device)
    mask = (positions[None, :] >= cu_seqlen_ks[:, None]) & (
        positions[None, :] < cu_seqlen_ke[:, None]
    )
    logits = DSAVanillaIndexer.weighted_relu_scores(q, kv, weights, mask)
    return logits.masked_fill(~mask, float("-inf"))


def paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    num_tokens: torch.Tensor,
    num_kv_tokens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    compress_ratio: int = 1,
) -> torch.Tensor:
    """Reference paged-decode logits from a dequantized KV cache."""
    batch_size, next_n = q.shape[0], q.shape[1]
    block_size = kv_cache.shape[1]
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    num_tokens_list = num_tokens.tolist()
    num_kv_tokens_list = num_kv_tokens.tolist()

    for i in range(batch_size):
        num_token, num_kv_token = num_tokens_list[i], num_kv_tokens_list[i]
        q_offsets = torch.arange(num_token - next_n, num_token, device=q.device)
        row = slice(i * next_n, (i + 1) * next_n)
        for block_rk in range((num_kv_token + block_size - 1) // block_size):
            block_start = block_rk * block_size
            block_end = min(block_start + block_size, max_model_len)
            block_width = block_end - block_start
            block = kv_cache[block_tables[i][block_rk]][:block_width]
            k_offsets = torch.arange(block_start, block_end, device=q.device)
            causal_mask = k_offsets[None, :] < (q_offsets[:, None] + 1) // compress_ratio
            mask = (k_offsets[None, :] < num_kv_token) & causal_mask
            scores = DSAVanillaIndexer.weighted_relu_scores(
                q[i], block.view(block_width, -1), weights[row], mask
            )
            logits[row, block_start:block_end] = torch.where(causal_mask, scores, float("-inf"))
    return logits


def paged_mqa_logits_quantized(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
    block_kv: int,
    epi_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Reference paged-decode logits with per-token KV scales."""
    batch, next_n = q.shape[0], q.shape[1]
    logits = torch.full(
        (batch * next_n, max_model_len), float("-inf"), device=q.device, dtype=epi_dtype
    )
    for b in range(batch):
        ctx_len = int(context_lens[b].item())
        q_positions = torch.arange(ctx_len - next_n, ctx_len, device=q.device)
        row = slice(b * next_n, (b + 1) * next_n)
        for blk_idx in range((ctx_len + block_kv - 1) // block_kv):
            phys = int(block_table[b, blk_idx].item())
            block_start = blk_idx * block_kv
            block_end = min(block_start + block_kv, max_model_len)
            block_width = block_end - block_start
            block = kv[phys][:block_width]
            scale = kv_scales[phys, :block_width].to(epi_dtype)
            k_positions = torch.arange(block_start, block_end, device=q.device)
            mask = (k_positions[None, :] < ctx_len) & (k_positions[None, :] <= q_positions[:, None])
            scores = DSAVanillaIndexer.weighted_relu_scores(
                q[b], block, weights[row], mask, epi_dtype
            )
            scores = scores * scale.unsqueeze(0)
            logits[row, block_start:block_end] = torch.where(
                mask, scores, torch.tensor(float("-inf"), device=q.device, dtype=epi_dtype)
            )
    return logits
