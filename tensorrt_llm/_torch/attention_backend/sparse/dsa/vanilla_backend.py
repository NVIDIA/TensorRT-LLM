# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vanilla correctness backend for DeepSeek Sparse Attention."""

import math
from dataclasses import replace
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    merge_attention_forward_args,
)
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention, VanillaAttentionMetadata

from .indexer import Indexer


class DSAVanillaIndexer:
    """FP32 golden for DSA indexer scoring."""

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    @property
    def uses_fp4(self) -> bool:
        return self.indexer.indexer_k_dtype == "fp4"

    def project_query(
        self,
        qr: torch.Tensor,
        position_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        rope_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        fp4_prep: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_tokens = qr.shape[0]
        q = F.linear(qr, self.indexer.wq_b.weight)
        q = q.view(num_tokens, self.indexer.n_heads, self.indexer.head_dim).unsqueeze(0)
        rope_fn(q[..., : self.indexer.rope_dim], freqs_cis[position_ids.long()])
        q = q.squeeze(0)
        return fp4_prep(q) if fp4_prep is not None else q

    def token_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights = F.linear(hidden_states, self.indexer.weights_proj.weight)
        return weights.float() * (self.indexer.n_heads**-0.5)

    def scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        head_scores = torch.einsum("hd,kd->hk", q.float(), k.float())
        head_scores = F.relu(head_scores) * self.indexer.softmax_scale
        return (head_scores * weights.float().unsqueeze(-1)).sum(dim=0)

    @staticmethod
    def topk(scores: torch.Tensor, topk: int) -> torch.Tensor:
        indices = torch.full((topk,), -1, dtype=torch.int32, device=scores.device)
        if scores.numel() > 0:
            valid_topk = min(topk, scores.numel())
            indices[:valid_topk] = torch.topk(scores.float(), valid_topk).indices.to(torch.int32)
        return indices


class DSAVanillaAttention(VanillaAttention):
    Metadata = VanillaAttentionMetadata

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        metadata: VanillaAttentionMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        del q, k, metadata
        sparse_backend_args = forward_args.sparse_backend_args
        topk_indices = sparse_backend_args.topk_indices if sparse_backend_args is not None else None
        return topk_indices, None

    @staticmethod
    def _load_latent_cache(
        kv_cache: torch.Tensor,
        block_ids: list[int],
        kv_len: int,
        kv_layout: str,
    ) -> torch.Tensor:
        if kv_layout == "NHD":
            tokens_per_block = kv_cache.shape[2]
        elif kv_layout == "HND":
            tokens_per_block = kv_cache.shape[3]
        else:
            raise ValueError(f"Unsupported KV cache layout: {kv_layout}")

        valid_block_ids = [block_id for block_id in block_ids if block_id != -1]
        num_required_blocks = math.ceil(kv_len / tokens_per_block)
        if len(valid_block_ids) < num_required_blocks:
            raise ValueError(
                f"DSA cache has {len(valid_block_ids)} blocks, but "
                f"{num_required_blocks} are required for {kv_len} tokens"
            )

        chunks = []
        remaining = kv_len
        for block_id in valid_block_ids[:num_required_blocks]:
            num_tokens = min(tokens_per_block, remaining)
            if kv_layout == "NHD":
                chunk = kv_cache[block_id, 0, :num_tokens, 0, :]
            else:
                chunk = kv_cache[block_id, 0, 0, :num_tokens, :]
            chunks.append(chunk)
            remaining -= num_tokens
        return torch.cat(chunks, dim=0)

    def _forward_sparse(
        self,
        fused_q: torch.Tensor,
        metadata: VanillaAttentionMetadata,
        latent_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attention_input_type: AttentionInputType,
    ) -> torch.Tensor:
        if attention_input_type == AttentionInputType.context_only:
            seq_start, seq_end = 0, metadata.num_contexts
        elif attention_input_type == AttentionInputType.generation_only:
            seq_start, seq_end = metadata.num_contexts, metadata.num_seqs
        else:
            raise ValueError("DSA requires a context-only or generation-only input")

        phase_seq_lens = metadata.seq_lens.tolist()[seq_start:seq_end]
        num_phase_tokens = sum(phase_seq_lens)
        fused_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        expected_q_shape = (num_phase_tokens, self.num_heads * fused_head_dim)
        if fused_q.shape != expected_q_shape:
            raise ValueError(
                f"DSA query must have shape {expected_q_shape}, got {tuple(fused_q.shape)}"
            )
        if latent_cache.shape != (num_phase_tokens, fused_head_dim):
            raise ValueError(
                "DSA latent cache must have shape "
                f"[{num_phase_tokens}, {fused_head_dim}], got {tuple(latent_cache.shape)}"
            )
        if topk_indices.ndim != 2 or topk_indices.shape[0] != num_phase_tokens:
            raise ValueError(
                "DSA top-k indices must have shape [num_phase_tokens, top_k], got "
                f"{tuple(topk_indices.shape)}"
            )
        if topk_indices.dtype != torch.int32:
            raise ValueError(f"DSA top-k indices must have dtype int32, got {topk_indices.dtype}")

        phase_past_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq[seq_start:seq_end]
        valid_mask = topk_indices >= 0
        if torch.any(topk_indices < -1):
            raise ValueError("DSA top-k indices may only use -1 as padding")
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("Every DSA query token must select at least one KV token")

        causal_limits = torch.cat(
            [
                torch.arange(
                    int(past),
                    int(past) + q_len,
                    dtype=topk_indices.dtype,
                    device=topk_indices.device,
                )
                for past, q_len in zip(phase_past_tokens, phase_seq_lens, strict=True)
            ]
        )
        if torch.any(valid_mask & (topk_indices > causal_limits.unsqueeze(1))):
            raise ValueError("DSA top-k index selects a future token")

        from ...utils import append_mla_latent_cache

        kv_cache = append_mla_latent_cache(
            metadata.kv_cache_manager,
            self.layer_idx,
            metadata.request_ids[seq_start:seq_end],
            phase_seq_lens,
            phase_past_tokens,
            latent_cache,
            kv_layout=metadata.kv_layout,
        )

        q = fused_q.view(num_phase_tokens, self.num_heads, fused_head_dim)
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        scale = 1.0 / (math.sqrt(qk_head_dim) * (self.q_scaling or 1.0))
        outputs = []
        token_offset = 0
        for phase_idx, q_len in enumerate(phase_seq_lens):
            seq_idx = seq_start + phase_idx
            kv_len = int(phase_past_tokens[phase_idx]) + q_len
            latent = self._load_latent_cache(
                kv_cache,
                metadata.block_ids_per_seq[seq_idx],
                kv_len,
                metadata.kv_layout,
            ).to(q.dtype)
            per_token_outputs = []
            for token_idx in range(q_len):
                row = topk_indices[token_offset + token_idx]
                selected = row[row >= 0].to(device=q.device, dtype=torch.long)
                per_token_outputs.append(
                    self._selected_mla_attention(
                        q[token_offset + token_idx],
                        latent.index_select(0, selected),
                        value_dim=self.kv_lora_rank,
                        scale=scale,
                    )
                )
            outputs.append(
                torch.stack(per_token_outputs).reshape(q_len, self.num_heads * self.kv_lora_rank)
            )
            token_offset += q_len
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: VanillaAttentionMetadata,
        forward_args: Optional[AttentionForwardArgs] = None,
        **kwargs,
    ) -> torch.Tensor:
        forward_args = merge_attention_forward_args(forward_args, kwargs)
        if not self.is_mla_enable:
            raise ValueError("DSAVanillaAttention requires MLA parameters")
        if metadata.multi_item_part_lens is not None:
            raise ValueError("DSA Vanilla attention does not support multi-item scoring")
        if metadata.kv_cache_manager is None:
            raise ValueError("DSA Vanilla attention requires a KV cache manager")
        if forward_args.latent_cache is None:
            raise ValueError("DSA Vanilla attention requires latent_cache")
        if k is not None or v is not None:
            raise ValueError("DSA Vanilla attention expects absorbed queries without K/V")

        sparse_attn_indices, sparse_attn_offsets = self.sparse_attn_predict(
            q, k, metadata, forward_args
        )
        if sparse_attn_indices is None:
            raise ValueError("DSA Vanilla attention requires sparse attention indices")
        forward_args.sparse_runtime_params = replace(
            forward_args.sparse_runtime_params,
            sparse_attn_indices=sparse_attn_indices,
            sparse_attn_offsets=sparse_attn_offsets,
        )
        return self._forward_sparse(
            q,
            metadata,
            forward_args.latent_cache,
            sparse_attn_indices,
            forward_args.attention_input_type,
        )


__all__ = ["DSAVanillaAttention", "DSAVanillaIndexer"]
