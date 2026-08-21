# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from ...vanilla import VanillaAttention
from .common import _INIT_SCORE, _LOCAL_SCORE, write_kv_slots
from .triton_backend import MiniMaxM3SparseRuntimeBackend
from .triton_metadata import MiniMaxM3TritonSparseAttentionMetadata


class MiniMaxM3VanillaAttention(MiniMaxM3SparseRuntimeBackend):
    """Vanilla golden backend for MiniMax-M3 block-sparse attention."""

    @staticmethod
    def _gather_rows(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        slots = slots.to(device=cache.device, dtype=torch.long)
        if cache.ndim == 3:
            return cache.index_select(0, slots)
        if cache.ndim == 4:
            tokens_per_block = cache.shape[1]
            return cache[slots // tokens_per_block, slots % tokens_per_block]
        raise ValueError(f"MiniMax-M3 cache must be 3D or 4D, got {cache.ndim}D")

    def _selected_blocks(
        self,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        valid_length: int,
        idx_sm_scale: float,
    ) -> torch.Tensor:
        config = self.m3_config
        num_blocks = (valid_length + config.block_size - 1) // config.block_size
        token_scores = (
            torch.matmul(idx_q.to(torch.float32), idx_k[:valid_length, 0].to(torch.float32).T)
            * idx_sm_scale
        )
        block_scores = torch.stack(
            [
                token_scores[
                    :,
                    block_idx * config.block_size : min(
                        (block_idx + 1) * config.block_size, valid_length
                    ),
                ].amax(dim=-1)
                for block_idx in range(num_blocks)
            ],
            dim=-1,
        )
        if config.init_blocks:
            block_scores[:, : min(config.init_blocks, num_blocks)] = _INIT_SCORE
        if config.local_blocks:
            local_start = max(0, num_blocks - config.local_blocks)
            block_scores[:, local_start:] = _LOCAL_SCORE

        selected_per_index_head = block_scores.topk(k=min(config.topk, num_blocks), dim=-1).indices
        index_heads_per_kv_head = config.num_index_heads // config.num_kv_heads
        max_selected_blocks = selected_per_index_head.shape[1] * index_heads_per_kv_head
        selected_blocks = torch.full(
            (config.num_kv_heads, max_selected_blocks),
            -1,
            dtype=selected_per_index_head.dtype,
            device=selected_per_index_head.device,
        )
        for kv_head in range(config.num_kv_heads):
            block_ids = torch.unique(
                selected_per_index_head[
                    kv_head * index_heads_per_kv_head : (kv_head + 1) * index_heads_per_kv_head
                ].flatten()
            )
            selected_blocks[kv_head, : block_ids.shape[0]] = block_ids
        return selected_blocks

    def _single_token_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        idx_query: torch.Tensor,
        idx_keys: torch.Tensor,
        valid_length: int,
        sm_scale: float,
        idx_sm_scale: float,
    ) -> torch.Tensor:
        selected_blocks = self._selected_blocks(idx_query, idx_keys, valid_length, idx_sm_scale)
        return VanillaAttention._single_token_sparse_attn_forward(
            query,
            keys,
            values,
            selected_blocks,
            self.sparse_params.indices_block_size,
            sm_scale,
        )

    def forward_sparse(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        idx_q: torch.Tensor,
        idx_k: torch.Tensor,
        idx_v: Optional[torch.Tensor],
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        idx_k_cache: torch.Tensor,
        idx_v_cache: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor,
        m3_metadata: MiniMaxM3TritonSparseAttentionMetadata,
        sm_scale: Optional[float] = None,
        idx_sm_scale: Optional[float] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        config = self.m3_config
        num_tokens = q.shape[0]
        q = q.view(num_tokens, config.num_q_heads, config.head_dim)
        k = k.view(num_tokens, config.num_kv_heads, config.head_dim)
        v = v.view(num_tokens, config.num_kv_heads, config.head_dim)
        idx_q = idx_q.view(num_tokens, config.num_index_heads, config.sparse_index_dim)
        idx_k = idx_k.view(num_tokens, 1, config.sparse_index_dim)

        if self.disable_index_value:
            if idx_v is not None or idx_v_cache is not None:
                raise ValueError("disable_index_value=True requires index V inputs to be None")
        elif idx_v is None or idx_v_cache is None:
            raise ValueError("disable_index_value=False requires index V inputs")

        write_kv_slots(k_cache, out_cache_loc, k)
        write_kv_slots(v_cache, out_cache_loc, v)
        write_kv_slots(idx_k_cache, out_cache_loc, idx_k)
        if idx_v is not None and idx_v_cache is not None:
            write_kv_slots(
                idx_v_cache,
                out_cache_loc,
                idx_v.view(num_tokens, 1, config.sparse_index_dim),
            )

        sm_scale = sm_scale if sm_scale is not None else config.head_dim**-0.5
        idx_sm_scale = idx_sm_scale if idx_sm_scale is not None else config.sparse_index_dim**-0.5
        slot_rows = m3_metadata.req_to_token.index_select(0, m3_metadata.slot_ids.to(torch.long))
        seq_lens = m3_metadata.seq_lens.to("cpu").tolist()
        if m3_metadata.is_prefill:
            if m3_metadata.q_batch_row is None or m3_metadata.q_positions is None:
                raise ValueError("MiniMax-M3 prefill metadata is not prepared")
            q_batch_rows = m3_metadata.q_batch_row.to("cpu").tolist()
            q_positions = m3_metadata.q_positions.to("cpu").tolist()
        else:
            if num_tokens != len(seq_lens):
                raise ValueError("MiniMax-M3 decode requires one query token per request")
            q_batch_rows = list(range(num_tokens))
            q_positions = [seq_len - 1 for seq_len in seq_lens]

        results = []
        for token_idx, (batch_row, query_position) in enumerate(
            zip(q_batch_rows, q_positions, strict=True)
        ):
            valid_length = int(query_position) + 1
            if valid_length <= 0 or valid_length > int(seq_lens[batch_row]):
                raise ValueError("MiniMax-M3 query position is outside its KV sequence")
            slots = slot_rows[batch_row, :valid_length]
            keys = self._gather_rows(k_cache, slots)
            values = self._gather_rows(v_cache, slots)
            index_keys = self._gather_rows(idx_k_cache, slots)
            results.append(
                self._single_token_attention(
                    q[token_idx],
                    keys,
                    values,
                    idx_q[token_idx],
                    index_keys,
                    valid_length,
                    sm_scale,
                    idx_sm_scale,
                )
            )

        result = (
            torch.stack(results).reshape(num_tokens, config.num_q_heads * config.head_dim)
            if results
            else q.new_empty((0, config.num_q_heads * config.head_dim))
        )
        if output is None:
            return result.to(q.dtype)
        if output.shape != result.shape:
            raise ValueError(
                f"output must have shape {tuple(result.shape)}, got {tuple(output.shape)}"
            )
        output.copy_(result)
        return output


__all__ = ["MiniMaxM3VanillaAttention"]
