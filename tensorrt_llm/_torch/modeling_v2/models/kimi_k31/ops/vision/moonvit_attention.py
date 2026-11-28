# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoonViT's attention: bidirectional attention within each image over fused ``[q | k | v]`` rows, no KV cache."""

from __future__ import annotations

from typing import List, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention

#: Kimi K3.1's MoonViT attention: 12 heads of 128.
HEADS, HEAD_DIM = 12, 128

#: The thop attention arguments the call holds fixed: the no-KV-cache mode (an encoder: no cache addressing),
#: one context sequence per image, the fused ``[q | k | v]`` rows already rotated (no rope in the op), bidirectional
#: within each sequence (mask_type 0), softmax scale head_dim^-0.5, unquantized.
_VISION_ATTN = dict(
    k=None,
    v=None,
    output_sf=None,
    max_context_q_len_override=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    latent_cache=None,
    q_pe=None,
    block_ids_per_seq=None,
    attention_sinks=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    # Unused without a KV cache; the engine backend's value for a cache-less metadata.
    tokens_per_block=64,
    beam_width=1,
    mask_type=0,
    quant_mode=0,
    q_scaling=1.0,
    position_embedding_type=0,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    use_paged_context_fmha=False,
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
)


def vision_step(
    seg_lens: List[int], max_seq_len: int, max_num_requests: int, device, workspace: torch.Tensor
) -> dict:
    """The batch state of one native attention call: one context sequence per grid, no KV cache. The capacities
    (``max_seq_len``, also the window and the longest context, and ``max_num_requests``) are the tower's fixed ones:
    the attention op caches one native op per configuration, without eviction, so a per-call value would build and
    keep a new op for every new image batch.

    A serving step never exceeds them: admission refuses grids beyond ``max_seq_len`` (the tower's
    ``validate_multimodal_data``) and ``moonvit_attention`` splits a call into groups of ``max_num_requests`` grids."""
    if max(seg_lens) > max_seq_len or len(seg_lens) > max_num_requests:
        raise ValueError(
            f"A MoonViT call of {len(seg_lens)} grids, the longest {max(seg_lens)} patches, exceeds the tower's "
            f"attention capacity: {max_num_requests} grids of at most {max_seq_len} patches"
        )
    total = sum(seg_lens)
    lens_dev = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    lens_host = torch.tensor(seg_lens, dtype=torch.int32)
    return dict(
        workspace=workspace,
        sequence_length=lens_dev,
        host_past_key_value_lengths=lens_host,
        host_total_kv_lens=torch.tensor([total, 0], dtype=torch.int32),
        context_lengths=lens_dev,
        host_context_lengths=lens_host,
        host_request_types=torch.zeros(len(seg_lens), dtype=torch.int32),
        max_num_requests=max_num_requests,
        max_context_length=max_seq_len,
        max_seq_len=max_seq_len,
        attention_window_size=max_seq_len,
        num_contexts=len(seg_lens),
        num_ctx_tokens=total,
    )


class MoonViTAttention(OpWrapper):
    """``out[s] = softmax(q_s k_s^T / sqrt(head_dim)) v_s`` for every image ``s``: the rows of ``qkv`` (``[N, 3 *
    heads * head_dim]``, the images' patches concatenated, ``seg_lens`` patches each) attend to their own image only,
    every head to its own k / v head. Writes ``out`` (``[N, heads * head_dim]``, ``qkv``'s dtype) and returns it.

    ``max_seq_len`` / ``max_num_requests`` are the fixed capacities of every native call (``vision_step``); a call with
    more images than ``max_num_requests`` runs as several native calls of at most that many, which the per-image
    attention makes the same rows. ``workspace`` is grown in place by the op.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one 64x64 image, K3.1 capacities (the processor's largest image, 67320 patches)",
            spec=dict(seg_lens=(4096,), heads=HEADS, head_dim=HEAD_DIM, capacity=67320),
        ),
        Cell(
            why="three images of one call, the smallest 2x2 among them",
            spec=dict(seg_lens=(4, 1536, 680), heads=HEADS, head_dim=HEAD_DIM, capacity=67320),
        ),
        Cell(
            why="one 128x128 image: a long sequence",
            spec=dict(seg_lens=(16384,), heads=HEADS, head_dim=HEAD_DIM, capacity=67320),
        ),
    )
    note = """
    The catalog ``thop_attention`` (the native attention op the engine's TRTLLM backend runs through its fallback
    FMHA for a cache-less encoder) in its no-KV-cache mode: one context sequence per image, the rows already rotated
    (no rope in the op), mask type 0 (bidirectional), softmax scale head_dim^-0.5, unquantized. The op caches one
    native op per configuration without eviction; the fixed capacities (and one layer index) keep that to one op for
    every image batch. The capacities do not change the result. Eager, never captured.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        out: torch.Tensor,
        seg_lens: Sequence[int],
        max_seq_len: int,
        max_num_requests: int,
        workspace: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        seg_lens = list(seg_lens)
        start = 0
        # A step can carry any number of images (one per placeholder run of every context request): groups of at most
        # max_num_requests images, each its own native call, keep the capacities fixed.
        for first in range(0, len(seg_lens), max_num_requests):
            group = seg_lens[first : first + max_num_requests]
            rows = slice(start, start + sum(group))
            thop_attention(
                q=qkv[rows],
                output=out[rows],
                local_layer_idx=0,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                head_size=head_dim,
                **vision_step(group, max_seq_len, max_num_requests, qkv.device, workspace),
                **_VISION_ATTN,
            )
            start = rows.stop
        return out

    def reference(
        self,
        qkv: torch.Tensor,
        out: torch.Tensor,
        seg_lens: Sequence[int],
        max_seq_len: int,
        max_num_requests: int,
        workspace: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """fp32 softmax attention per image and head, rounded once; ``out``, the capacities and ``workspace`` are
        ignored."""
        width = num_heads * head_dim
        rows: List[torch.Tensor] = []
        start = 0
        for n in seg_lens:
            q, k, v = (
                qkv[start : start + n, i * width : (i + 1) * width]
                .float()
                .view(n, num_heads, head_dim)
                .transpose(0, 1)
                for i in range(3)
            )
            scores = torch.matmul(q, k.transpose(1, 2)) * head_dim**-0.5
            rows.append(
                torch.matmul(torch.softmax(scores, dim=-1), v).transpose(0, 1).reshape(n, width)
            )
            start += n
        return torch.cat(rows).to(qkv.dtype)

    def is_valid(
        self,
        qkv: torch.Tensor,
        out: torch.Tensor,
        seg_lens: Sequence[int],
        max_seq_len: int,
        max_num_requests: int,
        workspace: torch.Tensor,
        num_heads: int,
        head_dim: int,
    ) -> None:
        if qkv.shape[0] != sum(seg_lens) or qkv.shape[1] != 3 * num_heads * head_dim:
            raise ValueError("qkv must hold [q | k | v] rows of every patch of seg_lens")


moonvit_attention = MoonViTAttention()
