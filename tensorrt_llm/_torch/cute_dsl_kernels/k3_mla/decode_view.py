# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch


def k3_mla_decode_view(attn, meta, num_tokens: int):
    """The per-layer inputs of trtllm::k3_mla_qkv and trtllm::k3_mla_attn(_vb)_out for a bf16 decode step of R
    generation requests of T = num_tokens / R tokens each (no context requests, R <= 8, T <= 8), or the reason it
    does not apply (a string; never raises). The dict: ``pool`` (flat bf16 pool), ``row_stride``, ``page_table``
    (int32 [R, W], request i's pages in row i: a strided view of ``kv_cache_block_offsets``, row stride
    ``page_table.stride(0)``), ``page_offset``, ``seq_len`` (int32 [R], the kv lengths including the step's tokens),
    ``softmax_scale``, ``num_requests`` (R) and ``tokens_per_request`` (T). The page table and lengths are the
    host-filled metadata buffers (``kv_cache_block_offsets``, ``kv_lens_cuda_runtime``), which the kernels may read
    before their grid wait. T is taken as uniform, as the engine pads every generation request to the same number of
    tokens."""
    from tensorrt_llm._torch.cute_dsl_kernels.k3_mla.op import MAX_REQUEST_TOKENS, MAX_REQUESTS

    num_requests = meta.num_generations
    if (
        meta.num_contexts != 0
        or not 0 < num_requests <= MAX_REQUESTS
        or num_tokens % num_requests
        or not 0 < num_tokens // num_requests <= MAX_REQUEST_TOKENS
    ):
        return (
            f"{meta.num_contexts} context / {num_requests} generation requests, {num_tokens} tokens"
        )
    if (
        meta.beam_width != 1
        or getattr(meta, "is_spec_dec_tree", False)
        or getattr(meta, "is_spec_dec_dynamic_tree", False)
    ):
        return f"beam width {meta.beam_width} or a tree speculative mask"
    if attn.num_heads % 6 or attn.kv_lora_rank != 512 or attn.qk_rope_head_dim != 64:
        return f"heads {attn.num_heads}, latent {attn.kv_lora_rank}, rope {attn.qk_rope_head_dim}"
    if meta.tokens_per_block != 64 or meta.helix_position_offsets is not None:
        return f"page {meta.tokens_per_block}, helix {meta.helix_position_offsets is not None}"
    if meta.kv_cache_manager is None or meta.kv_cache_block_offsets is None:
        return "no KV cache manager / block offsets"
    kv_pool = meta.kv_cache_manager.get_buffers(attn.layer_idx)
    if kv_pool.dtype != torch.bfloat16:
        return f"kv cache {kv_pool.dtype}"
    packed_block = 1
    for size in kv_pool.shape[1:]:
        packed_block *= size
    block_stride = kv_pool.stride(0)
    layers_in_pool = block_stride // packed_block if packed_block else 1
    layer_in_pool = 0
    if layers_in_pool > 1 and block_stride == layers_in_pool * packed_block:
        layer_in_pool = kv_pool.storage_offset() // packed_block
        kv_pool = kv_pool.as_strided(
            (kv_pool.shape[0] * layers_in_pool, *kv_pool.shape[1:]),
            (packed_block, *kv_pool.stride()[1:]),
            0,
        )
    if (
        kv_pool.dim() != 5
        or kv_pool.shape[1] != 1
        or kv_pool.shape[3] != 1
        or kv_pool.shape[2] != 64
    ):
        return f"pool layout {tuple(kv_pool.shape)}"
    if not (kv_pool.is_contiguous() and kv_pool.stride(2) >= 576 and kv_pool.stride(2) % 8 == 0):
        return f"pool strides {kv_pool.stride()}"
    pool_idx = int(meta.host_kv_cache_pool_mapping[attn.get_local_layer_idx(meta), 0])
    gen = slice(meta.num_contexts, meta.num_contexts + num_requests)
    page_table = meta.kv_cache_block_offsets[pool_idx, gen, 0, :]
    seq_len = meta.kv_lens_cuda_runtime[gen]
    if page_table.dtype != torch.int32 or seq_len.dtype != torch.int32:
        return f"page table {page_table.dtype} / length {seq_len.dtype} not int32"
    if (
        page_table.shape[0] != num_requests
        or seq_len.shape[0] != num_requests
        or page_table.stride(1) != 1
    ):
        return f"page table {tuple(page_table.shape)} strides {page_table.stride()} / lengths {tuple(seq_len.shape)}"
    softmax_scale = float(
        1.0 / (math.sqrt(attn.qk_nope_head_dim + attn.qk_rope_head_dim) * attn.q_scaling)
    )
    return dict(
        pool=kv_pool.view(-1),
        row_stride=kv_pool.stride(2),
        page_table=page_table,
        page_offset=int(layer_in_pool),
        seq_len=seq_len,
        softmax_scale=softmax_scale,
        num_requests=num_requests,
        tokens_per_request=num_tokens // num_requests,
    )
