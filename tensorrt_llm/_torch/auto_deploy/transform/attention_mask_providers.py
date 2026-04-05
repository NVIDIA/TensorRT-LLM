# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Model-specific attention mask providers."""

from __future__ import annotations

import torch

from .attention_mask_provider import AttentionMaskProviderRegistry


def _build_gemma4_token_type_mask(ctx, source_attn_node):
    q_meta = source_attn_node.args[0].meta["val"]
    batch_size, seq_len = q_meta.shape[:2]
    token_type_ids = ctx.add_or_retrieve_input(
        "token_type_ids",
        activate_arg=False,
        val=torch.zeros(batch_size, seq_len, dtype=torch.int64),
    )
    zeros = ctx.gm.graph.call_function(torch.ops.aten.zeros_like.default, args=(token_type_ids,))
    non_text = ctx.gm.graph.call_function(torch.ops.aten.ne.Scalar, args=(token_type_ids, 0))

    prev_token_types = ctx.gm.graph.call_function(
        torch.ops.aten.slice.Tensor, args=(token_type_ids, 1, 0, seq_len - 1)
    )
    prev_token_types = ctx.gm.graph.call_function(
        torch.ops.aten.cat.default,
        args=(
            [
                ctx.gm.graph.call_function(torch.ops.aten.slice.Tensor, args=(zeros, 1, 0, 1)),
                prev_token_types,
            ],
            1,
        ),
    )

    changed_type = ctx.gm.graph.call_function(
        torch.ops.aten.ne.Tensor, args=(token_type_ids, prev_token_types)
    )
    blob_starts = ctx.gm.graph.call_function(
        torch.ops.aten.bitwise_and.Tensor, args=(non_text, changed_type)
    )
    blob_ids = ctx.gm.graph.call_function(
        torch.ops.aten.cumsum.default,
        args=(
            ctx.gm.graph.call_function(torch.ops.aten.to.dtype, args=(blob_starts, torch.int64)),
            1,
        ),
    )
    token_blob_ids = ctx.gm.graph.call_function(
        torch.ops.aten.where.self, args=(non_text, blob_ids, zeros)
    )

    blob_q = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(token_blob_ids, 2))
    blob_k = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(token_blob_ids, 1))
    same_blob = ctx.gm.graph.call_function(torch.ops.aten.eq.Tensor, args=(blob_q, blob_k))
    media_query = ctx.gm.graph.call_function(torch.ops.aten.ne.Scalar, args=(blob_q, 0))
    bidirectional_media = ctx.gm.graph.call_function(
        torch.ops.aten.bitwise_and.Tensor, args=(same_blob, media_query)
    )

    positions = ctx.gm.graph.call_function(torch.ops.aten.arange.default, args=(seq_len,))
    pos_q = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(positions, 0))
    pos_k = ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(positions, 1))
    causal_mask = ctx.gm.graph.call_function(torch.ops.aten.le.Tensor, args=(pos_q, pos_k))
    causal_mask = ctx.gm.graph.call_function(
        torch.ops.aten.unsqueeze.default, args=(causal_mask, 0)
    )

    combined = ctx.gm.graph.call_function(
        torch.ops.aten.bitwise_or.Tensor, args=(causal_mask, bidirectional_media)
    )
    return ctx.gm.graph.call_function(torch.ops.aten.unsqueeze.default, args=(combined, 1))


def _build_or_reuse_gemma4_token_type_mask(ctx, source_attn_node):
    return ctx.get_or_create_cached_node(
        "gemma4_token_type_ids_mask",
        lambda: _build_gemma4_token_type_mask(ctx, source_attn_node),
    )


@AttentionMaskProviderRegistry.register("gemma4", "torch")
def _gemma4_torch_paged_mask_provider(ctx, source_attn_node):
    return _build_or_reuse_gemma4_token_type_mask(ctx, source_attn_node)


@AttentionMaskProviderRegistry.register("gemma4", "triton_paged")
def _gemma4_triton_paged_mask_provider(ctx, source_attn_node):
    return _build_or_reuse_gemma4_token_type_mask(ctx, source_attn_node)


@AttentionMaskProviderRegistry.register("gemma4", "torch_attention")
def _gemma4_torch_attention_mask_provider(ctx, source_attn_node):
    return _build_or_reuse_gemma4_token_type_mask(ctx, source_attn_node)
