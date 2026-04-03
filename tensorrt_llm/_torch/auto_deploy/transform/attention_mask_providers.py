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

import operator

import torch

from .attention_mask_provider import AttentionMaskProviderRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_type_ids_default_factory(seq_info):
    """Return a zeros tensor with the same (batch, seq) shape as ``input_ids``.

    Used as the default factory for ``token_type_ids`` so that initialization-time
    forward passes (e.g. ``resize_kv_cache``, CUDA-graph warmup) always receive a
    valid tensor even when no per-request multimodal data is present.

    ``get_arg("input_ids")`` already unflattens (input_ids is in _shapeable_args),
    so the result has shape [batch_size, seq_len].
    """
    input_ids_2d = seq_info.get_arg("input_ids")
    return torch.zeros(
        input_ids_2d.shape[0], input_ids_2d.shape[1], dtype=torch.int64, device=seq_info.device
    )


def _build_gemma4_token_type_mask(ctx, source_attn_node):
    q_meta = source_attn_node.args[0].meta["val"]
    batch_size, seq_len = q_meta.shape[:2]
    # Build a FakeTensor with the query's (possibly symbolic) batch/seq dims so that
    # the placeholder inherits the correct dynamic shape metadata.
    fake_val = q_meta.new_zeros(batch_size, seq_len, dtype=torch.int64)
    token_type_ids = ctx.add_or_retrieve_input(
        "token_type_ids",
        activate_arg=False,
    )
    # Set fake tensor meta directly — add_graph_input's static_shapes=True would
    # concretise symbolic dims, but we need them to stay dynamic.
    token_type_ids.meta["val"] = fake_val
    # Register a default factory so cm.named_args always includes token_type_ids
    # during initialization-time forward passes (resize_kv_cache, CUDA-graph warmup)
    # even when no multimodal per-request data is available.
    ctx.register_default_extra_arg("token_type_ids", _token_type_ids_default_factory)
    zeros = ctx.gm.graph.call_function(torch.ops.aten.zeros_like.default, args=(token_type_ids,))
    non_text = ctx.gm.graph.call_function(torch.ops.aten.ne.Scalar, args=(token_type_ids, 0))

    # Derive seq_len inside the graph so the symbolic variable survives recompile.
    g_seq_len = ctx.gm.graph.call_function(torch.ops.aten.sym_size.int, args=(token_type_ids, 1))
    g_seq_len_m1 = ctx.gm.graph.call_function(operator.sub, args=(g_seq_len, 1))
    prev_token_types = ctx.gm.graph.call_function(
        torch.ops.aten.slice.Tensor, args=(token_type_ids, 1, 0, g_seq_len_m1)
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

    # Derive 1D positions [seq_len] from token_type_ids to inherit correct device
    # during shape propagation (meta) and runtime (cuda).
    ones_2d = ctx.gm.graph.call_function(torch.ops.aten.ones_like.default, args=(token_type_ids,))
    # Take first row to get 1D [seq_len]
    ones_1d = ctx.gm.graph.call_function(torch.ops.aten.select.int, args=(ones_2d, 0, 0))
    positions = ctx.gm.graph.call_function(torch.ops.aten.cumsum.default, args=(ones_1d, 0))
    positions = ctx.gm.graph.call_function(torch.ops.aten.sub.Tensor, args=(positions, ones_1d))
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
