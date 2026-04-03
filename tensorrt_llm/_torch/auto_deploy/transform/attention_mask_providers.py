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

"""Model-specific attention mask providers.

Providers registered here add an optional ``custom_attn_mask`` graph input and
wire it to each ``torch_attention`` node.  The actual mask tensor is computed
**outside** the graph (e.g. in the VLM wrapper ``forward()``) and passed in at
runtime.  During warmup, text-only, and decode steps the wrapper passes
``None``, so the attention backend uses its fast causal kernel.
"""

from __future__ import annotations

from .attention_mask_provider import AttentionMaskProviderRegistry


def _add_custom_attn_mask_input(ctx, source_attn_node):
    """Add ``custom_attn_mask`` as an optional graph input (default ``None``).

    No mask computation nodes are inserted into the graph.  The mask is
    computed outside the graph by the model wrapper and supplied at runtime.
    """
    return ctx.add_or_retrieve_input(
        "custom_attn_mask",
        activate_arg=False,
        val=None,
    )


@AttentionMaskProviderRegistry.register("gemma4", "torch")
def _gemma4_torch_paged_mask_provider(ctx, source_attn_node):
    return _add_custom_attn_mask_input(ctx, source_attn_node)


@AttentionMaskProviderRegistry.register("gemma4", "triton_paged")
def _gemma4_triton_paged_mask_provider(ctx, source_attn_node):
    return _add_custom_attn_mask_input(ctx, source_attn_node)


@AttentionMaskProviderRegistry.register("gemma4", "torch_attention")
def _gemma4_torch_attention_mask_provider(ctx, source_attn_node):
    return _add_custom_attn_mask_input(ctx, source_attn_node)
