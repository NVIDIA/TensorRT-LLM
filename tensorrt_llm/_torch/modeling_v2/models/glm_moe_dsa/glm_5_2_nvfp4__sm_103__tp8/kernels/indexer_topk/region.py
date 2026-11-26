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
"""The generation region of ``Indexer.sparse_attn_indexer`` on a decode step.

``GlmIndexer._fill_decode_topk`` asks these entries, in order, to fill the generation rows of
the top-k buffer. Each decides from host state before any device work, so a ``False`` return
leaves ``out`` untouched for the next entry and, last, for the built-in decode branch.
"""

from typing import TYPE_CHECKING

import torch

from ... import dispatch

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
    from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
        DSAtrtllmAttentionMetadata,
    )


def cross_rank_topk_decode(
    indexer: "Indexer",
    metadata: "DSAtrtllmAttentionMetadata",
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
    bridge=None,
) -> bool:
    """Run the context-partitioned (cross-rank) top-k of ``cross_rank_topk`` when selected.

    Returns True when it filled ``out`` (the generation rows of the top-k buffer). The decision
    reads host state that is identical on every rank of the tensor-parallel group. Among the
    selected variants the first that serves the call wins:

    1. ``cross_rank_topk_global_emit``: the cluster-merge top-k that also writes the call's
       bridge outputs (the shared top-k copy and the global remap) and records them on
       ``bridge``;
    2. ``indexer_layer_front``: the same region with its logits kernel launched as the rope's
       programmatic dependent; when it declines, the variants below decline too;
    3. ``cross_rank_topk_cluster_merge``: the cluster-merge top-k (rows byte-identical to the
       plain cross-rank top-k);
    4. ``cross_rank_topk`` itself.
    """
    if not dispatch.is_enabled("cross_rank_topk"):
        return False
    if dispatch.is_enabled("cross_rank_topk_global_emit") and dispatch.get_impl(
        "cross_rank_topk_global_emit", "cp_indexer_decode"
    )(indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out, bridge):
        return True
    # The same region with the logits kernel waiting at entry and launched as the rope's
    # programmatic dependent.
    if dispatch.is_enabled("indexer_layer_front") and dispatch.get_impl(
        "indexer_layer_front", "cp_indexer_decode"
    )(indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out):
        return True
    if dispatch.is_enabled("cross_rank_topk_cluster_merge") and dispatch.get_impl(
        "cross_rank_topk_cluster_merge", "cp_indexer_decode"
    )(indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out):
        return True
    return dispatch.get_impl("indexer_topk", "cp_indexer_decode")(
        indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out
    )


def indexer_topk_decode(
    indexer: "Indexer",
    metadata: "DSAtrtllmAttentionMetadata",
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    token_offset: int,
    num_gen_tokens: int,
    out: torch.Tensor,
) -> bool:
    """Run the ``indexer_topk`` decode top-k when it is selected.

    Returns True when it filled ``out`` (the generation rows of the top-k buffer). Its
    eligibility is checked from host state before any device work.
    """
    if not dispatch.is_enabled("indexer_topk"):
        return False
    return dispatch.get_impl("indexer_topk", "indexer_decode")(
        indexer, metadata, q_fp8, weights, token_offset, num_gen_tokens, out
    )
