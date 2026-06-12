# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Speculative-decoding × KV-cache cross-cutting operations.

Functions in this module perform GPU-side KV cache relocation driven by
speculative decoding acceptance decisions.  They are called from
KVCacheManager.update_resources() and are kept separate so that the
conceptual grouping is explicit.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from ..llm_request import LlmRequest, LlmRequestState
from ..scheduler import ScheduledRequests

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
    from tensorrt_llm.llmapi.llm_args import DecodingBaseConfig

from ....mapping import Mapping


def get_pp_layers(
    num_layers: int,
    mapping: Mapping,
    spec_config: Optional["DecodingBaseConfig"] = None,
    layer_mask: Optional[List[bool]] = None,
) -> Tuple[List[int], int]:
    from ...speculative.utils import get_num_spec_layers

    total_num_layers = num_layers
    if layer_mask is not None:
        assert sum(layer_mask) == num_layers, (
            f"The number of enabled layers in layer_mask ({sum(layer_mask)}) "
            f"must match the number of layers ({num_layers}) "
            f"in KV cache manager, but got layer_mask: {layer_mask}"
        )
        total_num_layers = len(layer_mask)
    pp_layers = mapping.pp_layers(total_num_layers)
    if layer_mask is not None:
        pp_layers = [i for i in pp_layers if layer_mask[i]]
    # Only add speculative layers when layer_mask is not provided.
    # When layer_mask is provided, the caller explicitly controls which layers
    # to include, so we should not add extra layers automatically.
    if spec_config is not None and layer_mask is None:
        num_spec_layers = get_num_spec_layers(spec_config)
        total_num_layers += num_spec_layers
        if mapping.is_last_pp_rank():
            pp_layers.extend(range(total_num_layers - num_spec_layers, total_num_layers))
    if len(pp_layers) == 0:
        # Don't support empty KV cache for now, provide at least 1 layer
        pp_layers.append(0)
    return pp_layers, total_num_layers


def _locate_accepted_draft_tokens(requests: List[LlmRequest]):
    num_accepted_draft_tokens = []
    accepted_draft_tokens_indices = []
    rewind_draft_token_separate_adjustments = []
    # for context requests, the py_num_accepted_draft_tokens = 0, and py_num_accepted_draft_tokens_indices = []
    for seq in requests:
        num_accepted_draft_tokens.append(seq.py_num_accepted_draft_tokens)
        rewind_draft_token_separate_adjustments.append(
            seq.py_rewind_draft_token_separate_adjustment
        )
        accepted_draft_tokens_indices.extend(seq.py_num_accepted_draft_tokens_indices)
    batch_size = len(requests)
    num_accepted_draft_tokens_offset = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    num_accepted_draft_tokens_offset[1:] = torch.cumsum(
        torch.tensor(num_accepted_draft_tokens, dtype=torch.int32), dim=0
    )
    accepted_draft_tokens_indices = torch.tensor(
        accepted_draft_tokens_indices, dtype=torch.int32, device="cuda"
    )
    rewind_draft_token_separate_adjustments = torch.tensor(
        rewind_draft_token_separate_adjustments, dtype=torch.int32, device="cuda"
    )
    return (
        num_accepted_draft_tokens_offset,
        accepted_draft_tokens_indices,
        rewind_draft_token_separate_adjustments,
    )


def _update_kv_cache_draft_token_location(
    cache_manager,
    scheduled_batch: ScheduledRequests,
    attn_metadata: "AttentionMetadata",
    kv_cache_dtype_byte_size: float,
):
    run_kv_cache_relocation = False
    for request in scheduled_batch.generation_requests:
        if request.state != LlmRequestState.GENERATION_COMPLETE:
            if (
                request.py_num_accepted_draft_tokens > 0
                and len(request.py_num_accepted_draft_tokens_indices) > 0
            ):
                run_kv_cache_relocation = True
    if not run_kv_cache_relocation:
        return
    requests = scheduled_batch.all_requests()
    (
        accepted_draft_token_offsets,
        packed_accepted_draft_tokens_indices,
        rewind_draft_token_separate_adjustments,
    ) = _locate_accepted_draft_tokens(requests)
    past_key_value_lengths = attn_metadata.kv_lens_cuda[: len(requests)]
    if (
        attn_metadata.kv_cache_block_offsets is not None
        and attn_metadata.host_kv_cache_pool_pointers is not None
        and attn_metadata.host_kv_cache_pool_mapping is not None
    ):
        use_paged_kv_cache = True
    else:
        use_paged_kv_cache = False
    assert use_paged_kv_cache, "Only paged kv cache is supported"
    assert len(cache_manager.max_attention_window_vec) == 1, (
        "Currently, only one max attention window size is supported."
    )

    if use_paged_kv_cache:
        torch.ops.tensorrt_llm.update_kv_cache_draft_token_location(
            accepted_draft_token_offsets,
            packed_accepted_draft_tokens_indices,
            past_key_value_lengths,
            True,
            cache_manager.num_layers,
            cache_manager.num_kv_heads,
            int(cache_manager.head_dim * kv_cache_dtype_byte_size),
            cache_manager.max_total_draft_tokens,
            cache_manager.max_attention_window_vec[0],
            rewind_draft_token_separate_adjustments,
            None,
            cache_manager.kv_cache_pool_pointers,
            attn_metadata.kv_cache_block_offsets,
            cache_manager.max_blocks_per_seq,
            cache_manager.tokens_per_block,
            None,
        )
