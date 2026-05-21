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

import torch

from ..attention_interface import BatchInfo


@torch.library.custom_op("auto_deploy::gather_tokens", mutates_args=())
def gather_tokens(
    hidden_states: torch.Tensor,
    token_gather_indices: torch.Tensor,  # long tensor
    batch_info_host: torch.Tensor,  # int tensor (BatchInfo)
) -> torch.Tensor:
    """Gather hidden states using token_gather_indices before LM head.

    Args:
        hidden_states: Hidden states tensor of shape [batch_size, 1, *other_dims],
            [1, total_token_length, *other_dims], or [batch_size, seq_len, *other_dims].
        token_gather_indices: indices for gathering logits.
        batch_info_host: BatchInfo tensor containing tokens_gather_info.
    Returns:
        Gathered and flattened hidden states [num_gathered_tokens, hidden]
    """
    bsz, sl, *other_dims = hidden_states.shape
    hidden_states = hidden_states.view(bsz * sl, *other_dims)

    batch_info = BatchInfo(batch_info_host)
    num_tokens_to_gather = batch_info.get_num_tokens_to_gather()
    gather_required = batch_info.is_gather_required()

    if gather_required:
        out = hidden_states.index_select(0, token_gather_indices[:num_tokens_to_gather])
        num_tokens_final = num_tokens_to_gather
    else:
        out = hidden_states.clone(memory_format=torch.contiguous_format)
        num_tokens_final = bsz * sl
    # Generate-only batches use [batch, 1, ...] and need to preserve batch-major layout for the
    # downstream squeeze. Any shape with seq_len > 1 is treated as a flattened token batch.
    if sl == 1 and bsz > 1:
        return out.view(num_tokens_final, 1, *other_dims)
    else:
        return out.view(1, num_tokens_final, *other_dims)


@gather_tokens.register_fake
def gather_tokens_fake(
    hidden_states: torch.Tensor,
    token_gather_indices: torch.Tensor,
    batch_info_host: torch.Tensor,
) -> torch.Tensor:
    # NOTE: shape is not correct in fake mode
    # see https://github.com/NVIDIA/TensorRT-LLM/issues/9878
    return torch.empty_like(hidden_states)
