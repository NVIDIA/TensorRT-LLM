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

from typing import Optional

import torch


@torch.library.custom_op("auto_deploy::gather_tokens", mutates_args=("out",))
def gather_tokens(
    hidden_states: torch.Tensor,
    token_gather_indices: torch.Tensor,  # long tensor
    tokens_gather_info_host: torch.Tensor,  # int tensor
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather hidden states using token_gather_indices before LM head.

    Args:
        hidden_states: Hidden states tensor of shape [batch_size, 1, *other_dims] or
            [1, total_token_length, *other_dims]
        token_gather_indices: indices for gathering logits.
        tokens_gather_info_host: info for gathering logits.
        out: Optional pre-allocated output tensor to write into.
    Returns:
        Gathered and flattened hidden states [num_gathered_tokens, hidden]
    """
    # final shape is [total_tokens, *other_dims]
    bsz, sl, *other_dims = hidden_states.shape
    assert bsz == 1 or sl == 1, "expected batch size or sequence length to be 1"
    hidden_states = hidden_states.view(bsz * sl, *other_dims)

    # info object
    num_tokens_to_gather, gather_required = tokens_gather_info_host.tolist()

    if gather_required:
        result = hidden_states.index_select(0, token_gather_indices[:num_tokens_to_gather])
        num_tokens_final = num_tokens_to_gather
    else:
        num_tokens_final = bsz * sl
        if out is None:
            result = hidden_states.clone(memory_format=torch.contiguous_format)
        else:
            result = hidden_states
    if bsz == 1:
        result = result.view(1, num_tokens_final, *other_dims)
    else:
        result = result.view(num_tokens_final, 1, *other_dims)

    if out is not None:
        out.copy_(result)
        return out.new_empty(0)
    return result


@gather_tokens.register_fake
def gather_tokens_fake(
    hidden_states: torch.Tensor,
    token_gather_indices: torch.Tensor,
    tokens_gather_info_host: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if out is not None:
        return out.new_empty(0)
    # NOTE: shape is not correct in fake mode
    # see https://github.com/NVIDIA/TensorRT-LLM/issues/9878
    return torch.empty_like(hidden_states)
