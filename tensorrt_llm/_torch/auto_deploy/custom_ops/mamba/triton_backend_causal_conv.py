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

"""Triton-backed cached causal conv1d custom ops and attention descriptor.

This mirrors `cuda_backend_causal_conv.py` but uses Triton kernels instead of CUDA:
- Prefill uses Triton `causal_conv1d_fn`
- Decode uses Triton `causal_conv1d_update`

The flattened cached op integrates with the auto_deploy attention interface
and updates a slot-indexed convolution state cache internally.
"""

from typing import List, Optional

import torch

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)

from ..attention_interface import AttentionRegistry, MHACallable
from .causal_conv_common import BaseCausalConvDescriptor


@torch.library.custom_op("auto_deploy::triton_cached_causal_conv1d", mutates_args={"input"})
def _triton_cached_causal_conv1d(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k] but we expect depthwise use: [c_in, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    conv_state_cache: torch.Tensor,  # [max_batch_size, c_in, k-1]
    # CONSTANTS
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
    activation: Optional[str],
) -> None:
    """Flattened cached causal conv that respects slot-indexed state caches (Triton backend).

    Supports two layouts from the attention interface:
    - Generate-only: input is [b, 1, c_in]. We'll gather caches using slot_idx[:b].
    - Flattened context/mixed: input is [1, total_s, c_in] and seq_len/seq_start
      describe per-sequence segments. We'll process each segment and scatter final states to caches.

    NOTE: This op modifies `input` in-place.
    """
    b, s = input.shape[:2]

    num_prefill, num_prefill_tokens, num_decode = batch_info_host.tolist()
    num_seq = num_prefill + num_decode
    num_total_tokens = num_prefill_tokens + num_decode

    # Flatten tokens
    bs = b * s
    inp_flat = input.reshape(bs, *input.shape[2:])  # [total_s, C_in]

    # Prepare weight as [dim, width] (depthwise)
    if weight.ndim == 3:
        assert weight.shape[-2] == 1
        w2d = weight.squeeze(-2)
    else:
        w2d = weight

    # PREFILL: concatenate all prefill tokens and run one varlen forward
    if num_prefill > 0:
        # x_varlen: (dim, cu_seq_len)
        x_varlen = inp_flat[:num_prefill_tokens].transpose(0, 1).contiguous()

        prefill_cu_seqlen = cu_seqlen[: num_prefill + 1]
        seq_lens_cpu = seq_len[:num_prefill].tolist()

        # Run varlen conv; updates conv_state_cache in-place per cache_indices
        # Note: Triton kernel returns a new tensor (not in-place like CUDA)
        y_varlen = causal_conv1d_fn(
            x_varlen,
            w2d,
            bias,
            conv_state_cache,
            prefill_cu_seqlen,
            seq_lens_cpu,
            cache_indices=slot_idx[:num_prefill].to(torch.int32),
            has_initial_state=use_initial_states[:num_prefill],
            activation=activation,
            pad_slot_id=PAD_SLOT_ID,
        )  # (dim, total_prefill_tokens)
        # Scatter outputs back to input buffer
        inp_flat[:num_prefill_tokens] = y_varlen.transpose(0, 1)

    # DECODE: batch update for single-token sequences
    if num_decode > 0:
        x_decode = inp_flat[num_prefill_tokens:num_total_tokens]  # [num_decode, C_in]

        # Note: Triton causal_conv1d_update returns a new tensor (not in-place like CUDA version)
        # so we need to capture the output and write it back
        y_decode = causal_conv1d_update(
            x_decode,  # [batch, dim]
            conv_state_cache,
            w2d,
            bias,
            activation=activation,
            cache_seqlens=None,
            conv_state_indices=slot_idx[num_prefill:num_seq].to(torch.int32),
            pad_slot_id=PAD_SLOT_ID,
        )
        inp_flat[num_prefill_tokens:num_total_tokens] = y_decode


@_triton_cached_causal_conv1d.register_fake
def _triton_cached_causal_conv1d_fake(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k] but we expect depthwise use: [c_in, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
    seq_len: torch.Tensor,
    cu_seqlen: torch.Tensor,
    slot_idx: torch.Tensor,
    use_initial_states: torch.Tensor,
    # EXTRA METADATA
    #
    # CACHES
    conv_state_cache: torch.Tensor,  # [max_batch_size, c_in, k-1]
    # CONSTANTS
    stride: int,
    padding: int,
    dilation: int,
    groups: int,
    padding_mode: str,
    activation: Optional[str],
) -> None:
    pass


def triton_cached_causal_conv1d_wrapper(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    torch.ops.auto_deploy.triton_cached_causal_conv1d(input, *args, **kwargs)
    return input


@AttentionRegistry.register("triton_causal_conv")
class TritonBackendCausalConv(BaseCausalConvDescriptor):
    """Triton-backed causal conv1d attention descriptor.

    Inherits shared methods from BaseCausalConvDescriptor.
    Overrides get_standard_metadata_args to include seq_len (used directly by Triton kernel).
    """

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "seq_len", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return triton_cached_causal_conv1d_wrapper
