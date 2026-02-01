# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CUDA-backed cached causal conv1d custom ops and attention descriptor.

This mirrors `torch_backend_causal_conv.py` but reuses existing TRT-LLM CUDA
operators for performance:
- Prefill uses `torch.ops.trtllm.causal_conv1d_fwd`
- Decode uses `torch.ops.trtllm.causal_conv1d_update`

The flattened cached op integrates with the auto_deploy attention interface
and updates a slot-indexed convolution state cache internally.
"""

from typing import List, Optional

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from tensorrt_llm._torch.modules.mamba import PAD_SLOT_ID
from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    AttentionRegistry,
    CausalConvResourceHandler,
    Constant,
    MHACallable,
    ResourceHandlerDict,
)


@torch.library.custom_op("auto_deploy::cuda_cached_causal_conv1d", mutates_args={"input"})
def _cuda_cached_causal_conv1d(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k] but we expect depthwise use: [c_in, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
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
    """Flattened cached causal conv that respects slot-indexed state caches (CUDA backend).

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

        # Run varlen conv; updates conv_state_cache in-place per cache_indices
        y_varlen = causal_conv1d_fn(
            x_varlen,
            w2d,
            bias,
            query_start_loc=cu_seqlen[: num_prefill + 1],
            cache_indices=slot_idx[:num_prefill].to(torch.int32),
            has_initial_state=use_initial_states[:num_prefill],
            conv_states=conv_state_cache,
            activation=activation,
            pad_slot_id=PAD_SLOT_ID,
        )  # (dim, total_prefill_tokens)
        # Scatter outputs back to input buffer
        inp_flat[:num_prefill_tokens] = y_varlen.transpose(0, 1)

    # DECODE: batch update for single-token sequences
    if num_decode > 0:
        x_decode = inp_flat[num_prefill_tokens:num_total_tokens]  # [num_decode, C_in]

        causal_conv1d_update(
            x_decode,  # [batch, dim]
            conv_state_cache,
            w2d,
            bias,
            activation=activation,
            cache_seqlens=None,
            conv_state_indices=slot_idx[num_prefill:num_seq].to(torch.int32),
            pad_slot_id=PAD_SLOT_ID,
        )


@_cuda_cached_causal_conv1d.register_fake
def _cuda_cached_causal_conv1d_fake(
    # INPUTS (dense but may be flattened across sequences)
    input: torch.Tensor,  # [b, s, c_in]
    weight: torch.Tensor,  # [c_out, c_in/groups, k] but we expect depthwise use: [c_in, k]
    bias: Optional[torch.Tensor],
    # STANDARD METADATA
    batch_info_host: torch.Tensor,
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


def cuda_cached_causal_conv1d_wrapper(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    torch.ops.auto_deploy.cuda_cached_causal_conv1d(input, *args, **kwargs)
    return input


@AttentionRegistry.register("cuda_causal_conv")
class CudaBackendCausalConv(AttentionDescriptor):
    @classmethod
    def get_attention_layout(cls) -> AttentionLayout:
        # Hidden states follow [b, s, c]
        return "bsnd"

    @classmethod
    def get_num_qkv_args(cls) -> int:
        # torch_causal_conv1d signature has 3 relevant tensor arguments
        # TODO: bias can be optional!! How to handle None bias here?
        return 3

    @classmethod
    def get_source_attention_op(cls) -> OpOverloadPacket:
        return torch.ops.auto_deploy.torch_causal_conv1d.default

    @classmethod
    def get_cached_attention_op(cls) -> MHACallable:
        return cuda_cached_causal_conv1d_wrapper

    @classmethod
    def get_standard_metadata_args(cls) -> List[str]:
        return ["batch_info_host", "cu_seqlen", "slot_idx", "use_initial_states"]

    @classmethod
    def get_cache_initializers(
        cls, source_attn_node: Node, cache_config: KvCacheConfig
    ) -> ResourceHandlerDict:
        inp_fake: torch.Tensor = source_attn_node.args[0].meta["val"]
        w_fake: torch.Tensor = source_attn_node.args[1].meta["val"]

        in_channels = inp_fake.shape[-1]
        kernel_size = w_fake.shape[-1]

        # NOTE: cuda backend stores kernel_size - 1 elements in state.
        # CausalConvResourceHandler.state_shape = (conv_dim, d_conv - 1), so d_conv = kernel_size.
        # Ensure d_conv >= 1 (state_shape[-1] >= 0).
        conv_state_handler = CausalConvResourceHandler(
            conv_dim=in_channels,
            d_conv=max(1, kernel_size),  # state_shape[-1] = d_conv - 1 = kernel_size - 1
            dtype=cls.resolve_cache_dtype("auto", inp_fake.dtype),
        )
        return {"conv_state_cache": conv_state_handler}

    @classmethod
    def get_constants(cls, source_attn_node: Node) -> List[Constant]:
        stride, padding, dilation, groups, padding_mode = extract_op_args(
            source_attn_node, "stride", "padding", "dilation", "groups", "padding_mode"
        )
        # None is for activation parameter, which may not exist in the source node (added by fusion later)
        return [stride, padding, dilation, groups, padding_mode, None]
