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

from abc import abstractmethod
from typing import List

import torch
from torch._ops import OpOverloadPacket
from torch.fx import Node

from .....llmapi.llm_args import KvCacheConfig
from ...utils.node_utils import extract_op_args
from ..attention_interface import (
    AttentionDescriptor,
    AttentionLayout,
    CausalConvResourceHandler,
    Constant,
    MHACallable,
    ResourceHandlerDict,
)


class BaseCausalConvDescriptor(AttentionDescriptor):
    """Base class for causal conv1d backends.

    Provides shared implementations for:
    - get_attention_layout
    - get_num_qkv_args
    - get_source_attention_op
    - get_standard_metadata_args
    - get_cache_initializers
    - get_constants

    Subclasses must implement:
    - get_cached_attention_op
    """

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
    @abstractmethod
    def get_cached_attention_op(cls) -> MHACallable:
        """Return the cached attention op for this backend.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

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
