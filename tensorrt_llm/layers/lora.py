# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import List

import numpy as np

from .._common import default_net
from ..functional import Tensor, constant, dora_plugin, lora_plugin, where
from ..module import Module


class LoraRuntimeParams(object):

    def __init__(
        self,
        lora_ranks: List[Tensor] = None,
        lora_weights_pointers: List[Tensor] = None,
        host_request_types: Tensor = None,
        host_context_lengths: Tensor = None,
        max_encoder_context_length: Tensor = None,
        host_encoder_input_lengths: Tensor = None,
        weight_index: int = 0,
        partial_lora_mask: Tensor = None,
    ):

        self.lora_ranks = lora_ranks
        self.lora_weights_pointers = lora_weights_pointers
        self.host_request_types = host_request_types
        self.host_context_lengths = host_context_lengths
        self.max_encoder_context_length = max_encoder_context_length
        self.host_encoder_input_lengths = host_encoder_input_lengths
        self.weight_index = weight_index
        self.partial_lora_mask = partial_lora_mask  # Partial LoRA for https://arxiv.org/abs/2401.16420


class Lora(Module):

    def __init__(self,
                 in_hidden_size: int = 0,
                 out_hidden_sizes: List[int] = [0],
                 max_low_rank: int = 0) -> None:
        super().__init__()

        self.in_hidden_size = in_hidden_size
        self.out_hidden_sizes = out_hidden_sizes
        self.max_low_rank = max_low_rank

    def forward(self,
                x,
                lora_runtime_params: LoraRuntimeParams = None,
                is_cross_attention: bool = False):
        if default_net().plugin_config.lora_plugin:
            result = lora_plugin(
                x,
                in_hidden_size=self.in_hidden_size,
                out_hidden_sizes=self.out_hidden_sizes,
                host_request_types=lora_runtime_params.host_request_types,
                transb=True,
                # For cross attention, host_encoder_input_lengths should be used instead of host_context_lengths
                host_context_lengths=lora_runtime_params.host_context_lengths
                if not is_cross_attention else
                lora_runtime_params.host_encoder_input_lengths,
                max_low_rank=self.max_low_rank,
                lora_ranks=lora_runtime_params.lora_ranks,
                lora_weights_pointers=lora_runtime_params.lora_weights_pointers,
                weight_index=lora_runtime_params.weight_index,
            )
            if lora_runtime_params.partial_lora_mask is not None:
                zero_tensor = constant(np.array([0.0], dtype=np.float16))
                if isinstance(result, List):
                    result = [
                        where(lora_runtime_params.partial_lora_mask, r,
                              zero_tensor) for r in result
                    ]
                elif isinstance(result, Tensor):
                    result = where(lora_runtime_params.partial_lora_mask,
                                   result, zero_tensor)
                else:
                    assert False
        else:
            assert False, "Not support lora without plugin"

        return result


class Dora(Module):

    def __init__(self, out_hidden_sizes: List[int] = [0]) -> None:
        super().__init__()
        self.out_hidden_sizes = out_hidden_sizes

    def forward(self,
                x,
                lora_runtime_params: LoraRuntimeParams = None,
                is_cross_attention: bool = False):
        assert lora_runtime_params.weight_index == 0, "DoRA does not support weight_index != 0"
        if default_net().plugin_config.lora_plugin and default_net(
        ).plugin_config.dora_plugin:
            result = dora_plugin(
                x,
                out_hidden_sizes=self.out_hidden_sizes,
                host_request_types=lora_runtime_params.host_request_types,
                host_context_lengths=lora_runtime_params.host_context_lengths
                if not is_cross_attention else
                lora_runtime_params.host_encoder_input_lengths,
                lora_weights_pointers=lora_runtime_params.lora_weights_pointers,
            )
        else:
            assert False, "Not support dora without plugin"

        return result


class LoraParams(object):

    def __init__(
        self,
        lora_ranks=None,  # : List[dict[Tensor]]
        lora_weights_pointers=None,  # : List[dict[Tensor]]
        host_context_lengths: Tensor = None,
        max_encoder_context_length: Tensor = None,  # For cross attention
        host_request_types: Tensor = None,
        host_encoder_input_lengths: Tensor = None,  # For cross attention
        weight_index: int = 0,
        partial_lora_mask: Tensor = None,
    ):

        self.lora_ranks = lora_ranks
        self.lora_weights_pointers = lora_weights_pointers

        self.host_context_lengths = host_context_lengths
        self.max_encoder_context_length = max_encoder_context_length
        self.host_request_types = host_request_types
        self.host_encoder_input_lengths = host_encoder_input_lengths
        self.weight_index = weight_index

        self.partial_lora_mask = partial_lora_mask  # Partial LoRA for https://arxiv.org/abs/2401.16420

    def get_layer_params(self, layer_idx: int):
        return LoraParams(
            lora_ranks=[self.lora_ranks[layer_idx]],
            lora_weights_pointers=[self.lora_weights_pointers[layer_idx]],
            host_context_lengths=self.host_context_lengths,
            max_encoder_context_length=self.max_encoder_context_length,
            host_request_types=self.host_request_types,
            host_encoder_input_lengths=self.host_encoder_input_lengths,
            weight_index=self.weight_index,
            partial_lora_mask=self.partial_lora_mask)

    def get_runtime_params(self, layer_idx: int, lora_module: str):
        if f"{lora_module}_lora_ranks" in self.lora_ranks[layer_idx]:
            return LoraRuntimeParams(
                lora_ranks=[
                    self.lora_ranks[layer_idx][f"{lora_module}_lora_ranks"]
                ],
                lora_weights_pointers=[
                    self.lora_weights_pointers[layer_idx]
                    [f"{lora_module}_lora_weights_pointers"]
                ],
                host_context_lengths=self.host_context_lengths,
                max_encoder_context_length=self.max_encoder_context_length,
                host_request_types=self.host_request_types,
                host_encoder_input_lengths=self.host_encoder_input_lengths,
                weight_index=self.weight_index,
                partial_lora_mask=self.partial_lora_mask,
            )
        else:
            return None
