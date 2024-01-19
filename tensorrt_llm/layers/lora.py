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

from .._common import default_net
from ..functional import Tensor, lora_plugin
from ..module import Module


class LoraRuntimeParams(object):

    def __init__(
        self,
        lora_ranks: List[Tensor] = None,
        lora_weights_pointers: List[Tensor] = None,
        host_request_types: Tensor = None,
        host_context_lengths: Tensor = None,
        max_context_length: Tensor = None,
    ):

        self.lora_ranks = lora_ranks
        self.lora_weights_pointers = lora_weights_pointers
        self.host_request_types = host_request_types
        self.host_context_lengths = host_context_lengths
        self.max_context_length = max_context_length


class Lora(Module):

    def __init__(self,
                 in_hidden_size: int = 0,
                 out_hidden_sizes: List[int] = [0],
                 max_low_rank: int = 0) -> None:
        super().__init__()

        self.in_hidden_size = in_hidden_size
        self.out_hidden_sizes = out_hidden_sizes
        self.max_low_rank = max_low_rank

    def forward(self, x, lora_runtime_params: LoraRuntimeParams = None):
        if default_net().plugin_config.lora_plugin:
            result = lora_plugin(
                x,
                in_hidden_size=self.in_hidden_size,
                out_hidden_sizes=self.out_hidden_sizes,
                host_request_types=lora_runtime_params.host_request_types,
                transb=True,
                host_context_lengths=lora_runtime_params.host_context_lengths,
                max_context_length=lora_runtime_params.max_context_length,
                max_low_rank=self.max_low_rank,
                lora_ranks=lora_runtime_params.lora_ranks,
                lora_weights_pointers=lora_runtime_params.lora_weights_pointers)
        else:
            assert False, "Not support lora without plugin"

        return result


class LoraParams(object):

    def __init__(
            self,
            lora_ranks=None,  # : List[dict[Tensor]]
            lora_weights_pointers=None,  # : List[dict[Tensor]]
            host_context_lengths: Tensor = None,
            max_context_length: Tensor = None,
            host_request_types: Tensor = None):

        self.lora_ranks = lora_ranks
        self.lora_weights_pointers = lora_weights_pointers

        self.host_context_lengths = host_context_lengths
        self.max_context_length = max_context_length
        self.host_request_types = host_request_types

    def get_layer_params(self, layer_idx: int):
        return LoraParams(
            lora_ranks=[self.lora_ranks[layer_idx]],
            lora_weights_pointers=[self.lora_weights_pointers[layer_idx]],
            host_context_lengths=self.host_context_lengths,
            max_context_length=self.max_context_length,
            host_request_types=self.host_request_types)

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
                max_context_length=self.max_context_length,
                host_request_types=self.host_request_types,
            )
        else:
            return None
