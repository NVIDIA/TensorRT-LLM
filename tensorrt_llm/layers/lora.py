# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


class Lora(Module):

    def __init__(self,
                 in_hidden_size: int = 0,
                 out_hidden_size: int = 0,
                 max_low_rank: int = 0) -> None:
        super().__init__()

        self.in_hidden_size = in_hidden_size
        self.out_hidden_size = out_hidden_size
        self.max_low_rank = max_low_rank

    def forward(self,
                x,
                host_request_types=None,
                host_context_lengths=None,
                max_context_length: int = 0,
                lora_ranks=None,
                lora_weights_pointers=None):
        if default_net().plugin_config.lora_plugin:
            x = lora_plugin(x,
                            in_hidden_size=self.in_hidden_size,
                            out_hidden_size=self.out_hidden_size,
                            host_request_types=host_request_types,
                            transb=True,
                            host_context_lengths=host_context_lengths,
                            max_context_length=max_context_length,
                            max_low_rank=self.max_low_rank,
                            lora_ranks=lora_ranks,
                            lora_weights_pointers=lora_weights_pointers)
        else:
            assert False, "Not support lora without plugin"

        return x


class LoraParams(object):

    def __init__(self,
                 lora_ranks: Tensor = None,
                 lora_weights_pointers_list: List[Tensor] = None):

        self.lora_ranks = lora_ranks
        self.lora_weights_pointers_list = lora_weights_pointers_list
