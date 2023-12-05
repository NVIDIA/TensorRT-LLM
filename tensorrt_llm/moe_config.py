# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Tuple


class MoeLayerConfig:
    '''
    The MoeLayerConfig is a helper class to allow users to configure MoE on a per model or per layer way conveniently
    '''

    def __init__(self,
                 num_experts=0,
                 top_k=0,
                 per_layer: List[Tuple[int, int]] = None):
        '''
        :param num_experts: The number of experts every layer should use
        :param top_k: The top-k value every layer should use
        :param per_layer: A list of tuples (num experts, topk) describing the MoE config for each layer.
                          Use (0,0) or None to disable MoE for a specific layer
        '''
        num_experts = num_experts if num_experts is not None else 0
        top_k = top_k if top_k is not None else 0
        assert (num_experts == 0) == (top_k == 0)  # Both or neither
        assert not bool(per_layer) or (num_experts == 0)  # At most 1
        self._per_layer_config = per_layer
        self._num_experts = num_experts
        self._top_k = top_k

    def __getitem__(self, layer_idx) -> Tuple[int, int]:
        if self._per_layer_config:
            result = self._per_layer_config[layer_idx]
            return result if result else (0, 0)
        return self._num_experts, self._top_k

    def num_experts(self, layer_idx):
        return self.__getitem__(layer_idx)[0]

    def top_k(self, layer_idx):
        return self.__getitem__(layer_idx)[1]
