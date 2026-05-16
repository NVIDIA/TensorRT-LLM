# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math

import torch
from torch import nn

from tensorrt_llm.mapping import Mapping
from tensorrt_llm.functional import AllReduceStrategy
from tensorrt_llm._torch.distributed import AllReduce


class RMSNorm(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        has_weights: bool = True,
        use_gemma: bool = False,
        enable_tp: bool = False,
        mapping: Optional[Mapping] = None,
        allreduce_strategy: AllReduceStrategy = AllReduceStrategy.NCCL,
        force_sharding_alignment: Optional[int] = None,
    ):
        super().__init__()

        self.variance_epsilon = eps
        self.use_gemma = use_gemma

        self.mapping = mapping
        self.enable_tp = enable_tp

        if enable_tp:
            assert mapping is not None
            self.full_size = hidden_size
            self.align = force_sharding_alignment or 1
            assert self.full_size % self.align == 0, 'Cannot align shard if the alignment does not divide the sharding dim'
            shard = math.ceil(hidden_size / mapping.tp_size)
            shard = math.ceil(shard / self.align) * self.align
            start = shard * mapping.tp_rank
            end = min(shard * (mapping.tp_rank + 1), hidden_size)
            hidden_size = end - start

            self.allreduce = AllReduce(mapping=mapping,
                                       strategy=allreduce_strategy,
                                       dtype=torch.float32)
        else:
            self.allreduce = None

        if use_gemma and not has_weights:
            raise ValueError("has_weights must be True if use_gemma is True")
        if has_weights:
            if not use_gemma:
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, dtype=dtype, device=device))
            else:
                self.weight = nn.Parameter(
                    torch.zeros(hidden_size, dtype=dtype, device=device))
        else:
            self.register_buffer('weight',
                                 torch.ones(hidden_size,
                                            dtype=dtype,
                                            device=device),
                                 persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if self.allreduce:
            x2_sum = hidden_states.pow(2).sum(-1, keepdim=True)
            x2_sum = self.allreduce(x2_sum)
            variance = x2_sum / self.full_size
        else:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)

        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        if not self.use_gemma:
            hidden_states = self.weight * hidden_states.to(input_dtype)
        else:
            hidden_states = (self.weight + 1) * hidden_states.to(input_dtype)

        return hidden_states

    def load_weights(self, weights: torch.Tensor):
        for param_name, param in self._parameters.items():
            if param is None or param_name not in weights:
                continue
            if param_name == 'weight' and self.enable_tp:
                shard = math.ceil(self.full_size / self.mapping.tp_size)
                shard = math.ceil(shard / self.align) * self.align
                start = shard * self.mapping.tp_rank
                end = min(shard * (self.mapping.tp_rank + 1), self.full_size)
                data = weights[param_name][..., start:end]
            else:
                data = weights[param_name]

            param.data.copy_(data.to(param.dtype))

