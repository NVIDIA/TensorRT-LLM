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

"""Custom ops corresponding to fla's chunked delta rule.

Delta Rule is based on this paper: https://arxiv.org/abs/2406.06484

Kernels are based on this repo: https://github.com/fla-org/flash-linear-attention
"""

from typing import Optional

import torch

from .delta_rule.chunk import chunk_delta_rule_fwd


@torch.library.custom_op("auto_deploy::fla_delta_rule", mutates_args=())
def fla_chunked_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    o, A, final_state = chunk_delta_rule_fwd(
        q, k, v, beta, scale, initial_state=None, output_final_state=False, cu_seqlens=None
    )
    return o


@fla_chunked_delta_rule.register_fake
def fla_chunked_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return torch.empty_like(v)
