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

"""Custom op collection for uncached causal conv (sliding window with 1d)."""

from typing import Optional

import torch
import torch.nn.functional as F


@torch.library.custom_op("auto_deploy::torch_causal_conv1d", mutates_args={})
def _torch_causal_conv1d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    assert padding_mode == "zeros", "padding_mode must be zeros"

    batch_size, seq_len, _ = input.shape

    return (
        F.conv1d(
            input.transpose(1, 2),
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )[..., :seq_len]
        .transpose(1, 2)
        .contiguous()
    )


@_torch_causal_conv1d.register_fake
def _torch_causal_conv1d_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    return torch.empty_like(input)
