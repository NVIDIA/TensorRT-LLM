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

from typing import List, Optional

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
    enable_sharding: bool = False,
    output_sizes: Optional[List[int]] = None,
    layer_type: str = "ssm",
) -> torch.Tensor:
    """Causal 1D convolution along the sequence axis (sliding window over time).

    The input layout is ``[batch, seq_len, channels]``. The implementation transposes
    to ``[batch, channels, seq_len]``, applies :func:`torch.nn.functional.conv1d`,
    then trims to the original sequence length so the receptive field is causal.

    Args:
        input: Activations of shape ``[batch, seq_len, in_channels]`` (or compatible
            channel-last layout consumed by the transpose into conv1d).
        weight: Conv1d kernel of shape
            ``(out_channels, in_channels / groups, kernel_size)``.
        bias: Optional bias of shape ``(out_channels,)``.
        stride: Conv1d stride (default ``1``).
        padding: Conv1d padding (default ``0``).
        dilation: Conv1d dilation (default ``1``).
        groups: Conv1d groups (default ``1``).
        padding_mode: Must be ``"zeros"``; other modes raise.
        enable_sharding: When ``True``, ``apply_sharding_hints`` shards the conv1d
            ``weight`` along its **output channel** dimension (head-parallel conv
            weights). When ``False``, sharding passes leave weights unchanged.
        output_sizes: Optional group sizes for fused-weight proportional column
            sharding (same convention as linear fused projections; consumed by
            ``apply_sharding_hints`` when applicable).
        layer_type: Layer classification for selective sharding via ``shard_layers``
            config. Values: ``"mha"``, ``"mla"``, ``"mlp"``, ``"moe"``, ``"ssm"``,
            ``"delta"``, ``"unknown"``.

    Sharding hint arguments (graph-level metadata for ``apply_sharding_hints``):
        ``enable_sharding``: When ``True``, ``apply_sharding_hints`` will shard the op's
        weight ancestors along the conv output-channel dimension (per-rank conv).
        ``output_sizes``: Group sizes for fused-weight proportional column sharding
        when the surrounding graph uses fused projections.
        ``layer_type``: Layer classification for selective sharding via
        ``shard_layers`` config.

    Returns:
        Tensor of the same batch/sequence layout as ``input`` after causal conv.
    """
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
    enable_sharding: bool = False,
    output_sizes: Optional[List[int]] = None,
    layer_type: str = "ssm",
) -> torch.Tensor:
    return torch.empty_like(input)
