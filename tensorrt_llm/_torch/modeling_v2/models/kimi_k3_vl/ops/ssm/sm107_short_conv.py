# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Depthwise causal short-convolution weight of a KDA q / k / v stream.

:class:`ShortConvWeight` registers ``weight`` ``[channels, 1, kernel_size]`` and no bias, the
parameter a depthwise ``nn.Conv1d`` (``groups=channels``) registers, so the checkpoint's
``*_conv1d.weight`` loads by name. The convolution itself is the caller's kernel call on that
weight (``causal_conv1d_fn``, the KDA prefill / decode / verify kernels), with ``activation``
(SiLU) applied inside it.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ShortConvWeight(nn.Module):
    """``weight [channels, 1, kernel_size]`` of one rank's channels."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        activation: Optional[str] = "silu",
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size, self.kernel_size, self.activation = channels, kernel_size, activation
        self.weight = nn.Parameter(
            torch.empty(channels, 1, kernel_size, dtype=dtype, device=device)
        )
        self.register_parameter("bias", None)
