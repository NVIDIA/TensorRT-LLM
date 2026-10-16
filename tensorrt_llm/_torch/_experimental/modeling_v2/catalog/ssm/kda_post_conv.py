# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill's conv output to q / k / v: L2-normalize q and k per head and go token-major, in one launch."""

import torch

from tensorrt_llm._torch.modules.kimi_kda._kda_kernels import fused_kda_post_conv


def kda_post_conv(
    packed: torch.Tensor,
    num_heads: int,
    head_dim: int,
    l2_norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a channel-major `[3 * heads * head_dim, tokens]` conv output into token-major q, k, v.

    Returns three new `[1, tokens, heads, head_dim]` tensors in `packed.dtype`; q and k are
    L2-normalized along the head dim, v is copied as is.
    """
    return fused_kda_post_conv(packed, num_heads, head_dim, l2_norm_eps)
