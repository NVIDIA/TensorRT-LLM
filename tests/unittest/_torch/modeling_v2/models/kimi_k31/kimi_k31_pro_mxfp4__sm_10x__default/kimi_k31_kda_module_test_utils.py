# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA attention module fixture for the GPU tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31KdaLayer,
    )


def make_attention(
    heads: int,
    hidden_size: int = 7168,
    *,
    weight_seed: int = 17,
) -> K31KdaLayer:
    """Build an attention module with reproducible synthetic BF16 checkpoint values."""
    import torch

    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (  # noqa: E501
        K31KdaLayer,
    )

    cfg = SimpleNamespace(
        hidden_size=hidden_size,
        rms_norm_eps=1e-5,
        linear_attn_config=dict(
            num_heads=heads,
            head_dim=128,
            short_conv_kernel_size=4,
            use_full_rank_gate=True,
            gate_lower_bound=-5.0,
        ),
        kda_fp32_gate_params=True,
    )
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(weight_seed)
        with torch.device("cuda"):
            attention = K31KdaLayer(cfg, layer_idx=0, extra_attrs={})
        generator = torch.Generator(device="cuda").manual_seed(weight_seed)
        with torch.no_grad():
            for param in attention.parameters():
                if param.dim() >= 2:  # projections and convolutions
                    param.copy_(
                        torch.randn(param.shape, device="cuda", generator=generator)
                        / param[0].numel() ** 0.5
                    )
            attention.dt_bias.zero_()
        attention.finalize_decode_weights()
    attention.eval()
    return attention
