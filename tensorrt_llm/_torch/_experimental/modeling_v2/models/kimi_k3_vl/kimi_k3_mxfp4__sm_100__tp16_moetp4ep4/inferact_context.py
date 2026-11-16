# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare five-layer Inferact context with a single normalization/RoPE launch."""

import torch
import torch.nn.functional as F

from .context_fusion import context_into


def try_prepare_context(model, projected_hidden: torch.Tensor,
                        positions: torch.Tensor) -> tuple | None:
    """Return prepared context for supported operands, otherwise use stock."""
    if not (projected_hidden.dtype == torch.bfloat16
            and projected_hidden.ndim == 2
            and projected_hidden.shape[1] == 7168
            and 1 <= projected_hidden.shape[0] <= 128
            and projected_hidden.is_contiguous()
            and model._num_attn_layers == 5
            and model.kv_lora_rank == 512
            and model.qk_rope_head_dim == 64):
        return None
    if model._fused_kv_weight is None:
        model._build_fused_kv_buffers()
    latent = F.linear(projected_hidden, model._fused_kv_weight).view(-1, 5, 576)
    freqs = model._mla_freqs_cis(latent.device)[positions.view(-1).long()]
    out = torch.empty_like(latent)
    context_into(latent, model._kv_a_norm_stacked, torch.view_as_real(freqs), out,
                 float(model._kv_a_norm_eps), True)
    return out.unsqueeze(2).contiguous(), None
