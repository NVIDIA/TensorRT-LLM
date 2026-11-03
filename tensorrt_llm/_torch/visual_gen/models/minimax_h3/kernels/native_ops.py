# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's native ops (``../csrc/``, built into ``libth_visual_gen_minimax_h3.so``) and their fake kernels."""

import functools
from typing import Any

import torch

from tensorrt_llm._torch.visual_gen._native import load_model_ops


def _anti_alias_snake_beta_fake(x, up_filter, down_filter, log_alpha, log_beta):
    return torch.empty_like(x, memory_format=torch.contiguous_format)


def _gate_resid_rmsnorm_fake(x, attn, weight, gate, shift, scale, indices, eps):
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty_like(x, memory_format=torch.contiguous_format),
    )


@functools.cache
def load_ops() -> Any:
    """``torch.ops.vg_minimax_h3`` with its fake kernels registered."""
    ops = load_model_ops("minimax_h3")
    torch.library.register_fake("vg_minimax_h3::anti_alias_snake_beta")(_anti_alias_snake_beta_fake)
    torch.library.register_fake("vg_minimax_h3::gate_resid_rmsnorm_indexed_shift_scale")(
        _gate_resid_rmsnorm_fake
    )
    return ops
