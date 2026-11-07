# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused anti-aliased SnakeBeta activation of the MiniMax-H3 BigVGAN audio decoder: Diffusers'
``MiniMaxH3AudioActivation1d`` (2x upsample, SnakeBeta, 2x downsample, 12-tap filters, replicate padding) in one
kernel, ``torch.ops.vg_minimax_h3.anti_alias_snake_beta`` (``../csrc/antiAliasActivation.cu``)."""

import torch
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
    MiniMaxH3AudioActivation1d,
    MiniMaxH3AudioSnakeBeta,
)

from .native_ops import load_ops

_TAPS = 12


class FusedAntiAliasActivation1d(MiniMaxH3AudioActivation1d):
    """``MiniMaxH3AudioActivation1d`` with SnakeBeta, run by the fused kernel for fp32 CUDA input."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not (
            hidden_states.is_cuda
            and hidden_states.dtype == torch.float32
            and hidden_states.dim() == 3
        ):
            return super().forward(hidden_states)
        return torch.ops.vg_minimax_h3.anti_alias_snake_beta(
            hidden_states,
            self.upsample.filter.view(-1),
            self.downsample.lowpass.filter.view(-1),
            self.act.alpha,
            self.act.beta,
        )


def _fusable(module: torch.nn.Module) -> bool:
    up, low = (
        getattr(module, "upsample", None),
        getattr(getattr(module, "downsample", None), "lowpass", None),
    )
    return (
        type(module) is MiniMaxH3AudioActivation1d
        and type(module.act) is MiniMaxH3AudioSnakeBeta
        and module.act.alpha.dtype == torch.float32
        and up is not None
        and low is not None
        and up.ratio == 2
        and up.filter.numel() == _TAPS
        and (up.pad, up.pad_left, up.pad_right) == (5, 15, 15)
        and low.stride == 2
        and low.filter.numel() == _TAPS
        and (low.pad_left, low.pad_right) == (5, 6)
    )


def install_fused_activations(decoder: torch.nn.Module) -> int:
    """Run every fusable anti-aliased activation of ``decoder`` with the fused kernel on an SM100/SM103 GPU; returns
    the number installed. The decoder may still be on the host (CPU offloading); the kernel runs for CUDA input."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        return 0
    load_ops()
    count = 0
    for module in decoder.modules():
        if _fusable(module):
            module.__class__ = FusedAntiAliasActivation1d
            count += 1
    return count
