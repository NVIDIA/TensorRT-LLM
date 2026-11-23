# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 FP8 Video VAE and chunk CUDA-graph configuration for MiniMax-H3 FP8 U4."""

from __future__ import annotations

import torch

from tensorrt_llm.logger import logger

from ..minimax_h3_nvfp4__sm_107__u1_u4.config import SCENARIO
from ..minimax_h3_nvfp4__sm_107__u1_u4.kernels.vae_decoder_fp8 import (
    convert_decoder_linears_fp8,
    set_tiles,
)

__all__ = [
    "configure_u4_video_vae_pre_load",
    "configure_u4_video_vae_post_load",
]


def _is_sm107_cuda(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.get_device_capability(device) == (10, 7)


def configure_u4_video_vae_pre_load(vae, device: torch.device, *, max_shapes: int | None = None) -> bool:
    """Enable VAE decoder chunk CUDA graphs on SM107 before ``super().post_load_weights()``."""
    if vae is None or not _is_sm107_cuda(device):
        return False
    shapes = int(SCENARIO.vae_graph_max_shapes if max_shapes is None else max_shapes)
    vae.decoder_graph_max_shapes = shapes
    logger.info(f"MiniMax-H3 FP8 U4 VAE decoder_graph_max_shapes={shapes}")
    return True


def configure_u4_video_vae_post_load(
    vae,
    device: torch.device,
    *,
    ulysses_size: int = 4,
    enable_fp8: bool = True,
) -> bool:
    """Convert Video VAE decoder transformer Linears to tiled FP8 on SM107 after weights load."""
    if not enable_fp8 or vae is None or not _is_sm107_cuda(device):
        return False
    if SCENARIO.vae_fp8_tiled:
        set_tiles(
            mlp=SCENARIO.vae_fp8_tile,
            q=SCENARIO.resolved_vae_fp8_qtile(int(ulysses_size)),
        )
    convert_decoder_linears_fp8(vae.decoder)
    logger.info(f"MiniMax-H3 FP8 U4 Video VAE decoder converted to FP8 (ulysses={ulysses_size})")
    return True
