# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed backend and component settings of the single-GPU static-FP8 scenario."""

from ..minimax_h3_nvfp4__sm_107__u1_u4 import config as shared_config

PRESET = {
    "attention_fmha": "trtllm-gen-fp16sm",
    "skip_corr_threshold": 0.0,
    "fp8_scale_fusion": "qkstatic",
    "fp8_fusion_block_h": 8,
    "fp8_fusion_warps": 2,
    "grouped_cast": True,
    "vamax_fold": False,
    "fused_quant": False,
    "out_quant": False,
    "out_a2a_fp4": False,
    "vae_fp8": True,
    "vae_fp8_tiled": True,
    "vae_fused_norm": True,
    "vae_graph_max_shapes": 2,
    "text_encoder_graph": True,
    "preloop_device_plan": True,
    "preloop_async_copies": True,
    "preloop_noise_early": True,
}


def select() -> None:
    """Fix shared attention/decoder settings before their modules are imported."""
    shared_config.apply(PRESET)
