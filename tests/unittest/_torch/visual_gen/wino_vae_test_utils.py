# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures of the WinoVae single- and multi-GPU tests."""


def small_config() -> dict:
    """A tiny Cosmos3-shaped Wan VAE config (residual blocks, patch 2, 12 in/out channels);
    a fresh dict on every call, so a test may modify its copy."""
    return {
        "base_dim": 16,
        "decoder_base_dim": 24,
        "z_dim": 8,
        "dim_mult": [1, 2, 2, 2],
        "num_res_blocks": 1,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "dropout": 0.0,
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "scale_factor_temporal": 4,
        "scale_factor_spatial": 16,
    }
