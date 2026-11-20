# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed backend and component settings of the single-GPU static-FP8 scenario."""

import os
import shutil

from torch.utils.cpp_extension import CUDA_HOME

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


def missing_build_tools() -> list[str]:
    """The tools the conversion kernels are built and checked with at load that this process lacks."""
    missing = [tool for tool in ("ninja", "cuobjdump") if shutil.which(tool) is None]
    if CUDA_HOME is None or not os.path.isfile(os.path.join(CUDA_HOME, "bin", "nvcc")):
        missing.insert(0, "nvcc")
    return missing


def select() -> None:
    """Fix shared attention/decoder settings before their modules are imported.

    Raises RuntimeError when the scenario cannot run in this process: a build tool is missing, or
    the shared settings are already selected with other values.
    """
    missing = missing_build_tools()
    if missing:
        raise RuntimeError(
            "the conversion kernels are built at load and need " + ", ".join(missing)
        )
    shared_config.apply(PRESET)
