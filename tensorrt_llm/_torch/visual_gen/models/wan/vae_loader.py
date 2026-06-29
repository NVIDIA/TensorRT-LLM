# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from .wan_vae import WanVAE, WanVAEConfig

TRTLLM_USE_DIFFUSER_VAE_ENV = "TRTLLM_USE_DIFFUSER_VAE"


def _use_diffuser_vae_env() -> bool:
    """Whether the Diffusers Wan VAE is forced via the debug env var.

    Unset or ``0`` keeps the native Wan VAE default; any non-zero integer
    forces the Diffusers ``AutoencoderKLWan`` fallback.
    """
    fallback = os.environ.get(TRTLLM_USE_DIFFUSER_VAE_ENV, "").strip()
    if not fallback:
        return False

    try:
        return int(fallback) != 0
    except ValueError:
        raise ValueError(
            f"{TRTLLM_USE_DIFFUSER_VAE_ENV} must be an integer; "
            "set it to a non-zero value to load Diffusers Wan VAE."
        ) from None


def _vae_is_parallel(visual_gen_mapping: Any | None) -> bool:
    if visual_gen_mapping is None:
        return False
    return getattr(visual_gen_mapping, "parallel_vae_size", 1) > 1


def _use_native_wan_vae(visual_gen_mapping: Any | None) -> bool:
    """Select the Wan VAE backend and log the reason.

    The native ``WanVAE`` is the default. Diffusers ``AutoencoderKLWan`` is used
    only when the VAE is tensor-parallel (``parallel_vae_size > 1``), which the
    native path does not support yet, or when forced via
    ``TRTLLM_USE_DIFFUSER_VAE``.
    """
    if _use_diffuser_vae_env():
        logger.info(f"Loading Diffusers Wan VAE because {TRTLLM_USE_DIFFUSER_VAE_ENV} is non-zero.")
        return False
    if _vae_is_parallel(visual_gen_mapping):
        logger.info("Loading Diffusers Wan VAE because parallel VAE is not supported natively yet.")
        return False
    return True


def load_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    visual_gen_mapping: Any | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if not _use_native_wan_vae(visual_gen_mapping):
        return AutoencoderKLWan.from_pretrained(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)

    vae_dir = Path(checkpoint_dir) / "vae"
    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    state_dict = WeightLoader(components=PipelineComponent.VAE).load_weights(
        checkpoint_dir,
        Mapping(),
    )
    wan_vae.load_state_dict(state_dict, strict=True)

    return wan_vae.to(device=device, dtype=dtype).eval()
