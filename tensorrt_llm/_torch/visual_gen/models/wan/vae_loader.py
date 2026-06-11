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

WAN_VAE_PIPELINE_CONFIG_DEFAULTS: dict[str, str] = {}
TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV = "TRTLLM_WAN_VAE_BACKEND_FALLBACK"
WAN_VAE_LAYOUT_DEFAULT = "channels_last"


def _wan_vae_backend_fallback() -> bool:
    fallback = os.environ.get(TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV, "").strip()
    if not fallback:
        return False

    try:
        return int(fallback) != 0
    except ValueError:
        raise ValueError(
            f"{TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV} must be an integer; "
            "set it to a non-zero value to load Diffusers Wan VAE."
        ) from None


def _is_single_gpu_wan_vae(visual_gen_mapping: Any | None) -> bool:
    if visual_gen_mapping is None:
        return True
    return (
        getattr(visual_gen_mapping, "world_size", 1) == 1
        and getattr(visual_gen_mapping, "parallel_vae_size", 1) == 1
    )


def _should_use_wan_vae(visual_gen_mapping: Any | None) -> bool:
    return _is_single_gpu_wan_vae(visual_gen_mapping) and not _wan_vae_backend_fallback()


def load_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    visual_gen_mapping: Any | None = None,
) -> nn.Module:
    use_wan_vae = _should_use_wan_vae(visual_gen_mapping)
    if not use_wan_vae:
        if _wan_vae_backend_fallback():
            logger.info(
                "Loading Diffusers Wan VAE because "
                f"{TRTLLM_WAN_VAE_BACKEND_FALLBACK_ENV} is non-zero."
            )
        else:
            logger.info("Loading Diffusers Wan VAE because Wan VAE is single-GPU only.")
        return AutoencoderKLWan.from_pretrained(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
        ).to(device)

    layout_mode = WAN_VAE_LAYOUT_DEFAULT
    logger.info(f"Loading Wan VAE with layout_mode={layout_mode!r}.")
    vae_dir = Path(checkpoint_dir) / "vae"
    wan_vae = WanVAE(
        WanVAEConfig.from_json_file(vae_dir / "config.json"),
        layout_mode=layout_mode,
    )
    state_dict = WeightLoader(components=PipelineComponent.VAE).load_weights(
        checkpoint_dir,
        Mapping(),
    )
    wan_vae.load_diffusers_state_dict(state_dict, strict=True)

    return wan_vae.to(device=device, dtype=torch.bfloat16).eval()
