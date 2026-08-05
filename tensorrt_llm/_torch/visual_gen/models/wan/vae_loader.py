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

import json
import os
from pathlib import Path

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


def _use_native_wan_vae() -> bool:
    """Select the Wan VAE backend and log the reason.

    The native ``WanVAE`` is the default; the diffusers ``AutoencoderKLWan``
    is used only when forced via ``TRTLLM_USE_DIFFUSER_VAE``.
    """
    if _use_diffuser_vae_env():
        logger.info(f"Loading Diffusers Wan VAE because {TRTLLM_USE_DIFFUSER_VAE_ENV} is non-zero.")
        return False
    return True


def _is_nvfp4_vae_ckpt(vae_dir: Path) -> bool:
    """Return whether the VAE config declares ModelOpt NVFP4 quantization."""
    config_path = vae_dir / "config.json"
    if not config_path.exists():
        return False
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    quantization_config = config.get("quantization_config")
    return (
        isinstance(quantization_config, dict) and quantization_config.get("quant_algo") == "NVFP4"
    )


def _load_nvfp4_wan_vae(checkpoint_dir: str, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load ModelOpt weights and replace their quantized Conv3d modules.

    Quantized weights are first dequantized into the native state dict, then
    requantized in the KTRSC layout consumed by the CuTe kernel. Consequently,
    this path is accuracy-gated as a TensorRT-LLM implementation rather than
    expected to preserve ModelOpt's original packed bytes.
    """
    from safetensors.torch import load_file

    from .wan_vae import WanCausalConv3d, dequant_fp4_conv_weight, swap_wan_convs_to_fp4

    vae_dir = Path(checkpoint_dir) / "vae"
    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    raw_state_dict = load_file(str(vae_dir / "diffusion_pytorch_model.safetensors"))
    conv_modules = {
        name: module
        for name, module in wan_vae.named_modules()
        if isinstance(module, WanCausalConv3d)
    }
    quantized = {
        key.removesuffix(".weight_scale") for key in raw_state_dict if key.endswith(".weight_scale")
    }
    unknown_modules = quantized - conv_modules.keys()
    if unknown_modules:
        raise ValueError(
            f"NVFP4 checkpoint references unknown convolutions: {sorted(unknown_modules)}"
        )

    input_scales: dict[str, float] = {}
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        prefix = key.rsplit(".", 1)[0]
        if prefix in quantized and key.endswith(
            (".weight_scale", ".weight_scale_2", ".input_scale")
        ):
            if key.endswith(".input_scale"):
                if value.numel() != 1:
                    raise ValueError(f"{key} must contain one calibrated scale")
                input_scales[prefix] = float(value.item())
            continue
        if prefix in quantized and key.endswith(".weight"):
            module = conv_modules[prefix]
            if module.kernel_size != (3, 3, 3):
                raise ValueError(
                    f"NVFP4 Wan VAE supports only 3x3x3 weights, got {prefix}: {module.kernel_size}"
                )
            block_scale_key = f"{prefix}.weight_scale"
            global_scale_key = f"{prefix}.weight_scale_2"
            if block_scale_key not in raw_state_dict or global_scale_key not in raw_state_dict:
                raise ValueError(f"NVFP4 checkpoint is missing scales for {prefix}")
            state_dict[key] = (
                dequant_fp4_conv_weight(
                    value,
                    raw_state_dict[block_scale_key],
                    raw_state_dict[global_scale_key],
                    module.in_channels,
                )
                .reshape(-1, module.in_channels, 3, 3, 3)
                .to(dtype)
            )
        else:
            state_dict[key] = value
    wan_vae.load_state_dict(state_dict, strict=True)
    wan_vae = wan_vae.to(device=device, dtype=dtype).eval()
    n, n_static = swap_wan_convs_to_fp4(wan_vae, input_scales, only_names=quantized)
    if n != len(quantized):
        raise ValueError(
            f"NVFP4 checkpoint contains {len(quantized)} quantized convolutions, "
            f"but only {n} are supported Wan residual convolutions"
        )
    logger.info(
        f"Loaded NVFP4 Wan VAE: {len(quantized)} quantized convs; "
        f"{n} run on the FP4 kernel ({n_static} static, {n - n_static} dynamic)."
    )
    return wan_vae


def load_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if not _use_native_wan_vae():
        return AutoencoderKLWan.from_pretrained(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)

    vae_dir = Path(checkpoint_dir) / "vae"
    if _is_nvfp4_vae_ckpt(vae_dir):
        return _load_nvfp4_wan_vae(checkpoint_dir, device, dtype)

    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    state_dict = WeightLoader(components=PipelineComponent.VAE).load_weights(
        checkpoint_dir,
        Mapping(),
    )
    wan_vae.load_state_dict(state_dict, strict=True)

    return wan_vae.to(device=device, dtype=dtype).eval()
