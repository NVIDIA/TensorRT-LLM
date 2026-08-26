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
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan

from tensorrt_llm._torch.visual_gen.checkpoints import WeightLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from .wan_vae import WanVAE, WanVAEConfig, _supports_nvfp4_device

TRTLLM_USE_DIFFUSER_VAE_ENV = "TRTLLM_USE_DIFFUSER_VAE"
_NVFP4_DYNAMIC_MIN_CHANNELS = 64


def _nvfp4_enabled_for_device(
    device: torch.device,
    selected: set[str],
    *,
    explicit_request: bool,
) -> bool:
    """Resolve NVFP4 execution when selected convolutions cannot run on ``device``."""
    if not selected or _supports_nvfp4_device(device):
        return True
    if explicit_request:
        raise ValueError("NVFP4 Wan VAE requires an SM100-family GPU")
    logger.warning(
        "The NVFP4 VAE checkpoint will use dequantized BF16 operators "
        "because the selected device is not an SM100-family GPU."
    )
    return False


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


def _warn_dequantized_nvfp4_weights(
    quantized: set[str],
    selected: set[str],
    enable_fp4: bool,
) -> None:
    """Warn when a BF16 operator cannot restore the checkpoint's original weights."""
    dequantized_bf16 = quantized - selected if enable_fp4 else quantized
    if dequantized_bf16:
        logger.warning(
            f"{len(dequantized_bf16)} VAE convolution(s) will use BF16 operators with "
            "weights dequantized from the NVFP4 checkpoint. This changes the operator "
            "precision but cannot recover the original BF16 weights."
        )


def _validate_dynamic_weight_request(
    *, checkpoint_is_fp4: bool, selected: set[str], dynamic_weight_quant: bool | None
) -> None:
    """Validate an explicit weight mode against the selected checkpoint weights."""
    if not selected or dynamic_weight_quant is None:
        return
    if checkpoint_is_fp4 and dynamic_weight_quant:
        raise ValueError(
            "vae_quant_config weights.dynamic=true requires high-precision VAE weights, "
            "but the selected checkpoint layers contain only packed NVFP4 weights"
        )
    if not checkpoint_is_fp4 and not dynamic_weight_quant:
        raise ValueError(
            "vae_quant_config weights.dynamic=false requires packed NVFP4 VAE weights, "
            "but the selected checkpoint layers contain high-precision weights"
        )


def _resolve_input_scales(
    selected: set[str],
    input_scales: dict[str, float],
    dynamic_activation_quant: bool | None,
) -> dict[str, float]:
    """Resolve explicit or checkpoint-driven activation quantization per layer."""
    if dynamic_activation_quant is True:
        return {}
    valid_input_scales = {
        name: scale for name, scale in input_scales.items() if math.isfinite(scale) and scale > 0
    }
    if dynamic_activation_quant is False:
        missing = sorted(selected - valid_input_scales.keys())
        if missing:
            raise ValueError(
                "vae_quant_config input_activations.dynamic=false requires calibrated "
                "finite positive input scales for every selected FP4 convolution; "
                f"missing or invalid: {missing}"
            )
    return valid_input_scales


def _load_nvfp4_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    *,
    enable_fp4: bool = True,
    quant_config: QuantConfig | None = None,
    dynamic_weight_quant: bool | None = None,
    dynamic_activation_quant: bool | None = None,
) -> nn.Module:
    """Load ModelOpt weights and replace their quantized Conv3d modules.

    Quantized weights are first dequantized into the native state dict, then
    requantized in the KTRSC layout consumed by the CuTe kernel. Consequently,
    this path is accuracy-gated as a TensorRT-LLM implementation rather than
    expected to preserve ModelOpt's original packed bytes.
    """
    from .wan_vae import WanCausalConv3d, dequant_fp4_conv_weight, swap_wan_convs_to_fp4

    vae_dir = Path(checkpoint_dir) / "vae"
    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    raw_state_dict = WeightLoader(components=PipelineComponent.VAE).load_weights(
        checkpoint_dir,
        Mapping(),
    )
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
    selected = {
        name
        for name in quantized
        if quant_config is None or not quant_config.is_module_excluded_from_quantization(name)
    }
    if enable_fp4:
        _validate_dynamic_weight_request(
            checkpoint_is_fp4=True,
            selected=selected,
            dynamic_weight_quant=dynamic_weight_quant,
        )
        input_scales = _resolve_input_scales(
            selected,
            input_scales,
            dynamic_activation_quant,
        )
        enable_fp4 = _nvfp4_enabled_for_device(
            device,
            selected,
            explicit_request=quant_config is not None,
        )
    _warn_dequantized_nvfp4_weights(quantized, selected, enable_fp4)
    n = n_static = 0
    if enable_fp4:
        n, n_static = swap_wan_convs_to_fp4(wan_vae, input_scales, only_names=selected)
    if n != len(selected) and enable_fp4:
        raise ValueError(
            f"NVFP4 checkpoint selects {len(selected)} quantized convolutions, "
            f"but only {n} are kernel-compatible Wan convolutions"
        )
    if enable_fp4:
        logger.info(
            f"Loaded NVFP4 Wan VAE: {len(quantized)} quantized convs; "
            f"{n} run on the FP4 kernel ({n_static} static, {n - n_static} dynamic); "
            f"{len(quantized) - len(selected)} excluded by vae_quant_config."
        )
    else:
        logger.info("Loaded the NVFP4 Wan VAE checkpoint as dequantized BF16 modules.")
    return wan_vae


def _load_native_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> WanVAE:
    vae_dir = Path(checkpoint_dir) / "vae"
    wan_vae = WanVAE(WanVAEConfig.from_json_file(vae_dir / "config.json"))
    state_dict = WeightLoader(components=PipelineComponent.VAE).load_weights(
        checkpoint_dir,
        Mapping(),
    )
    wan_vae.load_state_dict(state_dict, strict=True)
    return wan_vae.to(device=device, dtype=dtype).eval()


def _select_dynamic_fp4_convs(vae: WanVAE, quant_config: QuantConfig) -> set[str]:
    """Select dynamic-FP4 Conv3d modules by kernel geometry and channel shape."""
    from .wan_vae import NVFP4WanCausalConv3d, _supports_nvfp4_conv3d

    selected: set[str] = set()
    for name, module in vae.named_modules():
        if (
            not isinstance(module, NVFP4WanCausalConv3d)
            and _supports_nvfp4_conv3d(module)
            # Small VAE boundary projections have little arithmetic payoff
            # and are disproportionately sensitive to uncalibrated FP4.
            and module.in_channels >= _NVFP4_DYNAMIC_MIN_CHANNELS
            and module.out_channels >= _NVFP4_DYNAMIC_MIN_CHANNELS
            and not quant_config.is_module_excluded_from_quantization(name)
        ):
            selected.add(name)
    return selected


def load_wan_vae(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    quant_config: QuantConfig | None = None,
    dynamic_weight_quant: bool | None = None,
    dynamic_activation_quant: bool | None = None,
) -> nn.Module:
    """Load the native or Diffusers Wan VAE with optional NVFP4 execution.

    Args:
        checkpoint_dir: Pipeline checkpoint root containing the ``vae`` component.
        device: Device on which the loaded VAE will execute.
        dtype: Floating-point dtype for VAE parameters and activations.
        quant_config: Optional VAE quantization selection and exclusion rules.
            ``None`` follows the checkpoint's quantization metadata.
        dynamic_weight_quant: Optional override for runtime weight quantization.
        dynamic_activation_quant: Optional override for runtime activation scales.

    Returns:
        The loaded native ``WanVAE`` or Diffusers ``AutoencoderKLWan`` module.

    Raises:
        ValueError: If the requested quantization mode, checkpoint, device, or
            backend combination is unsupported.
    """
    requested_algo = quant_config.quant_algo if quant_config is not None else None
    if requested_algo not in (None, QuantAlgo.NVFP4):
        raise ValueError(
            f"Wan VAE supports only NVFP4 quantization, got {requested_algo}. "
            "Use quant_config for transformer quantization."
        )
    if (
        dynamic_weight_quant is not None or dynamic_activation_quant is not None
    ) and requested_algo != QuantAlgo.NVFP4:
        raise ValueError("VAE dynamic quantization settings require quant_algo=NVFP4")

    if not _use_native_wan_vae():
        if requested_algo == QuantAlgo.NVFP4:
            raise ValueError(
                f"NVFP4 VAE requires the native Wan VAE; unset {TRTLLM_USE_DIFFUSER_VAE_ENV}."
            )
        return AutoencoderKLWan.from_pretrained(
            checkpoint_dir,
            subfolder="vae",
            torch_dtype=dtype,
        ).to(device)

    vae_dir = Path(checkpoint_dir) / "vae"
    checkpoint_is_fp4 = _is_nvfp4_vae_ckpt(vae_dir)
    # None means checkpoint-driven selection. An explicit QuantConfig with no
    # quant_algo requests BF16, including dequantizing a packed FP4 checkpoint.
    enable_fp4 = checkpoint_is_fp4 if quant_config is None else requested_algo == QuantAlgo.NVFP4

    if checkpoint_is_fp4:
        return _load_nvfp4_wan_vae(
            checkpoint_dir,
            device,
            dtype,
            enable_fp4=enable_fp4,
            quant_config=quant_config,
            dynamic_weight_quant=dynamic_weight_quant,
            dynamic_activation_quant=dynamic_activation_quant,
        )

    wan_vae = _load_native_wan_vae(checkpoint_dir, device, dtype)
    if enable_fp4:
        from .wan_vae import swap_wan_convs_to_fp4

        selected = _select_dynamic_fp4_convs(wan_vae, quant_config)
        _validate_dynamic_weight_request(
            checkpoint_is_fp4=False,
            selected=selected,
            dynamic_weight_quant=dynamic_weight_quant,
        )
        _resolve_input_scales(selected, {}, dynamic_activation_quant)
        # The explicit request makes this a validation-only call: unsupported
        # devices raise instead of selecting the BF16 fallback.
        _nvfp4_enabled_for_device(device, selected, explicit_request=True)
        n, n_static = swap_wan_convs_to_fp4(wan_vae, only_names=selected)
        if n != len(selected):
            raise ValueError(
                f"vae_quant_config selected {len(selected)} convolutions, "
                f"but only {n} were replaced"
            )
        logger.info(
            f"Prepared {n} Wan VAE convolutions for one-time weight quantization "
            "during warmup (dynamic activations)."
        )
        if n_static:
            raise RuntimeError("Load-time NVFP4 quantization unexpectedly produced static scales")
    return wan_vae
