# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any, Dict, Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from tensorrt_llm.logger import logger

from .modules import SConvTranspose1d, SnakeBeta, WNConv1d, WNConvTranspose1d


def _resolve_activation_name(
    model_config: Dict[str, Any], use_snake: bool
) -> Literal["elu", "snakebeta", "none"]:
    if not use_snake:
        return "elu"
    activation = model_config.get("activation", "snakebeta")
    if activation in ("snake", "snakebeta"):
        return "snakebeta"
    if activation == "none":
        return "none"
    raise ValueError(f"Unknown activation {activation}")


def _resolve_decoder_out_channels(model_config: Dict[str, Any]) -> int:
    if "dec_out_channels" in model_config:
        return model_config["dec_out_channels"]
    out_channels = model_config["input_channels"]
    if model_config.get("stereo", False):
        out_channels *= 2
    return out_channels


def _extract_decoder_state_dict(
    state_dict: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """Return checkpoint weights keyed for ``LatentAutoEncoderV2.decoder``."""
    prefixed = {key: value for key, value in state_dict.items() if key.startswith("decoder.")}
    if prefixed:
        return prefixed

    # Legacy checkpoints may omit the decoder. prefix.
    return {f"decoder.{key}": value for key, value in state_dict.items()}


def get_activation(
    activation: Literal["elu", "snake", "snakebeta", "none"],
    antialias: bool = False,
    channels: Optional[int] = None,
    use_cuda_kernel: bool = False,
    snake_logscale: bool = True,
) -> nn.Module:
    """
    Get activation module by name.

    Args:
        activation: Activation type ('elu', 'snakebeta', or 'none')
        antialias: Whether to wrap with anti-aliasing
        channels: Number of channels (required for snake activation)
        use_cuda_kernel: Whether to use CUDA kernel (not supported)
        snake_logscale: Whether SnakeBeta uses log-scaled parameters

    Returns:
        Activation module
    """
    if activation == "elu":
        act = nn.ELU()
    elif activation in ("snake", "snakebeta"):
        if channels is None:
            raise ValueError("channels is required for snake activation")
        act = SnakeBeta(channels, alpha_logscale=snake_logscale)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

    if use_cuda_kernel:
        raise NotImplementedError("CUDA kernel activation not supported")

    if antialias:
        raise NotImplementedError("antialias activation not supported")

    return act


class ResidualUnit(nn.Module):
    """
    Residual unit with dilated convolutions.
    Used in OobleckDecoderBlock.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate
        kernel_size: Convolution kernel size (default: 7)
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliasing (default: False)
        causal: Whether to use causal convolutions (default: False)
        padding_mode: Padding mode for convolutions (default: 'zeros')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
        kernel_size: int = 7,
        use_snake: bool = False,
        antialias_activation: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
        activation: Literal["elu", "snakebeta", "none"] = "elu",
        snake_logscale: bool = True,
        use_cuda_kernel: bool = False,
    ) -> None:
        super().__init__()

        self.dilation = dilation
        self.causal = causal
        self.kernel_size = kernel_size

        if causal:
            self.padding = dilation * (kernel_size - 1)
        else:
            self.padding = (dilation * (kernel_size - 1)) // 2

        self.padding_mode = padding_mode
        activation_name = activation if use_snake else "elu"

        self.snake1 = get_activation(
            activation_name,
            antialias=antialias_activation,
            channels=out_channels,
            snake_logscale=snake_logscale,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.conv1 = WNConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
            padding_mode=self.padding_mode,
        )
        self.snake2 = get_activation(
            activation_name,
            antialias=antialias_activation,
            channels=out_channels,
            snake_logscale=snake_logscale,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.conv2 = WNConv1d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T)
        """
        output_tensor = self.conv1(self.snake1(x))
        output_tensor = self.conv2(self.snake2(output_tensor))

        if self.causal:
            output_tensor = output_tensor[:, :, : -self.padding]
            res = x[:, :, : -self.padding]
            return res + output_tensor

        res = x
        padding = (res.shape[-1] - output_tensor.shape[-1]) // 2
        if padding > 0:
            res = res[..., padding:-padding]
        return res + output_tensor


class OobleckDecoderBlock(nn.Module):
    """
    Oobleck decoder block with upsampling and residual units.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Upsampling stride
        use_snake: Whether to use Snake activation (default: False)
        antialias_activation: Whether to use anti-aliasing (default: False)
        use_nearest_upsample: Whether to use nearest neighbor upsampling (default: False)
        causal: Whether to use causal convolutions (default: False)
        padding_mode: Padding mode for convolutions (default: 'zeros')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_snake: bool = False,
        antialias_activation: bool = False,
        use_nearest_upsample: bool = False,
        causal: bool = False,
        padding_mode: str = "zeros",
        activation: Literal["elu", "snakebeta", "none"] = "elu",
        snake_logscale: bool = True,
        use_cuda_kernel: bool = False,
    ) -> None:
        super().__init__()

        self.causal = causal
        activation_name = activation if use_snake else "elu"

        self.snake1 = get_activation(
            activation_name,
            antialias=antialias_activation,
            channels=in_channels,
            snake_logscale=snake_logscale,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.conv_t1 = self._create_upsample_layer(
            in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode
        )
        res_unit_kwargs = {
            "use_snake": use_snake,
            "causal": causal,
            "padding_mode": padding_mode,
            "activation": activation,
            "snake_logscale": snake_logscale,
            "use_cuda_kernel": use_cuda_kernel,
            "antialias_activation": antialias_activation,
        }
        self.res_unit1 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=1,
            **res_unit_kwargs,
        )
        self.res_unit2 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=3,
            **res_unit_kwargs,
        )
        self.res_unit3 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=9,
            **res_unit_kwargs,
        )

    def _create_upsample_layer(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        use_nearest_upsample: bool,
        causal: bool,
        padding_mode: str,
    ) -> nn.Module:
        """
        Create upsampling layer based on configuration.

        Note: padding_mode parameter is not used in this function.
        """

        if (
            causal
        ):  # use EnCodec's SConvTransposed1d for convenience. padding_mode is reflect by default
            assert not use_nearest_upsample, (
                "use_nearest_upsample is not implemented for causal mode!"
            )
            upsample_layer = SConvTranspose1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                causal=True,
                norm="weight_norm",
            )
        else:
            if use_nearest_upsample:
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=stride, mode="nearest"),
                    WNConv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2 * stride,
                        stride=1,
                        bias=False,
                        padding="same",
                    ),
                )
            else:
                # WNConvTranspose1d only supports zeros padding mode so it's hardcoded
                upsample_layer = WNConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                    output_padding=stride % 2,
                    padding_mode="zeros",
                )

        return upsample_layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, T)

        Returns:
            Output tensor of shape (B, C, T_upsampled)
        """
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all layers."""
        for layer in [self.conv_t1, self.res_unit1, self.res_unit2, self.res_unit3]:
            try:
                remove_weight_norm(layer)
            except (ValueError, AttributeError):
                pass


class TrimPadding(nn.Module):
    """
    Used for causal convolution support of a conv layer wrapped with nn.Sequential
    """

    def __init__(self, padding: int) -> None:
        super().__init__()
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.padding]


class OobleckDecoder(nn.Module):
    """
    Oobleck Decoder for audio synthesis.

    Decodes latent representations into audio waveforms using
    upsampling blocks with optional Snake activation and anti-aliasing.
    """

    def __init__(
        self: "OobleckDecoder",
        model_config: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.model_config = model_config

        latent_dim = model_config["vocoder_input_dim"]
        out_channels = _resolve_decoder_out_channels(model_config)

        channels = model_config["dec_dim"]
        c_mults = model_config["dec_c_mults"]
        strides = model_config["dec_strides"]
        use_snake = model_config["dec_use_snake"]
        use_nearest_upsample = model_config["dec_use_nearest_upsample"]
        antialias_activation = model_config["dec_anti_aliasing"]
        causal = model_config["causal"]
        final_tanh = model_config["dec_use_tanh_at_final"]
        padding_mode = model_config["padding_mode"]
        snake_logscale = model_config.get("snake_logscale", True)
        use_cuda_kernel = model_config.get("use_cuda_kernel", False)
        activation = _resolve_activation_name(model_config, use_snake)
        block_kwargs = {
            "use_snake": use_snake,
            "antialias_activation": antialias_activation,
            "use_nearest_upsample": use_nearest_upsample,
            "causal": causal,
            "padding_mode": padding_mode,
            "activation": activation,
            "snake_logscale": snake_logscale,
            "use_cuda_kernel": use_cuda_kernel,
        }

        c_mults = [1, *c_mults]

        self.depth = len(c_mults)

        self.first_padding = 6 if causal else 3
        self.conv1 = WNConv1d(
            in_channels=latent_dim,
            out_channels=c_mults[-1] * channels,
            kernel_size=7,
            padding=self.first_padding,
            padding_mode=padding_mode,
        )
        self.conv1_trim = TrimPadding(self.first_padding) if causal else nn.Identity()

        blocks = []
        for i in range(self.depth - 1, 0, -1):
            blocks += [
                OobleckDecoderBlock(
                    in_channels=c_mults[i] * channels,
                    out_channels=c_mults[i - 1] * channels,
                    stride=strides[i - 1],
                    **block_kwargs,
                )
            ]
        self.block = nn.ModuleList(blocks)

        self.final_padding = 6 if causal else 3
        self.snake1 = get_activation(
            activation,
            antialias=antialias_activation,
            channels=c_mults[0] * channels,
            snake_logscale=snake_logscale,
            use_cuda_kernel=use_cuda_kernel,
        )
        self.conv2 = WNConv1d(
            in_channels=c_mults[0] * channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=self.final_padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.conv2_trim = TrimPadding(self.final_padding) if causal else nn.Identity()
        self.final_activation = nn.Tanh() if final_tanh else nn.Identity()

    def forward(self: "OobleckDecoder", x: torch.Tensor) -> torch.Tensor:
        causal = self.model_config.get("causal", False)
        if causal:
            x = self.conv1(x)
            x = self.conv1_trim(x)
            for block in self.block:
                x = block(x)
            x = self.snake1(x)
            x = self.conv2(x)
            x = self.conv2_trim(x)
            return self.final_activation(x)

        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        x = self.conv2(x)
        if not isinstance(self.final_activation, nn.Identity):
            x = self.final_activation(x)
        return x

    def remove_weight_norm(self: "OobleckDecoder") -> None:
        for module in self.modules():
            if hasattr(
                module, "parametrizations"
            ):  # for new WN implementation using parameterizations
                remove_parametrizations(module, "weight")
            elif hasattr(module, "weight"):
                try:
                    remove_weight_norm(module)
                except ValueError:
                    pass


class LatentAutoEncoderV2(nn.Module):
    """
    Decoder-only autoencoder_v2 wrapper for Cosmos3 sound generation.

    Checkpoints store weights under the ``decoder.*`` prefix, e.g.
    ``decoder.block.0.conv_t1.weight_g`` and ``decoder.conv1.bias``.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        self.model_config = model_config
        self.stereo = model_config.get("stereo", False)

        if model_config.get("encoder_only", False):
            raise NotImplementedError("Encoder-only mode not supported")

        dec_type = model_config.get("dec_type", "oobleck")
        if dec_type != "oobleck":
            raise NotImplementedError(
                f"Decoder type '{dec_type}' not supported. Only 'oobleck' is supported."
            )

        self.decoder = OobleckDecoder(model_config)
        self.latent_mean = model_config.get("latent_mean", None)
        self.latent_std = model_config.get("latent_std", None)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        subfolder: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> "LatentAutoEncoderV2":
        if subfolder is not None:
            checkpoint_dir = os.path.join(checkpoint_dir, subfolder)

        with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(config)
        state_dict: Optional[Dict[str, Any]] = None

        for name in ["diffusion_pytorch_model.safetensors"]:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                from safetensors.torch import load_file

                state_dict = load_file(path, device="cpu")
                break

        if state_dict is None:
            raise FileNotFoundError(
                f"No weight file found in '{checkpoint_dir}'. "
                "Expected diffusion_pytorch_model.safetensors."
            )

        decoder_state = _extract_decoder_state_dict(state_dict)
        if not decoder_state:
            raise FileNotFoundError(
                f"No decoder weights found in '{checkpoint_dir}'. "
                "Expected keys prefixed with 'decoder.'."
            )

        missing, unexpected = model.load_state_dict(decoder_state, strict=False)
        decoder_missing = [key for key in missing if key.startswith("decoder.")]
        if decoder_missing:
            raise RuntimeError(
                f"Failed to load sound tokenizer decoder weights. Missing keys: {decoder_missing}"
            )
        if unexpected:
            logger.warning(f"Unexpected keys when loading sound tokenizer: {unexpected}")

        # Must remove weight norm AFTER loading the weight_g / weight_v parameters
        model.remove_weight_norm()

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        if dtype is not None:
            model = model.to(dtype=dtype)
        if device is not None:
            model = model.to(device=device)

        return model

    def decode(self: "LatentAutoEncoderV2", latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tokens back to waveform.

        Args:
            latent: Latent tensor [B, latent_ch, T_latent].

        Returns:
            Reconstructed waveform [B, audio_channels, T_samples].
        """
        if self.latent_mean is not None and self.latent_std is not None:
            latent = latent * self.latent_std + self.latent_mean

        return self.decoder(latent).clamp(-1.0, 1.0)

    def remove_weight_norm(self: "LatentAutoEncoderV2") -> None:
        """Remove weight normalization from all components."""
        if self.decoder is not None:
            self.decoder.remove_weight_norm()
