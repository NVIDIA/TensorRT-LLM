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


def get_activation(
    activation: Literal["elu", "snake", "none"],
    antialias: bool = False,
    channels: Optional[int] = None,
    use_cuda_kernel: bool = False,
) -> nn.Module:
    """
    Get activation module by name.

    Args:
        activation: Activation type ('elu', 'snake', or 'none')
        antialias: Whether to wrap with anti-aliasing
        channels: Number of channels (required for snake activation)
        use_cuda_kernel: Whether to use CUDA kernel (not supported)

    Returns:
        Activation module
    """
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")

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

        self.snake1 = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=out_channels,
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
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=out_channels,
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
        res = x

        x = self.conv1(self.snake1(x))
        x = self.conv2(self.snake2(x))

        if self.causal:
            # Trim right padding to get the causal output
            x = x[:, :, : -self.padding]

        return x + res


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
    ) -> None:
        super().__init__()

        self.causal = causal

        self.snake1 = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=in_channels,
        )
        self.conv_t1 = self._create_upsample_layer(
            in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode
        )
        self.res_unit1 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=1,
            use_snake=use_snake,
            causal=causal,
            padding_mode=padding_mode,
        )
        self.res_unit2 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=3,
            use_snake=use_snake,
            causal=causal,
            padding_mode=padding_mode,
        )
        self.res_unit3 = ResidualUnit(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=9,
            use_snake=use_snake,
            causal=causal,
            padding_mode=padding_mode,
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
        x = self.conv_t1(self.snake1(x))
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x

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

        out_channels = model_config["input_channels"]
        if model_config.get("stereo", False):
            out_channels *= 2

        channels = model_config["dec_dim"]
        c_mults = model_config["dec_c_mults"]
        strides = model_config["dec_strides"]
        use_snake = model_config["dec_use_snake"]
        use_nearest_upsample = model_config["dec_use_nearest_upsample"]
        antialias_activation = model_config["dec_anti_aliasing"]
        causal = model_config["causal"]
        final_tanh = model_config["dec_use_tanh_at_final"]
        padding_mode = model_config["padding_mode"]

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
                    use_snake=use_snake,
                    antialias_activation=antialias_activation,
                    use_nearest_upsample=use_nearest_upsample,
                    causal=causal,
                    padding_mode=padding_mode,
                )
            ]
        self.block = nn.ModuleList(blocks)

        self.final_padding = 6 if causal else 3
        self.snake1 = get_activation(
            "snake" if use_snake else "elu",
            antialias=antialias_activation,
            channels=c_mults[0] * channels,
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
        x = self.conv1(x)
        x = self.conv1_trim(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        x = self.conv2(x)
        x = self.conv2_trim(x)
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
    A Latent AutoEncoder class with cleaner implementation to generalize using bottleneck.py

    Attributes:
        model_config: Configuration object containing model hyperparameters.
        decoder (nn.Module): The decoder module based on configuration.
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        self.model_config = model_config

        # Set up basic model properties
        self.stereo = model_config.get("stereo", False)

        # Determine input type
        self.input_type = None
        if model_config.get("use_wav_as_input", False):
            self.input_type = "waveform"
            model_config["input_channels"] = 1
        elif model_config.get("use_linear_spec_as_input", False):
            self.input_type = "linear"
            model_config["input_channels"] = model_config["num_linears"]
        elif model_config.get("use_discrete_code_as_input", False):
            self.input_type = "discrete_code"
            model_config["input_channels"] = 1
        else:
            self.input_type = "mel"
            model_config["input_channels"] = model_config["num_mels"]

        # Check for encoder-only mode
        self.encoder_only = model_config.get("encoder_only", False)

        if self.encoder_only:
            raise NotImplementedError("Encoder-only mode not supported")

        self.dec_type = model_config.get("dec_type", "oobleck")
        if self.dec_type == "oobleck":
            self.decoder = OobleckDecoder(model_config)
        else:
            raise NotImplementedError(
                f"Decoder type '{self.dec_type}' not supported in cleaned AVAE. Only 'oobleck' is supported."
            )

        # Optional latent normalisation (from cosmos3-internal AVAEModel)
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

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading sound tokenizer: {missing}")
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

        return self.decoder(latent)

    def remove_weight_norm(self: "LatentAutoEncoderV2") -> None:
        """Remove weight normalization from all components."""
        if self.decoder is not None:
            self.decoder.remove_weight_norm()
