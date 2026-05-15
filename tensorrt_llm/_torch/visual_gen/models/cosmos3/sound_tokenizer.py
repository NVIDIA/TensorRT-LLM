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
from functools import partial
from typing import Any, Callable, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrize import remove_parametrizations

from tensorrt_llm.logger import logger

from .modules import (
    ConvNeXtBlock,
    SConv1d,
    SConvTranspose1d,
    SnakeBeta,
    VAEBottleneck,
    WNConv1d,
    WNConvTranspose1d,
)


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

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=self.padding,
                padding_mode=self.padding_mode,
            ),
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=out_channels,
            ),
            WNConv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
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

        # apply conv layers
        x = self.layers(x)

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

        self.layers = nn.Sequential(
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=in_channels,
            ),
            self._create_upsample_layer(
                in_channels, out_channels, stride, use_nearest_upsample, causal, padding_mode
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=1,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=3,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
            ResidualUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                dilation=9,
                use_snake=use_snake,
                causal=causal,
                padding_mode=padding_mode,
            ),
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
        return self.layers(x)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization from all layers."""

        for layer in self.layers:
            try:
                remove_weight_norm(layer)
            except (ValueError, AttributeError):
                # Layer doesn't have weight norm or is not a module with weight norm
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

        # Padding for the first convolution layer
        self.first_padding = 6 if causal else 3
        first_conv = WNConv1d(
            in_channels=latent_dim,
            out_channels=c_mults[-1] * channels,
            kernel_size=7,
            padding=self.first_padding,
            padding_mode=padding_mode,
        )

        if causal:
            first_conv = nn.Sequential(first_conv, TrimPadding(self.first_padding))

        layers = [first_conv]

        for i in range(self.depth - 1, 0, -1):
            layers += [
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

        # Padding for the final convolution layer
        self.final_padding = 6 if causal else 3
        final_conv = WNConv1d(
            in_channels=c_mults[0] * channels,
            out_channels=out_channels,
            kernel_size=7,
            padding=self.final_padding,
            padding_mode=padding_mode,
            bias=False,
        )

        if causal:
            final_conv = nn.Sequential(final_conv, TrimPadding(self.final_padding))

        layers += [
            get_activation(
                "snake" if use_snake else "elu",
                antialias=antialias_activation,
                channels=c_mults[0] * channels,
            ),
            final_conv,
            nn.Tanh() if final_tanh else nn.Identity(),
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self: "OobleckDecoder", x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
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


class SpectrogramConvNeXtEncoder(nn.Module):
    """
    Spectrogram Encoder with ConvNeXtBlocks

    This encoder processes input waveforms by converting them into spectrograms
    (magnitude and phase concatenated along the channel dimension) and encodes them
    using a sequence of ConvNeXtBlocks and downsampling layers.

    Args (mapped from h):
        in_channels (int): Number of input audio channels (1 for mono, 2 for stereo).
        channels (int): Base number of channels for the encoder.
        latent_dim (int): Dimensionality of the final latent representation.
        c_mults (List[int]): Channel multipliers at each depth of the encoder.
        strides (List[int]): Downsampling strides for each depth.
        num_blocks (int): Number of ConvNeXtBlocks to stack per depth.
        identity_init (bool): Whether to initialize the 1x1 convs in residual paths as zeros.
        n_fft (int): Number of FFT points for spectrogram computation.
        hop_length (int): Hop length for the STFT.
        use_snake (bool): Whether to use Snake activation in ConvNeXtBlocks.
        causal (bool): If True, uses causal convolutions.
        padding_mode (str): Padding mode for convolutions (default: 'zeros').

    Inputs:
        x (torch.Tensor): Input waveform tensor of shape `[batch, in_channels, time]`.

    Outputs:
        torch.Tensor: Encoded representation of shape `[batch, time_out, latent_dim]`.

    Forward Pass:
        - Converts waveform input into spectrograms (concatenates magnitude and phase).
        - Processes the spectrogram through stacked ConvNeXtBlocks and downsampling layers.
        - Outputs the final latent representation of specified dimensionality.

    Example:
        encoder = SpectrogramConvNeXtEncoder(
            in_channels=2, channels=256, latent_dim=128, c_mults=[1, 2, 4], strides=[4, 4, 8]
        )
        waveform = torch.randn(8, 2, 65536)  # [batch, channels, time]
        encoded = encoder(waveform)  # Output: [8, time_out, 128]

    NOTE: output is in [B, T, C] to be consistent with other encoders
    """

    def __init__(self, model_config: Dict[str, Any]) -> None:
        super().__init__()
        self.model_config = model_config

        self.in_channels = model_config["input_channels"]
        if model_config.get("stereo", False):
            self.in_channels *= 2

        # if "enc_latent_dim" is found in v2 config, set it as latent_dim
        if "enc_latent_dim" in model_config:
            self.latent_dim = model_config["enc_latent_dim"]
        else:
            # if not found, fallback to v1 logic
            self.latent_dim = model_config["vocoder_input_dim"]
            if model_config["model_type"] == "vae":
                self.latent_dim *= 2

        self.channels = model_config["enc_dim"]

        self.c_mults = model_config["enc_c_mults"]
        self.strides = model_config["enc_strides"]
        self.num_blocks = model_config["enc_num_blocks"]
        self.identity_init = model_config["enc_identity_init"]
        self.causal = model_config["causal"]
        self.padding_mode = model_config["padding_mode"]

        self.use_snake = model_config["enc_use_snake"]

        # Basic checks
        assert len(self.c_mults) == len(self.strides), (
            f"The length of c_mults and strides must match. Got {len(self.c_mults)} vs {len(self.strides)}."
        )

        # Spectrogram function
        self.n_fft = model_config["enc_n_fft"]
        self.hop_length = model_config["enc_hop_length"]
        self.spectrogram_fn = partial(
            self.spectrogram,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window_fn=torch.hann_window,
        )

        # ---------------------------------------------------------------------
        # 1) Initial projection (similar to the first_conv in OobleckEncoder),
        #    but here we typically use a 1x1 conv for a "spectrogram style" input.
        # ---------------------------------------------------------------------
        layers = []
        layers.append(
            WNConv1d(
                (self.n_fft + 2) * self.in_channels,
                self.c_mults[0] * self.channels,
                kernel_size=1,
                bias=False,
            )
        )

        # ---------------------------------------------------------------------
        # 2) Stages: For each i in range(len(c_mults)):
        #       - Stack num_blocks of ConvNeXtBlock
        #       - Downsample via stride convolution
        # ---------------------------------------------------------------------
        for i in range(len(self.c_mults)):
            dim_in = self.c_mults[i] * self.channels
            # Determine output dimension for the block
            if i < len(self.c_mults) - 1:  # If not the last block
                dim_out = self.c_mults[i + 1] * self.channels
            else:  # For the last block, dim_out is c_mults[-1] * channels
                dim_out = self.c_mults[-1] * self.channels
            ds_rate = self.strides[i]

            # (a) Repeated ConvNeXtBlocks
            for _ in range(self.num_blocks):
                layers.append(
                    ConvNeXtBlock(
                        dim=dim_in,
                        intermediate_dim=dim_in * 4,
                        identity_init=self.identity_init,
                        use_snake=self.use_snake,
                        causal=self.causal,
                    )
                )

            # (b) Downsampling convolution
            layers.append(
                self._create_downsample_layer(
                    dim_in, dim_out, ds_rate, self.causal, self.padding_mode
                )
            )

        # ---------------------------------------------------------------------
        # 3) Final projection from the last channel dimension to latent_dim.
        # ---------------------------------------------------------------------
        layers.append(
            WNConv1d(self.c_mults[-1] * self.channels, self.latent_dim, kernel_size=1, bias=False)
        )

        self.layers = nn.Sequential(*layers)

    def spectrogram(
        self: "SpectrogramConvNeXtEncoder",
        wav: Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
    ) -> Tensor:
        """
        wav: [batch_size?, time_steps], where batch_size? is an optional batch dimension
        """
        pad_size_l = (n_fft - hop_length) // 2
        pad_size_r = (n_fft - hop_length) - pad_size_l
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav = F.pad(wav, (pad_size_l, pad_size_r)).float()
            spec = torch.stft(
                wav,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window_fn(win_length).to(wav),
                center=False,
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        return spec

    def _create_downsample_layer(
        self: "SpectrogramConvNeXtEncoder",
        in_channels: int,
        out_channels: int,
        stride: int,
        causal: bool,
        padding_mode: str,
    ) -> nn.Module:
        if causal:
            downsample_layer = SConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                causal=True,
                norm="weight_norm",
            )
        else:  # original non-causal implementation
            downsample_layer = WNConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                padding_mode=padding_mode,
            )
        return downsample_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: waveform in [batch, in_channels, length] (mono: in_channels=1, stereo: in_channels=2)
        Returns: encoder output in [batch, length_out, dim_latent],
        where the spectrogram's magnitude and phase are concatenated along the channel dimension.
        """

        # Handle stereo input by merging channel dim into batch dim
        batch, channels, length = x.shape
        if channels > 1:  # Stereo case
            x = x.reshape(batch * channels, 1, length)  # [batch * channels, 1, length]

        # Compute the spectrogram
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram_fn(
                x.float().squeeze(1)
            )  # Remove the channel dimension for STFT
            mag, ph = torch.view_as_real(spec).chunk(2, dim=-1)  # Split real and imaginary parts
            spectrogram = torch.cat([mag, ph], dim=1).squeeze(
                -1
            )  # Concatenate along channel dim: [batch * channels, freq, frame]

        # Cast spectrogram back to original dtype
        spectrogram = spectrogram.to(x.dtype)

        # Restore stereo structure if needed
        if channels > 1:  # Stereo case
            freq = spectrogram.shape[1]  # Get the frequency dimension
            spectrogram = spectrogram.reshape(
                batch, channels * freq, *spectrogram.shape[2:]
            )  # [batch, freq * channels, frame]

        # forward pass the encoder
        output = self.layers(spectrogram)

        return output.transpose(1, 2)  # [B, T, C]

    def remove_weight_norm(self: "SpectrogramConvNeXtEncoder") -> None:
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
        h: Configuration object containing model hyperparameters.
        encoder (nn.Module): The encoder module based on configuration.
        bottleneck (VAEBottleneck): VAE Bottleneck module.
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

        # hop_size defines the down/up sampling factor of the autoencoder
        self.hop_size = model_config["hop_size"]

        # Initialize encoder
        self.enc_type = model_config.get("enc_type", "convnext")

        # Define encoder (only spec_convnext supported in cleaned version)
        if self.enc_type == "spec_convnext":
            self.encoder = SpectrogramConvNeXtEncoder(model_config)
        else:
            raise NotImplementedError(
                f"Encoder type '{self.enc_type}' not supported in cleaned AVAE. Only 'spec_convnext' is supported."
            )

        # Initialize encoder projector (Identity for spec_convnext)
        self.encoder_proj = nn.Identity()

        if "bottleneck" in model_config:
            self.bottleneck = VAEBottleneck()
        else:
            raise ValueError("Bottleneck configuration must be specified")

        # Check for encoder-only mode
        self.encoder_only = model_config.get("encoder_only", False)

        if not self.encoder_only:
            # Initialize decoder
            self.dec_type = model_config.get("dec_type", "oobleck")
            if self.dec_type == "oobleck":
                self.decoder = OobleckDecoder(model_config)
            else:
                raise NotImplementedError(
                    f"Decoder type '{self.dec_type}' not supported in cleaned AVAE. Only 'oobleck' is supported."
                )
        else:
            # Skip decoder initialization
            self.decoder = None

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

        # --- weight loading (mirrors cosmos3-internal AVAEModel._load_avae_model) ---
        state_dict: Optional[Dict[str, Any]] = None

        # 1. safetensors (standard TRT-LLM / HF format)
        sft_candidates = ["model.safetensors", "diffusion_pytorch_model.safetensors"]
        for name in sft_candidates:
            path = os.path.join(checkpoint_dir, name)
            if os.path.exists(path):
                from safetensors.torch import load_file

                state_dict = load_file(path, device="cpu")
                break

        # 2. PyTorch bin
        if state_dict is None:
            bin_candidates = ["pytorch_model.bin", "diffusion_pytorch_model.bin"]
            for name in bin_candidates:
                path = os.path.join(checkpoint_dir, name)
                if os.path.exists(path):
                    state_dict = torch.load(path, map_location="cpu", weights_only=True)
                    break

        if state_dict is None:
            raise FileNotFoundError(
                f"No weight file found in '{checkpoint_dir}'. "
                "Expected model.safetensors, pytorch_model.bin, or *.ckpt."
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

    def calculate_latent_lengths(
        self: "LatentAutoEncoderV2", audio_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the latent lengths given the original audio lengths.

        Args:
            audio_lengths (torch.Tensor): A tensor of shape [B] containing the lengths of the original audio samples.

        Returns:
            torch.Tensor: A tensor of shape [B] containing the corresponding latent lengths.
        """
        if self.input_type == "waveform":
            # The latent length is the audio length divided by the hop_size
            latent_lengths = torch.ceil(audio_lengths.float() / self.hop_size).long()
        else:
            # The latent length is same as audio_lengths
            latent_lengths = audio_lengths

        return latent_lengths

    def forward(self: "LatentAutoEncoderV2", x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model with shape [B, C, T].

        Returns:
            dict[str, torch.Tensor]: Dictionary of output tensors including:
                - encoder_out: Raw encoder output
                - latent: Bottleneck latent representation
                - decoder_out: Decoded output (if decoder exists)
                - Additional outputs specific to the bottleneck type
        """
        return_dict = {}

        # Encoder
        encoder_out = self.encoder(x)  # Shape: [B, T_frame, encoder_out_dim]
        encoder_out_proj = self.encoder_proj(encoder_out)  # Shape: [B, T_frame, encoder_proj_dim]

        # Apply bottleneck after reshaping to [B, C, T] again
        latent, bottleneck_enc_info = self.bottleneck.encode(
            encoder_out_proj.transpose(1, 2), return_info=True
        )

        # Update return dictionary
        return_dict.update({"encoder_out": encoder_out.transpose(1, 2), "latent": latent})
        # Add bottleneck-specific info to return dict
        for k, v in bottleneck_enc_info.items():
            return_dict[k] = v

        # Decode (if decoder exists)
        if self.decoder is not None:
            # Apply bottleneck decode
            decoded_latent, bottleneck_dec_info = self.bottleneck.decode(latent, return_info=True)
            # Apply decoder
            decoder_out = self.decoder(decoded_latent)

            # Update return dictionary
            return_dict["decoder_out"] = decoder_out
            # Add bottleneck-specific info to return dict
            for k, v in bottleneck_dec_info.items():
                return_dict[k] = v

        return return_dict

    def encode(self: "LatentAutoEncoderV2", x: torch.Tensor) -> torch.Tensor:
        """
        Encode waveform to latent tokens.

        Args:
            x: Input tensor with shape [B, C, T].

        Returns:
            Latent tensor [B, latent_ch, T_latent]. Normalised by latent_mean/std
            when configured (mirrors cosmos3-internal AVAEModel.encode).
        """
        encoder_out = self.encoder(x)
        encoder_out_proj = self.encoder_proj(encoder_out)
        latent = self.bottleneck.encode(encoder_out_proj.transpose(1, 2))

        if self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean) / self.latent_std

        return latent

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

        decoded_latent = self.bottleneck.decode(latent)
        return self.decoder(decoded_latent)

    @property
    def audio_channels(self) -> int:
        """Number of output audio channels (1 = mono, 2 = stereo)."""
        return self.model_config.get("dec_out_channels", 1)

    def remove_weight_norm(self: "LatentAutoEncoderV2") -> None:
        """Remove weight normalization from all components."""
        self.encoder.remove_weight_norm()
        if self.decoder is not None:
            self.decoder.remove_weight_norm()
